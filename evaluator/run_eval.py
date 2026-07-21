import os
import sys
import json
import time
import argparse
from typing import List, Dict, Any, Optional

# Add the project root directory to sys.path to allow absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Set output encoding to utf-8 for Windows console support
sys.stdout.reconfigure(encoding='utf-8')

try:
    from groq import RateLimitError
except ImportError:
    # Fallback if groq package is modified, though it's installed as verified
    RateLimitError = Exception

from rag_pipeline.retriever import Retriever
from rag_pipeline.generator import Generator
from evaluator.metrics.retrieval_metrics import evaluate_retrieval
from evaluator.metrics.judge_metrics import Judge

def call_with_retry(func, *args, **kwargs):
    """
    Helper function to wrap API calls with retry logic.
    On a 429 RateLimitError, it sleeps for 10 seconds and retries once.
    Other exceptions are propagated immediately.
    """
    try:
        return func(*args, **kwargs)
    except RateLimitError as e:
        print(f"\n[Rate Limit] 429 Rate Limit encountered. Waiting 10 seconds to retry...")
        time.sleep(10)
        try:
            return func(*args, **kwargs)
        except Exception as retry_err:
            print(f"[Retry Failed] Second attempt failed: {retry_err}")
            raise retry_err
    except Exception as e:
        raise e

def compute_metrics_for_subset(subset: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes aggregated statistics (retrieval metrics, faithfulness labels, and
    relevance scores) for a given subset of evaluation results.
    """
    if not subset:
        return {
            "total_questions": 0,
            "retrieval": {
                "average_precision_at_3": None,
                "average_recall_at_3": None,
                "average_hit_at_3": None,
                "average_mrr": None,
                "count": 0
            },
            "faithfulness": {
                "counts": {},
                "percentages": {}
            },
            "relevance": {
                "average_score": None,
                "count": 0
            }
        }

    total_q = len(subset)

    # 1. Retrieval metrics (only computed across answerable questions)
    answerable_subset = [q for q in subset if q["question_type"] != "unanswerable"]
    ret_stats = {
        "average_precision_at_3": None,
        "average_recall_at_3": None,
        "average_hit_at_3": None,
        "average_mrr": None,
        "count": len(answerable_subset)
    }
    
    if answerable_subset:
        sum_p = 0.0
        sum_r = 0.0
        sum_h = 0.0
        sum_mrr = 0.0
        for q in answerable_subset:
            m = q["retrieval_metrics"]
            sum_p += m.get("precision_at_k", 0.0) or 0.0
            sum_r += m.get("recall_at_k", 0.0) or 0.0
            sum_h += 1.0 if m.get("hit_at_k", False) else 0.0
            sum_mrr += m.get("mrr", 0.0) or 0.0
        
        n = len(answerable_subset)
        ret_stats["average_precision_at_3"] = sum_p / n
        ret_stats["average_recall_at_3"] = sum_r / n
        ret_stats["average_hit_at_3"] = sum_h / n
        ret_stats["average_mrr"] = sum_mrr / n

    # 2. Faithfulness metrics
    faithfulness_counts = {
        "fully_grounded": 0,
        "partially_grounded": 0,
        "hallucinated": 0,
        "incorrect_refusal": 0,
        "correct_refusal": 0,
        "incorrect_hallucination": 0
    }
    
    for q in subset:
        label = q["faithfulness"].get("label")
        if label:
            label_cleaned = label.strip().lower()
            if label_cleaned in faithfulness_counts:
                faithfulness_counts[label_cleaned] += 1
            else:
                faithfulness_counts[label_cleaned] = faithfulness_counts.get(label_cleaned, 0) + 1
                
    faithfulness_percentages = {}
    for label, count in faithfulness_counts.items():
        faithfulness_percentages[label] = count / total_q if total_q > 0 else 0.0

    # 3. Relevance metrics
    relevance_scores = []
    for q in subset:
        score = q["relevance"].get("score")
        if score is not None:
            try:
                relevance_scores.append(float(score))
            except (ValueError, TypeError):
                pass
                
    avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else None

    return {
        "total_questions": total_q,
        "retrieval": ret_stats,
        "faithfulness": {
            "counts": faithfulness_counts,
            "percentages": faithfulness_percentages
        },
        "relevance": {
            "average_score": avg_relevance,
            "count": len(relevance_scores)
        }
    }

def print_summary_to_console(summary: Dict[str, Any]) -> None:
    """
    Outputs calculated evaluation summary statistics to the console in a clean format.
    """
    print("\n" + "=" * 80)
    print("                   EVALUATION SUMMARY STATISTICS")
    print("=" * 80)
    
    overall = summary["overall"]
    print(f"Overall Total Questions: {overall['total_questions']}")
    
    ret = overall["retrieval"]
    print("\nRetrieval Metrics (Answerable Questions only):")
    if ret["count"] > 0:
        print(f"  Average Precision@3: {ret['average_precision_at_3']:.4f}")
        print(f"  Average Recall@3   : {ret['average_recall_at_3']:.4f}")
        print(f"  Average Hit@3      : {ret['average_hit_at_3']:.4f}")
        print(f"  Average MRR        : {ret['average_mrr']:.4f} (based on {ret['count']} questions)")
    else:
        print("  No answerable questions evaluated.")
        
    faith = overall["faithfulness"]
    print("\nFaithfulness Labels:")
    for label, count in faith["counts"].items():
        pct = faith["percentages"][label] * 100
        print(f"  {label:<25}: {count:<3} ({pct:.1f}%)")
        
    rel = overall["relevance"]
    print("\nRelevance Metrics:")
    if rel["count"] > 0:
        print(f"  Average Relevance Score: {rel['average_score']:.4f} (based on {rel['count']} questions)")
    else:
        print("  No relevance scores available.")
        
    print("\n" + "-" * 80)
    print("                     BREAKDOWN BY QUESTION TYPE")
    print("-" * 80)
    
    for qtype, stats in summary["by_question_type"].items():
        print(f"\nQuestion Type: {qtype} (N = {stats['total_questions']})")
        
        q_ret = stats["retrieval"]
        if q_ret["count"] > 0:
            print(f"  Retrieval: P@3={q_ret['average_precision_at_3']:.3f}, R@3={q_ret['average_recall_at_3']:.3f}, MRR={q_ret['average_mrr']:.3f}")
        else:
            print("  Retrieval: N/A")
            
        q_faith = stats["faithfulness"]
        faith_str_parts = []
        for label, count in q_faith["counts"].items():
            if count > 0:
                pct = q_faith["percentages"][label] * 100
                faith_str_parts.append(f"{label}: {count} ({pct:.0f}%)")
        if faith_str_parts:
            print(f"  Faithfulness: {', '.join(faith_str_parts)}")
        else:
            print("  Faithfulness: N/A")
            
        q_rel = stats["relevance"]
        if q_rel["count"] > 0:
            print(f"  Relevance: Avg Score = {q_rel['average_score']:.2f}")
        else:
            print("  Relevance: N/A")
            
    print("=" * 80 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Run RAG pipeline end-to-end evaluation.")
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None, 
        help="Limit evaluation to the first N questions."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume evaluation: skip questions already present in results/eval_report.json and merge new results."
    )
    args = parser.parse_args()

    # Paths
    eval_set_path = "data/eval_set.json"
    if not os.path.exists(eval_set_path):
        eval_set_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "eval_set.json"))

    if not os.path.exists(eval_set_path):
        print(f"Error: Could not find eval_set.json at {eval_set_path}")
        sys.exit(1)

    with open(eval_set_path, "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    limit = args.limit
    test_questions = eval_set[:limit] if limit is not None else eval_set

    # Determine report path
    report_path = "results/eval_report.json"
    if not os.path.exists("results") and os.path.exists(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "eval_report.json"))):
        report_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "eval_report.json"))

    results = []
    existing_ids = set()

    if args.resume:
        if os.path.exists(report_path):
            try:
                with open(report_path, "r", encoding="utf-8") as f:
                    results = json.load(f)
                existing_ids = {r.get("question_id") for r in results if isinstance(r, dict) and "question_id" in r}
                print(f"Resuming evaluation: Loaded {len(results)} existing results from {report_path} ({len(existing_ids)} unique question_ids).")
            except Exception as e:
                print(f"Warning: Failed to load existing report from {report_path}: {e}")
                results = []
                existing_ids = set()
        else:
            print(f"Notice: --resume specified, but {report_path} does not exist. Starting fresh evaluation.")

    if args.resume and existing_ids:
        questions_to_process = [q for q in test_questions if q.get("id") not in existing_ids]
        skipped_count = len(test_questions) - len(questions_to_process)
        print(f"Skipping {skipped_count} question(s) already in existing report.")
    else:
        questions_to_process = test_questions

    total_to_process = len(questions_to_process)

    print(f"Initializing Retriever, Generator, and Judge components...")
    try:
        retriever = Retriever()
        generator = Generator()
        judge = Judge()
    except Exception as e:
        print(f"Error initializing components: {e}")
        sys.exit(1)

    print(f"Starting end-to-end evaluation on {total_to_process} questions (Total in report will be {len(results) + total_to_process})...")
    print("=" * 80)

    # Ensure results folder exists before starting loop
    os.makedirs(os.path.dirname(report_path) or "results", exist_ok=True)

    for idx, entry in enumerate(questions_to_process, start=1):
        question_id = entry.get("id")
        question_text = entry.get("question")
        qtype = entry.get("question_type")
        difficulty = entry.get("difficulty")
        
        print(f"[{idx}/{total_to_process}] Processing {question_id} ({qtype})...", end="", flush=True)

        try:
            # 1. Retrieve top-3 chunks
            retrieved_chunks = retriever.retrieve(question_text, top_k=3)
            retrieved_ids = [c["chunk_id"] for c in retrieved_chunks]
            context_text = "\n\n".join([c["text"] for c in retrieved_chunks])

            # Rate limit delay (2 seconds before generator API call)
            time.sleep(2)

            # 2. Generate answer (uses llama-3.1-8b-instant via Groq)
            gen_res = call_with_retry(generator.generate, question_text, retrieved_chunks)
            generated_answer = gen_res["answer"]

            # 3. Score retrieval quality
            ret_metrics = evaluate_retrieval(entry, retrieved_ids, k=3)

            # Rate limit delay (2 seconds before judge API call)
            time.sleep(2)

            # 4. Score faithfulness (uses llama-3.3-70b-versatile via Groq)
            is_unanswerable = (qtype == "unanswerable")
            faith_metrics = call_with_retry(
                judge.judge_faithfulness, 
                question_text, 
                generated_answer, 
                context_text, 
                is_unanswerable
            )

            # Rate limit delay (2 seconds before relevance API call)
            time.sleep(2)

            # 5. Score relevance (uses llama-3.3-70b-versatile via Groq)
            relevance_metrics = call_with_retry(
                judge.judge_answer_relevance, 
                question_text, 
                generated_answer,
                is_unanswerable
            )

            # Combine output results
            result_dict = {
                "question_id": question_id,
                "question": question_text,
                "question_type": qtype,
                "difficulty": difficulty,
                "generated_answer": generated_answer,
                "retrieved_chunk_ids": retrieved_ids,
                "retrieval_metrics": {
                    "precision_at_k": ret_metrics.get("precision_at_k"),
                    "recall_at_k": ret_metrics.get("recall_at_k"),
                    "hit_at_k": ret_metrics.get("hit_at_k"),
                    "mrr": ret_metrics.get("mrr")
                },
                "faithfulness": faith_metrics,
                "relevance": relevance_metrics
            }
            results.append(result_dict)

            # Save progress incrementally to eval_report.json
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            print(" Done.", flush=True)

        except Exception as e:
            print(f" Failed.\nError processing question {question_id}: {e}", flush=True)
            # Log error and continue to the next question rather than crashing
            continue

    print("=" * 80)
    print(f"Evaluation complete. Total questions in report: {len(results)}.")

    # Compute overall statistics from the full merged set
    overall_summary = compute_metrics_for_subset(results)

    # Compute by-question-type statistics from the full merged set
    by_type_summary = {}
    qtypes_in_results = sorted(list(set(r["question_type"] for r in results)))
    for qtype in qtypes_in_results:
        type_subset = [r for r in results if r["question_type"] == qtype]
        by_type_summary[qtype] = compute_metrics_for_subset(type_subset)

    summary_report = {
        "overall": overall_summary,
        "by_question_type": by_type_summary
    }

    summary_path = "results/eval_summary.json"
    if os.path.dirname(report_path):
        summary_path = os.path.join(os.path.dirname(report_path), "eval_summary.json")

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2, ensure_ascii=False)

    print(f"Full results report saved to: {report_path}")
    print(f"Summary metrics saved to: {summary_path}")

    # Print summary reports to the console
    print_summary_to_console(summary_report)

if __name__ == "__main__":
    main()
