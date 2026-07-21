import os
import json
from typing import List, Dict, Any, Union, Optional

def precision_at_k(retrieved_chunk_ids: List[str], gold_chunk_ids: List[str], k: int) -> float:
    """
    Computes Precision at K.

    Precision at K is the fraction of the top-K retrieved chunks that are relevant
    (i.e., present in the gold_chunk_ids).

    Args:
        retrieved_chunk_ids: Ordered list of retrieved chunk IDs.
        gold_chunk_ids: List of ground-truth relevant chunk IDs.
        k: The number of top retrieved results to evaluate.

    Returns:
        The precision at k as a float between 0.0 and 1.0.
    """
    if k <= 0 or not retrieved_chunk_ids or not gold_chunk_ids:
        return 0.0
    retrieved_at_k = retrieved_chunk_ids[:k]
    gold_set = set(gold_chunk_ids)
    hits = sum(1 for chunk in retrieved_at_k if chunk in gold_set)
    return hits / k

def recall_at_k(retrieved_chunk_ids: List[str], gold_chunk_ids: List[str], k: int) -> float:
    """
    Computes Recall at K.

    Recall at K is the fraction of all gold_chunk_ids that appear in the top-K
    retrieved chunks.

    Args:
        retrieved_chunk_ids: Ordered list of retrieved chunk IDs.
        gold_chunk_ids: List of ground-truth relevant chunk IDs.
        k: The number of top retrieved results to evaluate.

    Returns:
        The recall at k as a float between 0.0 and 1.0.
    """
    if k <= 0 or not retrieved_chunk_ids or not gold_chunk_ids:
        return 0.0
    retrieved_at_k = set(retrieved_chunk_ids[:k])
    gold_set = set(gold_chunk_ids)
    hits = len(retrieved_at_k.intersection(gold_set))
    return hits / len(gold_set)

def hit_at_k(retrieved_chunk_ids: List[str], gold_chunk_ids: List[str], k: int) -> bool:
    """
    Computes Hit at K.

    Hit at K is a boolean indicating whether at least one of the gold_chunk_ids
    is present within the top-K retrieved chunks.

    Args:
        retrieved_chunk_ids: Ordered list of retrieved chunk IDs.
        gold_chunk_ids: List of ground-truth relevant chunk IDs.
        k: The number of top retrieved results to evaluate.

    Returns:
        True if there is a hit (at least one matching chunk in top-K), False otherwise.
    """
    if k <= 0 or not retrieved_chunk_ids or not gold_chunk_ids:
        return False
    retrieved_at_k = set(retrieved_chunk_ids[:k])
    gold_set = set(gold_chunk_ids)
    return not retrieved_at_k.isdisjoint(gold_set)

def mean_reciprocal_rank(retrieved_chunk_ids: List[str], gold_chunk_ids: List[str]) -> float:
    """
    Computes the Reciprocal Rank (RR) for a single query.

    Reciprocal Rank is 1 divided by the 1-indexed rank of the first correctly
    retrieved gold chunk. If no gold chunk is found in the retrieved list, it returns 0.0.

    Args:
        retrieved_chunk_ids: Ordered list of retrieved chunk IDs.
        gold_chunk_ids: List of ground-truth relevant chunk IDs.

    Returns:
        The reciprocal rank as a float (e.g., 1.0, 0.5, 0.333, etc.), or 0.0 if no hit.
    """
    if not retrieved_chunk_ids or not gold_chunk_ids:
        return 0.0
    gold_set = set(gold_chunk_ids)
    for rank, chunk_id in enumerate(retrieved_chunk_ids, start=1):
        if chunk_id in gold_set:
            return 1.0 / rank
    return 0.0

def evaluate_retrieval(eval_entry: Dict[str, Any], retrieved_chunk_ids: List[str], k: int = 3) -> Dict[str, Any]:
    """
    Evaluates the retrieval performance for a single evaluation entry.

    This wrapper normalizes the gold chunk IDs, handles unanswerable questions
    separately by returning None for metrics with a note, and computes the
    four retrieval metrics (precision@k, recall@k, hit@k, and mrr) for answerable questions.

    Args:
        eval_entry: A dictionary containing the evaluation question entry (e.g., from eval_set.json).
        retrieved_chunk_ids: Ordered list of retrieved chunk IDs.
        k: The value of k for @k metrics. Defaults to 3.

    Returns:
        A dictionary containing the evaluation results, including:
            - question_id: The ID of the question.
            - precision_at_k: Precision@k or None.
            - recall_at_k: Recall@k or None.
            - hit_at_k: Hit@k or None.
            - mrr: Reciprocal Rank or None.
            - gold_chunk_ids: List of ground-truth relevant chunk IDs (or None).
            - retrieved_chunk_ids: The list of retrieved chunk IDs.
            - note: Optional note (for unanswerable questions).
    """
    question_id = eval_entry.get("id")

    if eval_entry.get("question_type") == "unanswerable":
        return {
            "question_id": question_id,
            "precision_at_k": None,
            "recall_at_k": None,
            "hit_at_k": None,
            "mrr": None,
            "gold_chunk_ids": None,
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "note": "not applicable — unanswerable question"
        }

    raw_gold = eval_entry.get("gold_chunk_id")
    if raw_gold is None:
        gold_chunk_ids: List[str] = []
    elif isinstance(raw_gold, list):
        gold_chunk_ids = raw_gold
    else:
        gold_chunk_ids = [raw_gold]

    p_k = precision_at_k(retrieved_chunk_ids, gold_chunk_ids, k)
    r_k = recall_at_k(retrieved_chunk_ids, gold_chunk_ids, k)
    h_k = hit_at_k(retrieved_chunk_ids, gold_chunk_ids, k)
    mrr_val = mean_reciprocal_rank(retrieved_chunk_ids, gold_chunk_ids)

    return {
        "question_id": question_id,
        "precision_at_k": p_k,
        "recall_at_k": r_k,
        "hit_at_k": h_k,
        "mrr": mrr_val,
        "gold_chunk_ids": gold_chunk_ids,
        "retrieved_chunk_ids": retrieved_chunk_ids
    }

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    
    # Add root directory to path to allow importing rag_pipeline
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    
    try:
        from rag_pipeline.retriever import Retriever
    except ImportError as e:
        print(f"Error importing Retriever. Ensure python path contains the project root: {e}")
        sys.exit(1)
        
    eval_set_path = "data/eval_set.json"
    if not os.path.exists(eval_set_path):
        # Try finding it relative to project root
        eval_set_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "eval_set.json"))

    if not os.path.exists(eval_set_path):
        print(f"Could not find eval_set.json at {eval_set_path}")
        sys.exit(1)

    with open(eval_set_path, "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    # Initialize retriever
    # We construct the default Retriever which loads FAISS and metadata using default paths.
    try:
        retriever = Retriever()
    except Exception as e:
        # Try instantiating with paths relative to project root
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        index_path = os.path.join(root_dir, "rag_pipeline", "vector_store", "chunk_index.faiss")
        metadata_path = os.path.join(root_dir, "rag_pipeline", "vector_store", "chunk_metadata.json")
        try:
            retriever = Retriever(index_path=index_path, metadata_path=metadata_path)
        except Exception as inner_e:
            print(f"Failed to initialize Retriever: {inner_e}")
            sys.exit(1)

    # Filter: first 5 non-unanswerable questions + 1 unanswerable question
    non_unanswerable = [q for q in eval_set if q.get("question_type") != "unanswerable"]
    unanswerable = [q for q in eval_set if q.get("question_type") == "unanswerable"]

    test_questions = non_unanswerable[:5] + unanswerable[:1]

    # Run evaluation
    answerable_results = []
    k_val = 3

    print("=" * 80)
    print(f"Starting Retrieval Evaluation Smoke Test ({len(test_questions)} questions)...")
    print("=" * 80)

    for i, entry in enumerate(test_questions, start=1):
        query = entry["question"]
        try:
            results = retriever.retrieve(query, top_k=k_val)
        except Exception as e:
            print(f"Error retrieving for query '{query}': {e}")
            continue
            
        retrieved_ids = [res["chunk_id"] for res in results]
        eval_res = evaluate_retrieval(entry, retrieved_ids, k=k_val)

        print(f"Test Question {i}/{len(test_questions)} (ID: {entry.get('id')}, Type: {entry.get('question_type')}):")
        print(f"  Question  : {query}")
        print(f"  Gold Chunk(s): {eval_res['gold_chunk_ids']}")
        print(f"  Retrieved : {eval_res['retrieved_chunk_ids']}")
        
        if entry.get("question_type") == "unanswerable":
            print(f"  Metrics   : {eval_res.get('note')}")
        else:
            print(f"  Metrics   : Precision@{k_val} = {eval_res['precision_at_k']:.4f}, "
                  f"Recall@{k_val} = {eval_res['recall_at_k']:.4f}, "
                  f"Hit@{k_val} = {eval_res['hit_at_k']}, "
                  f"MRR = {eval_res['mrr']:.4f}")
            answerable_results.append(eval_res)
        print("-" * 80)

    if answerable_results:
        avg_precision = sum(r["precision_at_k"] for r in answerable_results) / len(answerable_results)
        avg_recall = sum(r["recall_at_k"] for r in answerable_results) / len(answerable_results)
        avg_hit = sum(1.0 if r["hit_at_k"] else 0.0 for r in answerable_results) / len(answerable_results)
        avg_mrr = sum(r["mrr"] for r in answerable_results) / len(answerable_results)

        print("AVERAGE METRICS ACROSS ANSWERABLE QUESTIONS (N = 5):")
        print(f"  Average Precision@{k_val}: {avg_precision:.4f}")
        print(f"  Average Recall@{k_val}   : {avg_recall:.4f}")
        print(f"  Average Hit@{k_val}      : {avg_hit:.4f}")
        print(f"  Average MRR        : {avg_mrr:.4f}")
        print("=" * 80)
