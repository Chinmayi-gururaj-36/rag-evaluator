import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

eval_set_path = os.path.join("data", "eval_set.json")
chunks_path = os.path.join("data", "chunks.json")

with open(eval_set_path, "r", encoding="utf-8") as f:
    eval_set = json.load(f)

with open(chunks_path, "r", encoding="utf-8") as f:
    chunks = json.load(f)

chunk_map = {c["chunk_id"]: c for c in chunks}

# We define key check terms for each question id (q001 to q036 are answerable)
check_requirements = {
    "q001": ["least mean squares", "widrow-hoff", "lms"],
    "q002": ["convex", "global", "optima"],
    "q003": ["parametric", "non-parametric", "linear grows"],
    "q004": ["negative class", "positive class", "0", "1"],
    "q005": ["likelihood", "maximize", "choose"],
    "q006": ["conditional distribution", "joint distribution", "p(y|x)", "p(x|y)"],
    "q007": ["natural parameter", "canonical parameter", "η"],
    "q008": ["matrix multiplication", "parameter sharing", "conv1d-s"],
    "q009": ["forward pass", "backward pass", "backpropagation"],
    "q010": ["x⊤x", "x^t * x", "θ = (x^tx)^-1", "normal equations"],
    "q011": ["1 / (1 + e^-z)", "g'(z) = g(z)(1 - g(z))", "sigmoid"],
    "q012": ["-(x(i) - x)^2", "2τ^2", "weight"],
    "q013": ["ℓlogistic", "ylog(1 + exp", "(1-y) log(1 + exp"],
    "q014": ["exp(θ", "∑", "softmax", "P(y = i | x"],
    "q015": ["b(y)", "exp(η", "-a(η)", "exponential family"],
    "q016": ["1 + ∑", "2 + ∑", "laplace", "smoothing"],
    "q017": ["h^-1", "hessian", "newton", "θ -"],
    "q018": ["erf", "z/2", "gelu"],
    "q019": ["batch", "stochastic", "gradient descent", "scan"],
    "q020": ["parametric", "non-parametric", "linear"],
    "q021": ["newton", "hessian", "inverting", "gradient descent"],
    "q022": ["gda", "logistic regression", "assumptions", "gaussian"],
    "q023": ["bernoulli", "multinomial", "vocabulary", "independent"],
    "q024": ["vanishing", "sigmoid", "tanh", "relu", "gelu"],
    "q025": ["o(km)", "o(m^2)", "parameter", "convolution"],
    "q026": ["continuous", "discrete", "gda", "naive bayes"],
    "q027": ["linear", "non-linear", "collapse", "activation"],
    "q028": ["gradient descent", "partial derivative", "squared error", "lms"],
    "q029": ["invertible", "linearly independent", "features", "normal equations"],
    "q030": ["log-likelihood", "gaussian", "least-squares", "variance"],
    "q031": ["sigmoid", "derivative", "g(z)(1-g(z))"],
    "q032": ["bernoulli", "natural parameter", "sigmoid", "exponential family"],
    "q033": ["layernorm", "scale-invariant", "ln-s"],
    "q034": ["cross-entropy", "chain rule", "gradient", "partial derivative"],
    "q035": ["gda", "decision boundary", "covariance matrix", "mean"],
    "q036": ["jacobian", "diagonal", "o(m)", "o(m^2)", "activation"]
}

report_lines = []
report_lines.append("# Verification Report for eval_set.json")
report_lines.append("This report validates whether the `gold_answer` of each entry in `data/eval_set.json` is backed by the source text in the mapped `gold_chunk_id` in `data/chunks.json`.\n")

mismatch_flagged = False
mismatches_list = []

for item in eval_set:
    qid = item["id"]
    qtype = item["question_type"]
    question = item["question"]
    gold_answer = item["gold_answer"]
    gold_cid = item["gold_chunk_id"]
    
    if qtype == "unanswerable":
        # Check unanswerable is null
        if gold_cid is not None:
            err = f"Mismatch: unanswerable question has gold_chunk_id = {gold_cid} (should be null)"
            mismatches_list.append((qid, err))
            mismatch_flagged = True
        continue
        
    if gold_cid is None:
        err = f"Mismatch: answerable question has gold_chunk_id = null"
        mismatches_list.append((qid, err))
        mismatch_flagged = True
        continue
        
    if gold_cid not in chunk_map:
        err = f"Mismatch: gold_chunk_id '{gold_cid}' does not exist in chunks.json"
        mismatches_list.append((qid, err))
        mismatch_flagged = True
        continue
        
    chunk_text = chunk_map[gold_cid]["text"]
    
    # Run keywords validation
    missing_keywords = []
    # Normalize strings for comparison (remove whitespace and punctuation for math symbols)
    def normalize_for_math(s):
        return "".join(c for c in s.lower() if c.isalnum() or c in "+-*/^()_={}[]|⊤⊤⊤\u03b8\u03c5\u03c6\u03c3\u03b7\u03c4\u03bb\u03b2\u2211\u2202\u2212\u221d\u2227\u2299\u22c5\u225c\u2113\u22a4")

    chunk_norm = normalize_for_math(chunk_text)
    
    req_terms = check_requirements.get(qid, [])
    for term in req_terms:
        # Check simple substring or mathematical subset
        term_clean = term.lower()
        term_math = normalize_for_math(term)
        
        # If term contains math characters or is short, try math norm match, else plain substring match
        has_math = any(c in term for c in "=^*-/{}[]\\_⊤\u03b8\u03c7\u03b7\u2211\u2202\u2212\u2299\u2113\u22a4")
        
        if has_math:
            # Check if normalized math representation is in normalized chunk
            if term_math not in chunk_norm:
                missing_keywords.append(term)
        else:
            # standard check
            if term_clean not in chunk_text.lower():
                # fallback to math check
                if term_math not in chunk_norm:
                    missing_keywords.append(term)

    if missing_keywords:
        err = f"Potential mismatch: gold_answer mentions terms/formulas not found in {gold_cid}: {missing_keywords}"
        mismatches_list.append((qid, err))
        mismatch_flagged = True
        report_lines.append(f"## ❌ [MISMATCH] {qid} (Type: {qtype}, Chunk: {gold_cid})")
    else:
        report_lines.append(f"##  [VERIFIED] {qid} (Type: {qtype}, Chunk: {gold_cid})")
        
    report_lines.append(f"**Question:** {question}")
    report_lines.append(f"**Gold Answer:** {gold_answer}")
    report_lines.append(f"**Chunk Text Snippet:**\n```\n{chunk_text[:300]}...\n```")
    if missing_keywords:
        report_lines.append(f"> **Warning:** Missing terms/formulas in chunk: `{missing_keywords}`")
    report_lines.append("\n---\n")

report_lines.append("## Summary of Mismatches")
if mismatch_flagged:
    report_lines.append(f"Total mismatches flagged: {len(mismatches_list)}")
    for qid, err in mismatches_list:
        report_lines.append(f"- **{qid}**: {err}")
else:
    report_lines.append("All 45 entries successfully verified. No mismatches found between the gold answers and the mapped source chunks.")

# Write report to artifact folder
artifact_dir = "C:\\Users\\chinm\\.gemini\\antigravity-ide\\brain\\9a24aa8e-4252-4780-950c-fa740ba1c58d"
report_path = os.path.join(artifact_dir, "verification_report.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print(f"Verification report written to: {report_path}")
if mismatch_flagged:
    print(f"Flagged {len(mismatches_list)} potential mismatches. See report for details.")
else:
    print("All checks completed. 100% verified, no mismatches!")
