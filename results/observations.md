# Retrieval Observations and Failure Patterns

This document tracks observations and failure patterns identified during retrieval evaluation runs.

## Retrieval Failure Patterns (Smoke Test Findings)

During the smoke test evaluating the retrieval system on a subset of the evaluation set (top-3 retrieved chunks), two main failure patterns were observed:

### Pattern 1: Conceptual/Semantic Misalignment on Specific Terminology
* **Question ID**: `q003`
* **Question**: *"In the context of locally weighted linear regression, how is a non-parametric learning algorithm defined, and how does it differ from a parametric learning algorithm?"*
* **Gold Chunk**: [`chunk_008`](file:///c:/Users/chinm/Documents/Projects/rag-evaluator/rag_pipeline/vector_store/chunk_metadata.json#L66-L73) (contains the explicit definition of parametric vs. non-parametric algorithms).
* **Retrieved Chunks instead**: `['chunk_032', 'chunk_002', 'chunk_021']`
  * **Rank 1**: [`chunk_032`](file:///c:/Users/chinm/Documents/Projects/rag-evaluator/rag_pipeline/vector_store/chunk_metadata.json#L282-L289) (Section 7.1: Supervised learning with non-linear models - discusses linear/non-linear parameters).
  * **Rank 2**: [`chunk_002`](file:///c:/Users/chinm/Documents/Projects/rag-evaluator/rag_pipeline/vector_store/chunk_metadata.json#L12-L18) (Section 1.1: LMS algorithm - discusses gradient descent convergence).
  * **Rank 3**: [`chunk_021`](file:///c:/Users/chinm/Documents/Projects/rag-evaluator/rag_pipeline/vector_store/chunk_metadata.json#L183-L189) (Chapter 4: Generative learning algorithms - introduces discriminative vs. generative).
* **Failure Analysis**: The retriever failed to retrieve `chunk_008` (which has a 0.0 recall@3, precision@3, and hit@3). The embedding model likely over-prioritized the general keywords like "linear regression" and "learning algorithm" in `chunk_032` (which contains "linear in the parameters $\theta$") and `chunk_002`, missing the specific semantic definition of "non-parametric" and "parametric" located in `chunk_008`.

---

### Pattern 2: forced Retrieval of Irrelevant Chunks for Unanswerable Questions
* **Question ID**: `q037`
* **Question**: *"How does the Adam optimizer's update rule adjust the learning rate using both first and second moments of the gradients, and how is bias correction applied in the initial steps?"*
* **Gold Chunk**: `None` (The question is designated as `unanswerable` because the Adam optimizer is not covered anywhere in the source material).
* **Retrieved Chunks instead**: `['chunk_002', 'chunk_001', 'chunk_034']`
  * **Rank 1**: [`chunk_002`](file:///c:/Users/chinm/Documents/Projects/rag-evaluator/rag_pipeline/vector_store/chunk_metadata.json#L12-L18) (Section 1.1: LMS algorithm - batch gradient descent).
  * **Rank 2**: [`chunk_001`](file:///c:/Users/chinm/Documents/Projects/rag-evaluator/rag_pipeline/vector_store/chunk_metadata.json#L3-L10) (Section 1.1: LMS algorithm - gradient descent / LMS update rule).
  * **Rank 3**: [`chunk_034`](file:///c:/Users/chinm/Documents/Projects/rag-evaluator/rag_pipeline/vector_store/chunk_metadata.json#L300-L306) (Section 7.1: Supervised learning with non-linear models - SGD and mini-batch SGD).
* **Failure Analysis**: Since the vector database does not contain information about the Adam optimizer, this is an out-of-distribution / unanswerable question. However, because the vector retriever performs a nearest-neighbor L2 search, it is forced to retrieve chunks. It matches keywords such as "learning rate", "update rule", and "gradients" to chunks describing gradient descent and SGD update rules (`chunk_001` and `chunk_034`), leading to false positive retrievals.

---

## Pattern 3: Free-Tier Daily Token Quota Constraint on LLM-as-Judge Evaluation

During the full 45-question evaluation run, the LLM-as-judge component (llama-3.3-70b-versatile) hit Groq's free-tier daily token quota (100,000 TPD) after processing 17 questions (~34 judge calls), causing all subsequent questions to fail with 429 errors. The pipeline's error-handling design correctly caught each failure, logged it, and continued rather than crashing — producing a clean partial result (17/45) with no data corruption.

This is a realistic constraint for any RAG evaluation system built on free-tier LLM APIs: LLM-as-judge evaluation is token-intensive (each judgment call includes the question, generated answer, and full retrieved context), and daily quotas — not just per-minute rate limits — become a real bottleneck at scale.

Fix: added `--resume` capability to `run_eval.py` (skips already-evaluated questions on re-run) and reduced per-call token usage (context truncation, shorter reasoning field) to fit the remaining evaluation within quota.

---

## Pattern 4: Generation-Side Repetition Loop on Complex Multi-Step Derivations

Question `q031` (`multi_hop`, sigmoid derivative derivation) retrieved the correct context perfectly (hit@3=True, MRR=1.0), ruling out retrieval as the cause of failure. Instead, the generator (`llama-3.1-8b-instant`) entered a degenerate repetition loop while attempting the multi-step algebraic derivation, repeatedly rewriting equivalent forms of the same expression without converging on an answer, until truncated by the token limit.

This is a known LLM failure mode (repetition/degeneration) rather than a RAG-specific issue, but it's relevant to RAG evaluation because it demonstrates why measuring faithfulness alone is insufficient: the faithfulness judge labeled this "partially_grounded" (technically reasonable, since one correct fact was embedded in the repetitive text), while the relevance judge correctly identified the answer as a near-total failure (score: 1/5) for never actually answering the question. This validates the evaluator's multi-metric design — faithfulness and relevance capture genuinely different failure modes, and a complete evaluation requires both.


