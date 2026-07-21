# RAG Evaluator: Faithfulness & Retrieval Quality Evaluation Framework

## Overview
A framework for evaluating retrieval-augmented generation (RAG) systems on faithfulness, retrieval quality, and hallucination resistance — built and tested on a knowledge base of CS229 (Stanford) machine learning and deep learning lecture notes. Rather than building another RAG chatbot, this project builds the evaluation harness that determines whether a RAG system's answers can actually be trusted.

---

## Why This Project?
Evaluating RAG systems is a critical discipline as LLM products transition from prototypes to production scale. The fundamental failure mode of naive RAG isn't just failing to retrieve an answer — it is generating confident, plausible-sounding hallucinations when provided incomplete or irrelevant context. This framework addresses that challenge by measuring retrieval quality and answer faithfulness as separate, decoupled evaluation dimensions, isolating failures to either the retrieval component or the generator LLM.

---

## Architecture

```text
┌─────────────────┐     ┌──────────────────┐     ┌────────────────────────┐
│   Source PDFs   │ ──> │ Chunking Pipeline│ ──> │ FAISS Vector Store     │
│ (CS229 Lecture) │     │ (Section/Overlap)│     │ (all-MiniLM-L6-v2)     │
└─────────────────┘     └──────────────────┘     └───────────┬────────────┘
                                                             │
                                                             ▼
┌─────────────────┐     ┌──────────────────┐     ┌────────────────────────┐
│ Streamlit App   │ <── │ Evaluator Module │ <── │ RAG Pipeline           │
│ (Dashboard UI)  │     │ (Metrics + Judge)│     │ (Retriever + Generator)│
└─────────────────┘     └──────────────────┘     └────────────────────────┘
```

---

## Tech Stack

| Component | Technology | Rationale / Purpose |
| :--- | :--- | :--- |
| **Embeddings** | `sentence-transformers` (`all-MiniLM-L6-v2`) | Generates dense 384-dimensional vector embeddings for text chunks and queries. |
| **Vector Index** | `FAISS` (`faiss-cpu`) | Performs fast L2 similarity search over chunk embeddings. |
| **Generator LLM** | Groq API (`llama-3.1-8b-instant`) | Fast, lightweight model for producing standard RAG answers. |
| **Judge LLM** | Groq API (`llama-3.3-70b-versatile`) | **Why a larger model?** LLM-as-judge requires superior logical reasoning, strict adherence to JSON schemas, and nuanced factual grounding analysis compared to standard generation. |
| **Dashboard & Viz** | `Streamlit` + `Plotly` | Interactive browser dashboard for metric inspection, filtering, and failure analysis. |

---

## Evaluation Design

### Dataset Structure
The dataset consists of **45 hand-verified question-answer pairs** categorized evenly across 5 distinct query types (9 questions each):
- **`factual`**: Direct facts, definitions, and explicit property lookups.
- **`formula`**: Mathematical equations, update rules, and formal derivations.
- **`comparative`**: Contrasting two concepts (e.g., L1 vs. L2 regularization, discriminative vs. generative).
- **`multi_hop`**: Complex questions requiring synthesis across multiple chunks or multi-step reasoning.
- **`unanswerable`**: Out-of-corpus queries where the gold chunk is `None`. Evaluates refusal capability and hallucination resistance.

> **Note on Gold Chunks**: `gold_chunk_id` supports both single-string IDs (`"chunk_008"`) and multi-chunk lists (`["chunk_010", "chunk_011"]`) to account for information spanning adjacent sections.

### Evaluation Metrics

#### 1. Retrieval Quality Metrics (Evaluated over 36 Answerable Questions)
- **`Precision@k`** ($k=3$): Proportion of retrieved chunks that are gold chunks.
- **`Recall@k`** ($k=3$): Proportion of relevant gold chunks retrieved.
- **`Hit@k`** ($k=3$): Binary metric ($1.0$ if at least one gold chunk is present in top-$k$, else $0.0$).
- **`MRR` (Mean Reciprocal Rank)**: Reciprocal of the rank of the first correct gold chunk.

#### 2. Generation & Judge Metrics (Evaluated over all 45 Questions via LLM-as-Judge)
- **Faithfulness Classification**:
  - `fully_grounded`: All claims supported by context (or correct refusal for answerable questions missing context).
  - `correct_refusal`: Correctly declines to answer an unanswerable question.
  - `partially_grounded`: Mix of supported and unsupported claims.
  - `hallucinated`: Answer contains unsupported factual claims.
  - `incorrect_refusal`: Refuses to answer when context contains the exact answer.
  - `incorrect_hallucination`: Generates an answer to an unanswerable question.
- **Answer Relevance Score**: Integer rating from **1** (completely irrelevant or unhelpful refusal on valid question) to **5** (perfectly addresses question or correctly refuses unanswerable query).

---

## Results Summary

Across the full **45-question evaluation run**, the framework yielded the following benchmark statistics:

### Overall Summary Statistics
- **Total Questions Evaluated**: `45`
- **Faithfulness Rate** (`fully_grounded` + `correct_refusal`): **`97.8%`** (44/45 healthy outcomes)
- **Average Relevance Score**: **`3.84` / `5.00`**
- **Retrieval Hit@3**: **`75.0%`** (27/36 answerable questions)
- **Retrieval MRR**: **`0.6991`**
- **Retrieval Precision@3**: **`0.2593`** *(Capped naturally because $k=3$ while most queries have 1 gold chunk)*
- **Retrieval Recall@3**: **`0.7222`**

### Performance Breakdown by Question Type

| Question Type | Count (N) | Precision@3 | Recall@3 | Hit@3 (%) | MRR | Faithfulness Distribution | Avg Relevance |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **Comparative** | 9 | `0.259` | `0.667` | `66.7%` | `0.611` | `100%` fully_grounded | `3.44` |
| **Factual** | 9 | `0.222` | `0.667` | `66.7%` | `0.667` | `100%` fully_grounded | `3.44` |
| **Formula** | 9 | `0.259` | `0.778` | `77.8%` | `0.778` | `100%` fully_grounded | `4.22` |
| **Multi-Hop** | 9 | `0.296` | `0.778` | `77.8%` | `0.741` | `89%` fully_grounded, `11%` partially_grounded | `3.11` |
| **Unanswerable** | 9 | N/A | N/A | N/A | N/A | `100%` correct_refusal | `5.00` |

---

## Key Findings & Failure Patterns

### 1. Math-Notation Dilution in Dense Embeddings
In mathematical and formula-heavy queries (such as derivation of the Normal Equation or Sigmoid gradient properties), dense embedding models suffer from semantic dilution. Mathematical symbols, LaTeX expressions, and inline variables ($\theta$, $\nabla$) disperse semantic density across sub-word tokens, causing dense vector retrievers to occasionally rank general descriptive paragraphs higher than the specific formula chunk.

### 2. Lexical False-Friend Retrieval Failure (`q003`)
In question `q003` (*"In the context of locally weighted linear regression, how is a non-parametric learning algorithm defined, and how does it differ from a parametric learning algorithm?"*), the retriever failed completely (**Hit@3 = False, Recall@3 = 0.0**). The embedding model over-prioritized dominant domain keywords ("linear regression", "learning algorithm") present in general sections (`chunk_032`, `chunk_002`), missing the specific conceptual definition located in `chunk_008`.

### 3. Free-Tier Daily Token Quota Constraints & Resiliency (`--resume`)
During full-scale evaluation, the 70B judge model (`llama-3.3-70b-versatile`) hit Groq's daily free-tier quota (100,000 TPD) after 17 questions (~34 judge calls). The pipeline's exception handler caught 429 errors and logged partial results without data corruption. To solve this constraint:
- Added `--resume` flag to `run_eval.py` to skip already-evaluated `question_id`s on re-run.
- Implemented **context truncation** (~800 words per chunk max) and constrained reasoning output length (**"2-3 sentences maximum"**) in `judge_metrics.py`, reducing token usage per call by over 45% and allowing full 45-question completion within quota.

### 4. Generation-Side Repetition Degeneration Loop (`q031`)
Question `q031` (sigmoid derivative derivation) achieved **perfect retrieval** (**Hit@3 = True, MRR = 1.0**). However, the generator LLM (`llama-3.1-8b-instant`) entered an infinite algebraic repetition loop, repeatedly rewriting equivalent expressions until reaching token truncation.
- **Why this proves multi-metric evaluation is necessary**: The faithfulness judge classified the answer as `partially_grounded` (since one valid math fact was present in the output). However, the relevance judge correctly assigned a score of **`1 / 5`** for complete failure to answer the prompt. Measuring faithfulness alone would have obscured this severe generator failure.

---

## Limitations

- **Small-Scale Knowledge Base**: The vector index contains 56 chunks from CS229 lecture notes. While ideal for controlled benchmarks, findings may not fully generalize to enterprise-scale corpora (10,000+ chunks).
- **LLM-as-Judge Biases & Prompt Sensitivity**: Judge models can exhibit biases or misclassify edge cases. During development, the judge initially penalized valid refusals using outside domain knowledge; this required strict prompt constraints forbidding external background knowledge and mandating context quotes for `incorrect_refusal` labels.
- **Precision@k Metric Floor**: Precision@k ($k=3$) is capped at $\approx 0.333$ for queries with only 1 gold chunk, because 2 out of the 3 retrieved slots will inherently count as non-gold even if relevant.
- **API Rate & Quota Limits**: Evaluating large suites requires rate-limit delays (2 seconds between calls) and quota management strategies.

---

## Streamlit Dashboard

An interactive visualization dashboard is located in `dashboard/app.py`.

### Features
- **Summary Cards**: Quick inspection of global KPIs.
- **Question Type Charts**: Plotly bar charts for retrieval, relevance, and faithfulness.
- **Faithfulness Donut Chart**: Visual distribution of groundedness vs. refusals.
- **Question Explorer**: Interactive filterable dataframe and detailed inspect expander.
- **Failure Spotlight**: Styled case study containers highlighting failure patterns.

### Running the Dashboard
```powershell
python -m streamlit run dashboard/app.py
```
*Access in browser at: `http://localhost:8501`*

![Dashboard Overview Placeholder](docs/dashboard_preview.png)

---

## Project Structure

```text
rag-evaluator/
├── data/
│   ├── chunks.json                   # 56 chunked knowledge base entries
│   ├── chunk_notes.py                # Chunking script used to generate chunks.json
│   ├── eval_set.json                 # 45 hand-crafted benchmark questions & gold chunk IDs
│   ├── supervised_learning_ch1-4.pdf # Source PDF (CS229 ch1-4)
│   └── deep_learning_ch7.pdf         # Source PDF (CS229 ch7)
├── rag_pipeline/
│   ├── embed_chunks.py               # Generates FAISS vector index from chunks.json
│   ├── retriever.py                  # Top-k vector retriever using FAISS
│   ├── generator.py                  # Groq API generator interface
│   └── vector_store/                 # Persisted FAISS index (chunk_index.faiss) & metadata
├── evaluator/
│   ├── run_eval.py                   # End-to-end evaluation runner with --resume support
│   └── metrics/
│       ├── retrieval_metrics.py      # Precision@k, Recall@k, Hit@k, MRR
│       └── judge_metrics.py          # Faithfulness & Relevance LLM-as-Judge
├── results/
│   ├── eval_report.json              # Full raw 45-question evaluation output
│   ├── eval_summary.json             # Aggregated summary statistics
│   └── observations.md               # Documented failure patterns & case studies
├── dashboard/
│   └── app.py                        # Streamlit & Plotly dashboard application
├── scratch/                          # Development & verification scripts (not part of core pipeline)
│   ├── generate_eval_set.py
│   ├── print_chunks.py
│   └── verify_eval_set.py
├── requirements.txt                  # Dependencies (sentence-transformers, faiss, groq, streamlit, plotly)
└── README.md                         # Project documentation
```

---

## Setup & Usage

### 1. Clone & Setup Environment
```powershell
git clone https://github.com/your-username/rag-evaluator.git
cd rag-evaluator
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Build Vector Store (Indexing)
```powershell
python rag_pipeline/embed_chunks.py
```

### 4. Run Evaluation
```powershell
# Run full evaluation (or resume previous interrupted run)
python evaluator/run_eval.py --resume

# Run a quick smoke test on the first 5 questions
python evaluator/run_eval.py --limit 5
```

### 5. Launch Dashboard
```powershell
python -m streamlit run dashboard/app.py
```

---

## Future Work

- **Multi-Domain Benchmark Expansion**: Scale knowledge base to include diverse domains (financial, legal, technical documentation).
- **Hybrid Retrieval Integration**: Compare dense vector search against hybrid retrieval combining BM25 lexical search with FAISS dense embeddings.
- **Cross-Judge Validation**: Evaluate consistency across multiple judge model providers (e.g. Anthropic Claude 3.5, OpenAI GPT-4o, and local Llama 3 models).
