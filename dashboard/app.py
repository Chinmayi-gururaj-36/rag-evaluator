import os
import json
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="RAG Evaluator Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling cards, badges, and headers
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
    }
    .spotlight-card {
        background-color: #1E1B4B;
        border-left: 5px solid #6366F1;
        border-radius: 8px;
        padding: 18px;
        margin-bottom: 15px;
    }
    .spotlight-warning {
        background-color: #451A03;
        border-left: 5px solid #F59E0B;
        border-radius: 8px;
        padding: 18px;
        margin-bottom: 15px;
    }
    .spotlight-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #F3F4F6;
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Data Loading (Cached)
# -----------------------------------------------------------------------------
@st.cache_data
def load_eval_data():
    """
    Loads eval_report.json and eval_summary.json with path fallback.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(base_dir, ".."))
    
    report_paths = [
        os.path.join(project_dir, "results", "eval_report.json"),
        "results/eval_report.json",
        "eval_report.json"
    ]
    summary_paths = [
        os.path.join(project_dir, "results", "eval_summary.json"),
        "results/eval_summary.json",
        "eval_summary.json"
    ]
    
    report_data = None
    summary_data = None
    
    for p in report_paths:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                report_data = json.load(f)
            break
            
    for p in summary_paths:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                summary_data = json.load(f)
            break
            
    if not report_data or not summary_data:
        st.error("Error: Could not locate evaluation result files in results/.")
        st.stop()
        
    return report_data, summary_data

report_data, summary_data = load_eval_data()
df_report = pd.DataFrame(report_data)

# Flatten metrics for DataFrame table usage
df_table = pd.DataFrame([
    {
        "question_id": r["question_id"],
        "question_type": r["question_type"],
        "difficulty": r.get("difficulty", "N/A"),
        "faithfulness_label": r.get("faithfulness", {}).get("label", "N/A"),
        "relevance_score": r.get("relevance", {}).get("score", "N/A"),
        "hit_at_k": r.get("retrieval_metrics", {}).get("hit_at_k", False),
        "precision_at_k": r.get("retrieval_metrics", {}).get("precision_at_k", 0.0),
        "recall_at_k": r.get("retrieval_metrics", {}).get("recall_at_k", 0.0),
        "mrr": r.get("retrieval_metrics", {}).get("mrr", 0.0)
    }
    for r in report_data
])

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">📊 RAG Evaluator Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">End-to-End Evaluation Analysis: Retrieval, Faithfulness, and Relevance Metrics</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Section 1: Overall Summary Cards
# -----------------------------------------------------------------------------
st.subheader("🎯 Overall Evaluation Summary")

overall_summary = summary_data.get("overall", {})
total_questions = overall_summary.get("total_questions", len(df_report))

# Faithfulness rate calculation (fully_grounded + correct_refusal)
faith_counts = overall_summary.get("faithfulness", {}).get("counts", {})
fully_grounded_cnt = faith_counts.get("fully_grounded", 0)
correct_refusal_cnt = faith_counts.get("correct_refusal", 0)
healthy_count = fully_grounded_cnt + correct_refusal_cnt
faithfulness_rate = (healthy_count / total_questions * 100) if total_questions > 0 else 0.0

# Relevance & Retrieval
avg_relevance = overall_summary.get("relevance", {}).get("average_score", 0.0) or 0.0
ret_metrics = overall_summary.get("retrieval", {})
avg_hit = (ret_metrics.get("average_hit_at_3", 0.0) or 0.0) * 100
avg_mrr = ret_metrics.get("average_mrr", 0.0) or 0.0

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Questions", f"{total_questions}")
col2.metric("Faithfulness Rate", f"{faithfulness_rate:.1f}%", help="Percentage of fully_grounded + correct_refusal outcomes")
col3.metric("Avg Relevance Score", f"{avg_relevance:.2f} / 5.0")
col4.metric("Avg Hit@3", f"{avg_hit:.1f}%", help="Evaluated across 36 answerable questions")
col5.metric("Avg MRR", f"{avg_mrr:.3f}")

st.markdown("---")

# -----------------------------------------------------------------------------
# Section 2 & 3: Charts (Breakdown by Question Type & Overall Distribution)
# -----------------------------------------------------------------------------
st.subheader("📈 Performance Breakdown & Label Distribution")

tab1, tab2, tab3 = st.tabs(["Question Type Breakdown", "Retrieval & Relevance", "Faithfulness Distribution"])

by_type = summary_data.get("by_question_type", {})
qtypes = list(by_type.keys())

with tab1:
    col_left, col_right = st.columns(2)
    
    with col_left:
        # Stacked bar chart showing faithfulness label distribution per question_type
        type_faith_data = []
        for qt, stats in by_type.items():
            counts = stats.get("faithfulness", {}).get("counts", {})
            for lbl, cnt in counts.items():
                if cnt > 0:
                    type_faith_data.append({"Question Type": qt, "Label": lbl, "Count": cnt})
                    
        df_type_faith = pd.DataFrame(type_faith_data)
        
        color_map = {
            "fully_grounded": "#10B981",
            "correct_refusal": "#3B82F6",
            "partially_grounded": "#F59E0B",
            "hallucinated": "#EF4444",
            "incorrect_refusal": "#8B5CF6",
            "incorrect_hallucination": "#EC4899"
        }
        
        fig_type_faith = px.bar(
            df_type_faith,
            x="Question Type",
            y="Count",
            color="Label",
            title="Faithfulness Label Distribution by Question Type",
            color_discrete_map=color_map,
            template="plotly_dark"
        )
        fig_type_faith.update_layout(barmode="stack", height=380, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_type_faith, use_container_width=True)
        
    with col_right:
        # Average Relevance Score per question_type
        rel_data = []
        for qt, stats in by_type.items():
            rel_score = stats.get("relevance", {}).get("average_score", 0.0) or 0.0
            rel_data.append({"Question Type": qt, "Avg Relevance Score": round(rel_score, 2)})
            
        df_rel = pd.DataFrame(rel_data)
        fig_rel = px.bar(
            df_rel,
            x="Question Type",
            y="Avg Relevance Score",
            color="Question Type",
            title="Average Relevance Score by Question Type (1-5 Scale)",
            text="Avg Relevance Score",
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_rel.update_yaxes(range=[0, 5.5])
        fig_rel.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
        st.plotly_chart(fig_rel, use_container_width=True)

with tab2:
    # Retrieval Metrics per question type (Hit@3 & MRR)
    ret_type_data = []
    for qt, stats in by_type.items():
        ret = stats.get("retrieval", {})
        if ret.get("count", 0) > 0:
            hit3 = (ret.get("average_hit_at_3", 0.0) or 0.0) * 100
            mrr = ret.get("average_mrr", 0.0) or 0.0
            ret_type_data.append({"Question Type": qt, "Metric": "Hit@3 (%)", "Value": round(hit3, 1)})
            ret_type_data.append({"Question Type": qt, "Metric": "MRR", "Value": round(mrr, 3)})
            
    df_ret_type = pd.DataFrame(ret_type_data)
    
    fig_ret = px.bar(
        df_ret_type,
        x="Question Type",
        y="Value",
        color="Metric",
        barmode="group",
        title="Retrieval Metrics (Hit@3 and MRR) Across Answerable Question Types",
        template="plotly_dark",
        text="Value"
    )
    fig_ret.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_ret, use_container_width=True)

with tab3:
    # Donut chart showing overall percentage breakdown of all faithfulness labels across all 45 questions
    overall_faith = overall_summary.get("faithfulness", {}).get("counts", {})
    df_overall_faith = pd.DataFrame([
        {"Label": lbl, "Count": cnt}
        for lbl, cnt in overall_faith.items() if cnt > 0
    ])
    
    fig_donut = px.pie(
        df_overall_faith,
        values="Count",
        names="Label",
        title="Overall Faithfulness Label Breakdown (N = 45)",
        hole=0.4,
        color="Label",
        color_discrete_map=color_map,
        template="plotly_dark"
    )
    fig_donut.update_traces(textinfo="percent+label+value")
    fig_donut.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_donut, use_container_width=True)

st.markdown("---")

# -----------------------------------------------------------------------------
# Section 4: Question Explorer (Interactive Table & Detail Inspector)
# -----------------------------------------------------------------------------
st.subheader("🔍 Question Explorer")

col_filter1, col_filter2 = st.columns([1, 2])

with col_filter1:
    available_labels = ["All"] + sorted(list(df_table["faithfulness_label"].unique()))
    selected_label = st.selectbox("Filter by Faithfulness Label:", available_labels)
    
with col_filter2:
    st.caption("Select a question below to inspect generated answer, context chunks, and LLM-as-judge reasoning.")

if selected_label != "All":
    filtered_df = df_table[df_table["faithfulness_label"] == selected_label]
else:
    filtered_df = df_table

st.dataframe(
    filtered_df[[
        "question_id", "question_type", "difficulty", 
        "faithfulness_label", "relevance_score", "hit_at_k"
    ]],
    use_container_width=True,
    hide_index=True
)

st.markdown("##### 📖 Deep-Dive Question Inspector")
selected_q_id = st.selectbox(
    "Select Question ID to Inspect Details:",
    options=filtered_df["question_id"].tolist()
)

if selected_q_id:
    q_entry = next(r for r in report_data if r["question_id"] == selected_q_id)
    
    with st.expander(f"Inspect Details for Question {selected_q_id} ({q_entry.get('question_type')})", expanded=True):
        st.markdown(f"**Question:** {q_entry.get('question')}")
        st.markdown(f"**Difficulty:** `{q_entry.get('difficulty')}` | **Type:** `{q_entry.get('question_type')}`")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Generated Answer")
            st.text_area("LLM Output:", value=q_entry.get("generated_answer", ""), height=220, disabled=True)
            
        with c2:
            st.markdown("#### Judge Metrics & Reasoning")
            faith = q_entry.get("faithfulness", {})
            rel = q_entry.get("relevance", {})
            ret = q_entry.get("retrieval_metrics", {})
            
            st.markdown(f"**Faithfulness Label:** `{faith.get('label')}`")
            st.info(f"**Faithfulness Reasoning:** {faith.get('reasoning')}")
            
            st.markdown(f"**Relevance Score:** `{rel.get('score')} / 5`")
            st.warning(f"**Relevance Reasoning:** {rel.get('reasoning')}")
            
            st.markdown(f"**Retrieved Chunks:** `{q_entry.get('retrieved_chunk_ids')}` (Hit@3: `{ret.get('hit_at_k')}`, MRR: `{ret.get('mrr')}`)")

st.markdown("---")

# -----------------------------------------------------------------------------
# Section 5: Failure Spotlight
# -----------------------------------------------------------------------------
st.subheader("💡 Failure Spotlight & Key Learnings")
st.caption("Distinctly styled case studies highlighting non-trivial evaluation failure patterns discovered during the run.")

col_spot1, col_spot2 = st.columns(2)

with col_spot1:
    st.markdown("""
    <div class="spotlight-warning">
        <div class="spotlight-title">⚠️ Pattern 1: Conceptual Retrieval Miss (q003)</div>
        <p><b>Question:</b> <i>Definition of non-parametric vs parametric learning in locally weighted linear regression</i></p>
        <p><b>What Went Wrong:</b> The retriever fetched general linear regression chunks (<code>chunk_032</code>, <code>chunk_002</code>) instead of the exact definition chunk (<code>chunk_008</code>), resulting in <b>Hit@3 = False (0.0 recall)</b>.</p>
        <p><b>Key Insight:</b> Vector embedding models can over-prioritize dominant domain keywords ("linear regression") and miss nuanced conceptual definitions.</p>
    </div>
    """, unsafe_allow_html=True)

with col_spot2:
    st.markdown("""
    <div class="spotlight-warning">
        <div class="spotlight-title">⚠️ Pattern 2: Generator Degeneration Loop (q031)</div>
        <p><b>Question:</b> <i>Step-by-step derivation of the sigmoid function derivative</i></p>
        <p><b>What Went Wrong:</b> Perfect retrieval (<b>Hit@3 = True, MRR = 1.0</b>), but <code>llama-3.1-8b-instant</code> entered an infinite algebraic repetition loop, repeatedly rewriting equivalent expressions until token truncation.</p>
        <p><b>Key Insight:</b> Faithfulness alone (labeled <code>partially_grounded</code>) misses degenerations. Multi-metric evaluation is crucial: the relevance judge caught it as a total failure (<b>Score: 1/5</b>).</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="spotlight-card">
    <div class="spotlight-title">⚙️ Pattern 3: API Token Quota Constraints & Resiliency</div>
    <p><b>Constraint:</b> LLM-as-judge evaluation (<code>llama-3.3-70b-versatile</code>) hit Groq's 100k TPD daily quota after 17 questions (~34 judge calls) during initial run.</p>
    <p><b>Solution Implemented:</b> Added <code>--resume</code> flag to <code>run_eval.py</code> with incremental persistence, chunk truncation (~800 words), and 2-3 sentence reasoning length caps to complete all 45 questions within quota.</p>
</div>
""", unsafe_allow_html=True)

# Footer
st.caption("RAG Evaluator Dashboard • Built with Streamlit & Plotly")
