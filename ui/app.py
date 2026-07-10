# ============================================================
# ui/app.py — Main Streamlit Application for JobLens AI
#
# HOW TO RUN:
#   streamlit run ui/app.py
#
# STRUCTURE:
#   Tab 1 — 📊 Skill Trends      : ML model predictions + SHAP chart
#   Tab 2 — 📄 Resume Analyser   : Upload resume + JD → match score + gap analysis
#   Tab 3 — 🤖 Career Coach      : LangGraph agent chatbot
#   Tab 4 — ✍️  Resume Rewriter   : Rewrite weak bullet points
# ============================================================

import os

# Disable parallel native libraries
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import sys
import streamlit as st
import pandas    as pd
import plotly.express        as px
import plotly.graph_objects  as go

# Add project root to Python path so all imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



from models.skill_model import predict_top_skills, generate_shap_chart, load_artifacts
from rag.resume_jd_rag  import (
    get_embeddings, load_pdf_to_chunks, build_vectorstore,
    calculate_match_score, get_retriever, extract_resume_info
)
from utils.gap_analyzer import (
    extract_skills_from_text, generate_gap_report,
    rewrite_full_resume_section, calculate_ats_score
)
from agent.career_agent import run_career_agent
from langchain_groq     import ChatGroq


# ─────────────────────────────────────────────────────────────
# PAGE CONFIG — Must be first Streamlit command
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "JobLens AI",
    page_icon  = "💼",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

if "groq_api_key" not in st.session_state:
    st.session_state["groq_api_key"] = ""

user_api_key = st.sidebar.text_input(
    "🔑 Enter your Groq API Key",
    type="password"
)

if st.sidebar.button("Connect API"):
    st.session_state["groq_api_key"] = user_api_key
    
# ─────────────────────────────────────────────────────────────
# CUSTOM CSS — Clean, professional dark-friendly styling
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6C63FF, #FF6584);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #888;
        margin-bottom: 1.5rem;
    }
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #6C63FF44;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    /* Skill badge */
    .skill-badge {
        display: inline-block;
        background: #6C63FF22;
        border: 1px solid #6C63FF66;
        color: #A89FFF;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        margin: 3px;
    }
    /* Chat message styling */
    .chat-user {
    background-color: #23244d;
    color: white !important;
    padding: 15px;
    border-radius: 12px;
    border-left: 5px solid #4F7CFF;
    margin: 12px 0;
    font-size: 16px;
    line-height: 1.6;
}

    .chat-user strong {
        color: white !important;
    }

    .chat-ai {
        background-color: #17351d;
        color: white !important;
        padding: 15px;
        border-radius: 12px;
        border-left: 5px solid #2ECC71;
        margin: 12px 0;
        font-size: 16px;
        line-height: 1.6;
    }

    .chat-ai strong {
        color: white !important;
    }
            
    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #A89FFF;
        border-bottom: 1px solid #333;
        padding-bottom: 0.4rem;
        margin: 1rem 0 0.8rem;
    }
    /* Match score display */
    .match-score-high   { color: #4CAF50; font-size: 2rem; font-weight: 700; }
    .match-score-medium { color: #FF9800; font-size: 2rem; font-weight: 700; }
    .match-score-low    { color: #F44336; font-size: 2rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# CACHED RESOURCES — Load once, reuse across reruns
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embeddings_cached():
    """
    Cache the embedding model so it is loaded only ONCE.
    Loading sentence-transformers takes ~3 seconds — caching avoids
    this delay on every user interaction.
    """
    return get_embeddings()


@st.cache_resource(show_spinner="Loading skill prediction model...")
def load_ml_model_cached():
    """Cache the XGBoost model artifacts for fast repeated predictions."""
    try:
        return load_artifacts()
    except FileNotFoundError:
        return None


# ─────────────────────────────────────────────────────────────
# SIDEBAR — User settings and information
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💼 JobLens AI")
    st.markdown("*Your AI-powered career intelligence platform*")
    st.divider()

    # User sets their target role and level here — used across all tabs
    st.markdown("### 🎯 Your Career Target")
    target_role = st.selectbox(
        "Target Job Role",
        options=[
            "ML Engineer",
            "Data Scientist",
            "Data Analyst",
            "Data Engineer",
            "Software Engineer",
            "DevOps / Cloud Engineer",
            "NLP / GenAI Engineer",
            "Product Manager",
        ],
        index=0,
        help="Select the role you are applying for"
    )

    target_level = st.selectbox(
        "Experience Level",
        options=["Fresher", "Mid-Level", "Senior"],
        index=0,
    )

    st.divider()

    # Check if API key is configured
    if st.session_state["groq_api_key"]:
        st.success("✅ Groq API Connected")
    else:
        st.warning("⚠️ Please enter your Groq API Key above.")

    # Check if ML model is trained
    ml_model = load_ml_model_cached()
    if ml_model:
        st.success("✅ ML model loaded")
    else:
        st.warning("⚠️ ML model not trained.\nRun: python models/skill_model.py")

    st.divider()
    st.markdown("**Quick Links**")
    st.markdown("📊 [Download LinkedIn Dataset](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings)")
    st.markdown("🔑 [Get Free Groq API Key](https://console.groq.com)")
    st.markdown("📚 [HuggingFace Models](https://huggingface.co/sentence-transformers)")


# ─────────────────────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">💼 JobLens AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Intelligent Job Market Analyser · Resume Coach · Career Intelligence Platform</div>',
            unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TABS — 4 main sections of the app
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Skill Trends",
    "📄 Resume Analyser",
    "🤖 Career Coach",
    "✍️ Resume Rewriter",
])


# ══════════════════════════════════════════════════════════════
# TAB 1 — SKILL TRENDS (ML Model Predictions)
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### 📊 In-Demand Skills Predictor")
    st.markdown("*Powered by XGBoost trained on 33,000+ real LinkedIn job postings*")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("#### Select Parameters")
        # These are pre-filled from sidebar but can be overridden here
        t1_role  = st.selectbox("Role", ["ML Engineer", "Data Scientist", "Data Analyst",
                                          "Data Engineer", "Software Engineer",
                                          "DevOps / Cloud Engineer", "NLP / GenAI Engineer"],
                                  index=0, key="t1_role")
        t1_level = st.selectbox("Level", ["Fresher", "Mid-Level", "Senior"],
                                 index=0, key="t1_level")
        top_n    = st.slider("Number of skills to show", 5, 20, 12)

        predict_btn = st.button("🔮 Predict Top Skills", type="primary", use_container_width=True)

    with col2:
        if predict_btn:
            if ml_model is None:
                st.error("❌ ML model not trained yet!\n\n"
                         "**Steps to fix:**\n"
                         "1. Download dataset from Kaggle (link in sidebar)\n"
                         "2. Place files in `data/raw/`\n"
                         "3. Run: `python data/prepare_dataset.py`\n"
                         "4. Run: `python models/skill_model.py`")
            else:
                with st.spinner("Analysing job market trends..."):
                    # Get predictions from ML model
                    skills_df = predict_top_skills(t1_role, t1_level, top_n)

                # Display as horizontal bar chart using Plotly
                fig = px.bar(
                    skills_df,
                    x="probability_pct",
                    y="skill",
                    orientation="h",
                    color="probability_pct",
                    color_continuous_scale=["#FF6584", "#6C63FF"],
                    labels={"probability_pct": "Demand Score (%)", "skill": "Skill"},
                    title=f"Top {top_n} Skills for {t1_role} ({t1_level})",
                )
                fig.update_layout(
                    yaxis=dict(autorange="reversed"),  # Highest on top
                    coloraxis_showscale=False,
                    height=420,
                    margin=dict(l=10, r=10, t=40, b=10),
                )
                st.plotly_chart(fig, use_container_width=True)

                # Show skills as badges
                st.markdown("**Skills as Tags:**")
                badges_html = " ".join([
                    f'<span class="skill-badge">#{row["skill"]}</span>'
                    for _, row in skills_df.iterrows()
                ])
                st.markdown(badges_html, unsafe_allow_html=True)

        else:
            # Show placeholder when no prediction yet
            st.info("👈 Select a role and level, then click **Predict Top Skills**")

    # SHAP Chart section
    st.divider()
    st.markdown("#### 🔍 SHAP Explainability — Why These Skills?")
    st.markdown("SHAP shows which input features (Role vs Experience Level) most influenced the predictions.")

    if st.button("Generate SHAP Chart", key="shap_btn"):
        if ml_model is None:
            st.warning("Train the model first!")
        else:
            with st.spinner("Computing SHAP values..."):
                fig = generate_shap_chart(t1_role, t1_level)
            st.pyplot(fig)
            st.caption("Higher SHAP value = that feature had more influence on skill predictions")


# ══════════════════════════════════════════════════════════════
# TAB 2 — RESUME ANALYSER
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📄 Resume × Job Description Analyser")
    st.markdown("*Upload your resume and a JD to get match score, gap analysis, and ATS score*")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### 📄 Upload Your Resume")
        resume_file = st.file_uploader(
            "Upload Resume (PDF)",
            type=["pdf"],
            key="resume_uploader",
            help="Upload your resume as a PDF file"
        )

        if resume_file:
            st.success(f"✅ Resume uploaded: {resume_file.name}")

    with col_right:
        st.markdown("#### 📋 Upload Job Description")
        jd_file = st.file_uploader(
            "Upload JD (PDF) or paste below",
            type=["pdf"],
            key="jd_uploader",
            help="Upload the job description as PDF"
        )
        # Alternative: paste JD as text
        jd_text_input = st.text_area(
            "OR paste Job Description text here",
            height=150,
            placeholder="Paste the job description text here if you don't have a PDF...",
            key="jd_text"
        )

    # Analyse button
    analyse_btn = st.button(
        "🔍 Analyse Resume vs JD",
        type="primary",
        use_container_width=True,
        disabled=(resume_file is None)
    )

    if analyse_btn and resume_file:
        embeddings = load_embeddings_cached()

        with st.spinner("📖 Reading and chunking resume..."):
            # Load and chunk the resume PDF
            resume_bytes  = resume_file.read()
            resume_chunks = load_pdf_to_chunks(resume_bytes, doc_type="resume")
            resume_text   = "\n".join([c.page_content for c in resume_chunks])

            # Store resume text in session state
            st.session_state["resume_text"] = resume_text

            # Resume vector store is temporarily disabled
            st.session_state["resume_store"] = None

        with st.spinner("📖 Reading job description..."):
            # Handle JD from file or text input
            if jd_file:
                jd_bytes  = jd_file.read()
                jd_chunks = load_pdf_to_chunks(jd_bytes, doc_type="jd")
                jd_text   = "\n".join([c.page_content for c in jd_chunks])
            elif jd_text_input.strip():
                jd_text   = jd_text_input.strip()
                # Wrap pasted text as Document chunks manually
                from langchain.schema import Document
                from langchain.text_splitter import RecursiveCharacterTextSplitter
                splitter  = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=120)
                jd_chunks = splitter.create_documents([jd_text])
            else:
                st.error("Please upload a JD PDF or paste JD text!")
                st.stop()

            st.session_state["jd_content"] = jd_text

        # ── Match Score Calculation ──────────────────────────
        with st.spinner("🎯 Calculating match score..."):
            match_score, matched_snippets = calculate_match_score(
                resume_chunks, jd_chunks, embeddings
            )

        # ── Display Match Score ──────────────────────────────
        st.markdown("---")
        m1, m2, m3 = st.columns(3)

        with m1:
            score_class = (
                "match-score-high"   if match_score >= 70 else
                "match-score-medium" if match_score >= 45 else
                "match-score-low"
            )
            verdict = (
                "🟢 Strong Match"  if match_score >= 70 else
                "🟡 Partial Match" if match_score >= 45 else
                "🔴 Weak Match"
            )
            st.markdown(f"""
            <div class="metric-card">
                <div style="color:#aaa; font-size:0.85rem">JD Match Score</div>
                <div class="{score_class}">{match_score}%</div>
                <div style="color:#888; font-size:0.8rem">{verdict}</div>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div style="color:#aaa; font-size:0.85rem">Resume Chunks</div>
                <div style="font-size:2rem; font-weight:700; color:#A89FFF">{len(resume_chunks)}</div>
                <div style="color:#888; font-size:0.8rem">sections indexed</div>
            </div>
            """, unsafe_allow_html=True)

        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div style="color:#aaa; font-size:0.85rem">JD Chunks</div>
                <div style="font-size:2rem; font-weight:700; color:#FF6584">{len(jd_chunks)}</div>
                <div style="color:#888; font-size:0.8rem">requirements indexed</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Skill Gap Analysis ───────────────────────────────
        st.markdown("---")

        if not st.session_state["groq_api_key"]:
            st.error("Please connect your Groq API Key first.")
            st.stop()

        with st.spinner("🧠 Running gap analysis with AI..."):
            llm = ChatGroq(
                                api_key=st.session_state["groq_api_key"],
                                model="llama-3.3-70b-versatile",
                                temperature=0.1
                                )
            
            resume_skills = extract_skills_from_text(resume_text,"resume",st.session_state["groq_api_key"])
            jd_skills = extract_skills_from_text(jd_text, "jd", st.session_state["groq_api_key"])
            gap_report = generate_gap_report(
                                                resume_skills,
                                                jd_skills,
                                                resume_text,
                                                jd_text,
                                                target_role,
                                                target_level,
                                                st.session_state["groq_api_key"]
                                                )
            
            # Store gap report for use in Career Coach tab
            st.session_state["gap_analysis"] = gap_report

        st.markdown("### 📊 Gap Analysis Report")
        st.markdown(gap_report)

        # ── ATS Score ───────────────────────────────────────
        st.markdown("---")
        with st.spinner("🤖 Calculating ATS compatibility score..."):
            ats_result = calculate_ats_score(resume_text,jd_text,st.session_state["groq_api_key"])

        st.markdown("### 🤖 ATS Compatibility Analysis")
        st.markdown(ats_result["report"])

        # ── Top Matching Snippets ────────────────────────────
        with st.expander("🔍 View Top Matching Resume Sections"):
            for i, snippet in enumerate(matched_snippets[:4], 1):
                st.markdown(f"**Match {i}:**")
                st.text(snippet[:300])
                st.divider()


# ══════════════════════════════════════════════════════════════
# TAB 3 — CAREER COACH CHATBOT (LangGraph Agent)
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 🤖 Career Coach Agent")
    st.markdown("*Powered by LangGraph — asks web, reads your resume, gives real advice*")

    # Initialize chat history in session state if not exists
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Check if resume has been analysed (needed for context)
    resume_ctx = st.session_state.get("resume_text", "")
    gap_ctx    = st.session_state.get("gap_analysis", "")

    if not resume_ctx:
        st.info("💡 **Tip:** Go to **📄 Resume Analyser** tab first and upload your resume "
                "to give the agent full context about you. You can still chat without it!")

    # Suggested queries to help the user get started
    st.markdown("**💡 Try asking:**")
    sugg_cols = st.columns(3)
    suggestions = [
        "How do I make my RAG project sound stronger?",
        "What skills am I missing for ML Engineer role?",
        "Give me a 30-day plan to learn LangChain",
    ]
    for i, (col, suggestion) in enumerate(zip(sugg_cols, suggestions)):
        with col:
            if st.button(suggestion, key=f"sugg_{i}", use_container_width=True):
                st.session_state["pending_query"] = suggestion

    st.divider()

    # Display chat history
    for msg in st.session_state["chat_history"]:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">👤 <strong>You:</strong> {msg["content"]}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-ai">🤖 <strong>JobLens AI:</strong><br>{msg["content"]}</div>',
                        unsafe_allow_html=True)

    # Chat input
    user_input = st.chat_input(
        "Ask me anything — resume tips, skill gaps, learning resources, career advice...",
        key="career_chat_input"
    )

    # Handle pending query from suggestion buttons
    if "pending_query" in st.session_state:
        user_input = st.session_state.pop("pending_query")

    # Process user input
    if user_input and user_input.strip():
        # Add user message to history
        st.session_state["chat_history"].append({"role": "user", "content": user_input})

        # Show thinking spinner
        with st.spinner("🤔 Career Coach is thinking and searching the web..."):
            response = run_career_agent(
                api_key=st.session_state["groq_api_key"],
                user_query     = user_input,
                resume_context = resume_ctx,
                gap_analysis   = gap_ctx,
                role_target    = target_role,
                level_target   = target_level,
                chat_history   = st.session_state["chat_history"][:-1],  # Exclude current
            )

        # Add AI response to history
        st.session_state["chat_history"].append({"role": "assistant", "content": response})

        # Rerun to show updated chat
        st.rerun()

    # Clear chat button
    if st.session_state["chat_history"]:
        if st.button("🗑️ Clear Chat History", key="clear_chat"):
            st.session_state["chat_history"] = []
            st.rerun()


# ══════════════════════════════════════════════════════════════
# TAB 4 — RESUME REWRITER
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### ✍️ Resume Bullet Point Rewriter")
    st.markdown("*Paste weak bullet points and get 3 powerful rewritten versions*")

    col_rw1, col_rw2 = st.columns(2)

    with col_rw1:
        st.markdown("#### Original Bullet Point")
        original_bullet = st.text_area(
            "Paste your weak bullet point here",
            height=150,
            placeholder=(
                "Example:\n"
                "- Worked on a machine learning project\n"
                "- Helped in developing a RAG system\n"
                "- Was involved in backend development"
            ),
            key="original_bullet"
        )

        rw_role = st.selectbox(
            "Target Role",
            ["ML Engineer", "Data Scientist", "Software Engineer",
             "Data Analyst", "Data Engineer", "NLP / GenAI Engineer"],
            key="rw_role"
        )

        rewrite_btn = st.button(
            "✨ Rewrite This Bullet",
            type="primary",
            use_container_width=True,
            key="rewrite_btn"
        )

    with col_rw2:
        st.markdown("#### Rewritten Versions")

        if rewrite_btn and original_bullet.strip():
            with st.spinner("Crafting powerful bullet points..."):
                rewritten = rewrite_full_resume_section(original_bullet,rw_role,st.session_state["groq_api_key"])
            st.markdown(rewritten)

        elif rewrite_btn:
            st.warning("Please paste a bullet point first!")
        else:
            st.info("👈 Paste a bullet point and click Rewrite")

    # ── Full Section Rewriter ────────────────────────────────
    st.divider()
    st.markdown("#### 📝 Rewrite Entire Resume Section")
    st.markdown("Paste an entire section (e.g., all your project descriptions) for a full rewrite")

    full_section = st.text_area(
        "Paste entire section here",
        height=200,
        placeholder="Paste your Projects or Experience section here...",
        key="full_section"
    )

    if st.button("🔄 Rewrite Full Section", type="secondary", key="rewrite_full"):
        if full_section.strip():
            with st.spinner("Rewriting full section..."):
                rewritten_section = rewrite_full_resume_section(full_section,target_role,st.session_state["groq_api_key"])
            st.markdown("#### ✅ Rewritten Section:")
            st.markdown(rewritten_section)
            # Allow download of rewritten section
            st.download_button(
                "📥 Download Rewritten Section",
                data=rewritten_section,
                file_name="rewritten_resume_section.txt",
                mime="text/plain"
            )
        else:
            st.warning("Please paste a section first!")

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center; color:#555; font-size:0.8rem'>"
    "JobLens AI · Built with LangGraph + RAG + XGBoost + Groq (Free) + Streamlit · "
    "All APIs used are FREE 🆓"
    "</div>",
    unsafe_allow_html=True
)
