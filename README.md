# 💼 JobLens AI — Intelligent Career Intelligence Platform

<div align="center">

<br/>

**An end-to-end AI-powered career intelligence platform combining Machine Learning, RAG, and Agentic AI to help students and professionals optimise their resumes, identify skill gaps, and accelerate their career growth.**

<br/>

[Features](#-key-features) · [Tech Stack](#-tech-stack) · [Architecture](#-system-architecture) · [Setup](#-setup--installation) · [Screenshots](#-screenshots) · [Project Structure](#-project-structure)

</div>

---

## 📌 Overview

**JobLens AI** is a comprehensive career intelligence system built entirely on **free APIs and open-source tools**. It intelligently analyses resumes against job descriptions, identifies missing skills, calculates ATS compatibility, predicts in-demand technical skills using a real-world trained XGBoost model, rewrites resume sections using LLMs, and provides personalised career guidance through a LangGraph-powered AI agent.

Unlike traditional resume analysers that rely solely on keyword matching, JobLens AI leverages **semantic similarity** through vector embeddings to understand the actual meaning of resume content and job descriptions — enabling more accurate, context-aware analysis and recommendations.

> 🎓 **Built as a placement portfolio project** — demonstrates depth across the full ML/GenAI stack using only free-tier APIs.

---

## ✨ Key Features

### 📊 Tab 1 — In-Demand Skills Predictor (Machine Learning)
- Predicts **top 15 in-demand technical skills** for any job role and experience level
- Powered by **XGBoost classifier** trained on **33,000+ real LinkedIn job postings**
- Displays skills as an **interactive Plotly bar chart** ranked by demand score
- **SHAP explainability chart** — shows whether Job Role or Experience Level drove the prediction more
- Supports 8 job roles: ML Engineer, Data Scientist, Data Analyst, Data Engineer, Software Engineer, DevOps/Cloud Engineer, NLP/GenAI Engineer, Product Manager

### 📄 Tab 2 — Resume Analyser (RAG Pipeline)
- Upload **Resume PDF** + Upload **JD PDF** or paste job description text directly
- Calculates **semantic similarity match score (0–100%)** between resume and JD
- Generates a detailed **Resume–JD Gap Analysis Report** with matching skills, missing skills, and priority levels
- **ATS Compatibility Analysis** — keyword match, keyword misses, formatting feedback, and improvement recommendations
- Displays top matching resume sections that contributed to the similarity score

### 🤖 Tab 3 — Career Coach Agent (LangGraph + Agentic AI)
- Powered by a **multi-tool LangGraph agent** — not a simple one-shot LLM call
- Agent **reasons → decides → searches web → observes → answers** in a loop
- Performs **real-time DuckDuckGo web search** — no API key required
- Generates **personalised 30-day learning roadmaps** with real YouTube links and course recommendations
- Maintains **conversation memory** — context of resume and gap analysis carried throughout the session

### ✍️ Tab 4 — Resume Rewriter (GenAI)
- Rewrites weak resume bullet points using **strong action verbs** and **quantified impact**
- Follows **STAR methodology** (Situation → Task → Action → Result)
- Returns **3 rewritten versions** of any bullet point per request
- Rewrites **entire resume sections** for maximum ATS keyword coverage
- Offers **download option** for the rewritten content

---

## 🛠 Tech Stack

| Category | Technology | Purpose |
|---|---|---|
| **LLM** | Groq API — LLaMA 3.3 70B Versatile | Gap analysis, resume rewriting, career coaching |
| **Embeddings** | HuggingFace all-MiniLM-L6-v2 | Text → semantic vectors (runs locally, free) |
| **Vector DB** | FAISS (Facebook AI) | In-memory vector store for resume/JD similarity search |
| **PDF Parsing** | PyPDFLoader (LangChain) | Extract text from uploaded PDF files |
| **Chunking** | RecursiveCharacterTextSplitter | Split documents into semantic chunks |
| **Agent** | LangGraph | Multi-step reasoning agent with tool loop |
| **ML Model** | XGBoost + Scikit-learn | Skill trend prediction from job postings |
| **Explainability** | SHAP | Explain which feature drove the ML prediction |
| **Web Search** | DuckDuckGo Search | Real-time search — no API key required |
| **Frontend** | Streamlit | 4-tab interactive dashboard |
| **Charts** | Plotly Express + Matplotlib | Interactive skill chart + SHAP bar chart |
| **Config** | Python Dotenv | Secure environment variable management |
| **Data** | Pandas + NumPy | Dataset cleaning and processing |

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT FRONTEND                        │
│   Tab 1: Skill Trends | Tab 2: Resume Analyser              │
│   Tab 3: Career Coach | Tab 4: Resume Rewriter              │
└──────────────┬──────────────────────────────┬───────────────┘
               │                              │
    ┌──────────▼──────────┐       ┌──────────▼──────────┐
    │    RAG PIPELINE      │       │   ML PIPELINE        │
    │                      │       │                      │
    │  PDF Upload          │       │  LinkedIn Dataset    │
    │      ↓               │       │       ↓              │
    │  PyPDFLoader         │       │  Data Cleaning       │
    │      ↓               │       │       ↓              │
    │  Text Chunking       │       │  LabelEncoder        │
    │  (400 / 600 chars)   │       │       ↓              │
    │      ↓               │       │  XGBoost Training    │
    │  HuggingFace         │       │       ↓              │
    │  Embeddings          │       │  skill_trend.pkl     │
    │      ↓               │       │       ↓              │
    │  FAISS Vector DB     │       │  Predict + SHAP      │
    │      ↓               │       └──────────────────────┘
    │  Cosine Similarity   │
    │      ↓               │       ┌──────────────────────┐
    │  Groq LLaMA 3.3 70B  │       │  LANGGRAPH AGENT     │
    │      ↓               │       │                      │
    │  Gap Report + ATS    │       │  agent_node          │
    └──────────────────────┘       │  (Groq LLM decides)  │
                                   │       ↓              │
                                   │  tools_condition     │
                                   │  ┌────────────────┐  │
                                   │  │ web_search     │  │
                                   │  │ rewrite_bullet │  │
                                   │  │ skill_plan     │  │
                                   │  │ gap_analysis   │  │
                                   │  └────────────────┘  │
                                   │       ↓              │
                                   │  tool_executor_node  │
                                   │       ↓              │
                                   │  Loop back to agent  │
                                   │       ↓              │
                                   │  Final Answer → UI   │
                                   └──────────────────────┘
```

---

## 📂 Project Structure

```
joblens_ai/
│
├── 📄 config.py                    # Central config — all settings, paths, hyperparameters
├── 📄 requirements.txt             # All dependencies with pinned versions
├── 📄 .env.example                 # Template — copy to .env and add API key
├── 📄 README.md
│
├── 📁 data/
│   ├── prepare_dataset.py          # Merge, clean, standardize LinkedIn Kaggle CSVs
│   └── raw/                        # Place Kaggle CSVs here
│       ├── job_postings.csv        ← Download from Kaggle
│       └── job_skills.csv          ← Download from Kaggle
│
├── 📁 models/
│   ├── skill_model.py              # Train XGBoost + SHAP explainability
│   └── skill_trend_model.pkl       # Auto-generated after training
│
├── 📁 rag/
│   └── resume_jd_rag.py            # PDF loader + chunking + FAISS + match score
│
├── 📁 agent/
│   └── career_agent.py             # LangGraph agent — 4 tools, reasoning loop
│
├── 📁 utils/
│   └── gap_analyzer.py             # LLM calls — gap report, ATS score, rewriter
│
└── 📁 ui/
    └── app.py                      # Main Streamlit app — 4 tabs, full UI
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+
- Conda (recommended) or virtualenv
- Free Groq API key from [console.groq.com](https://console.groq.com)
- LinkedIn Job Postings dataset from [Kaggle](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings)

---

### Step 1 — Clone the Repository

```bash
git clone https://github.com/your-username/joblens-ai.git
cd joblens-ai
```

### Step 2 — Create and Activate Environment

```bash
# Using Conda (recommended)
conda create -n joblens python=3.10
conda activate joblens

# OR using virtualenv
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Configure Environment Variables

```bash
cp .env.example .env
```

Open `.env` and add your free Groq API key:

```env
GROQ_API_KEY=gsk_your_actual_key_here
GROQ_MODEL_NAME=llama-3.3-70b-versatile
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

> 🔑 Get your **free** Groq API key at [console.groq.com](https://console.groq.com) — no credit card required.

### Step 5 — Download the Dataset

1. Go to [LinkedIn Job Postings Dataset on Kaggle](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings)
2. Download and unzip
3. Place these two files into `data/raw/`:
   - `job_postings.csv`
   - `job_skills.csv`

### Step 6 — Prepare Dataset

```bash
python data/prepare_dataset.py
```

Expected output:
```
✅ Dataset prepared and saved
   Total rows: 280,000+
   Unique roles: 8
   Unique skills: 250+
```

### Step 7 — Train the ML Model

```bash
python models/skill_model.py
```

Training takes **2–4 minutes**. Expected output:
```
🤖 Training XGBoost model...
[50]  validation_0-mlogloss: 0.92
[100] validation_0-mlogloss: 0.61
...
✅ Model saved to: models/skill_trend_model.pkl
```

### Step 8 — Launch the Application

```bash
streamlit run ui/app.py
```

Open your browser at **[http://localhost:8501](http://localhost:8501)**

---

## 📸 Screenshots

### 📊 In-Demand Skills Predictor — Skill Trends Tab

> XGBoost predicts top skills for Data Scientist (Fresher) with interactive demand chart

![Skill Trends](screenshots/skill_trends.png)
<img width="1470" height="810" alt="Screenshot 2026-07-06 at 10 03 33 PM" src="https://github.com/user-attachments/assets/a1fb7760-57e5-446d-aa1a-723776b57277" />

---

### 📄 Resume Analyser — JD Match Score + Gap Analysis

> Resume uploaded, JD pasted → JD Match Score: 6.7%, 23 resume chunks indexed

![Resume Analyser](screenshots/resume_analyser.png)
<img width="1465" height="754" alt="Screenshot 2026-07-06 at 10 04 06 PM" src="https://github.com/user-attachments/assets/27f06014-c3e0-4988-b5a3-b21f9966fa23" />

---

### 🤖 Career Coach Agent — LangGraph

> User asked "What skills am I missing for ML Engineer role?" → Agent searched web, returned real YouTube and Coursera links

![Career Coach](screenshots/career_coach.png)
<img width="1122" height="785" alt="Screenshot 2026-07-06 at 10 05 44 PM" src="https://github.com/user-attachments/assets/58b53dc6-537d-49a0-ad50-51fff8b7e98d" />

---

### ✍️ Resume Rewriter

> Weak bullet "Worked on machine learning project" → 3 powerful ATS-optimised rewrites

![Resume Rewriter](screenshots/resume_rewriter.png)
<img width="1070" height="684" alt="Screenshot 2026-07-06 at 10 25 04 PM" src="https://github.com/user-attachments/assets/b20efda0-cac6-44ba-9dee-65d4242ef059" />

---

## 🆓 All Free — Zero Cost Breakdown

| Tool | Cost | Limit |
|---|---|---|
| Groq API (LLaMA 3.3 70B) | **FREE** | 14,400 requests/day |
| HuggingFace Embeddings | **FREE** | Runs locally, unlimited |
| DuckDuckGo Search | **FREE** | No limit, no API key |
| FAISS Vector DB | **FREE** | In-memory, unlimited |
| Streamlit | **FREE** | Local hosting |
| LinkedIn Dataset (Kaggle) | **FREE** | One-time download |

**Total monthly infrastructure cost: ₹0**

---

## 🤖 Machine Learning Pipeline

```
LinkedIn Job Postings CSV (33,000+ rows)
            ↓
data/prepare_dataset.py
  ├── Load job_postings.csv + job_skills.csv
  ├── Merge on job_id
  ├── categorize_role()  → Standardize 500+ job titles → 8 clean categories
  ├── normalize_level()  → Standardize experience levels → Fresher / Mid-Level / Senior
  └── Save → job_skills_dataset.csv
            ↓
models/skill_model.py
  ├── LabelEncoder  → role/level/skill text → integers
  ├── train_test_split(stratify=y, test_size=0.2)
  ├── XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05)
  ├── model.predict_proba() → probability for every skill class
  ├── SHAP TreeExplainer → feature importance visualization
  └── pickle.dump() → skill_trend_model.pkl
```

---

## 🔍 RAG Pipeline

```
User uploads Resume PDF + Job Description
            ↓
PyPDFLoader → Extract raw text from PDF pages
            ↓
RecursiveCharacterTextSplitter
  Resume → chunk_size=400, overlap=80
  JD     → chunk_size=600, overlap=120
            ↓
HuggingFace all-MiniLM-L6-v2
  → Convert each chunk to 384-dim embedding vector
  → normalize_embeddings=True (for cosine similarity)
            ↓
FAISS.from_documents()
  → Index all resume chunk vectors in memory
            ↓
Cosine Similarity Search
  → JD text → embed → search FAISS → top-6 resume chunks
  → dot product (unit vectors) = cosine similarity
  → average × 100 = match score %
            ↓
Groq LLaMA 3.3 70B
  → extract_skills_from_text() → skill lists
  → generate_gap_report()      → full gap analysis
  → calculate_ats_score()      → ATS compatibility
```

---

## 🕸 LangGraph Agent Architecture

```
User Query + Resume Context + Gap Analysis
            ↓
        agent_node
  (Groq LLM with 4 bound tools)
            ↓
    tools_condition?
    ┌──── YES ────┐
    ↓             ↓
tool_executor  END (final answer)
    │
    ├── web_search_learning_resources (DuckDuckGo)
    ├── rewrite_resume_bullet (Groq LLM call)
    ├── generate_skill_learning_plan (Groq LLM call)
    └── analyze_skill_gaps (Python set operations)
            ↓
    Result → back to agent_node
            ↓
    (loops until final answer)
            ↓
        END → response to UI
```

---

## 📋 Non-Functional Requirements & API Limits

| Requirement | Implementation |
|---|---|
| Groq free tier (14,400 req/day) | Max 1 LLM call per user action — no unnecessary chaining |
| DuckDuckGo soft limits | Agent capped at 2 searches per conversation turn |
| HuggingFace embeddings | Runs locally post-download — zero API calls, zero cost |
| Response time — skill prediction | < 2 seconds (loaded model, no API call) |
| Response time — resume analysis | < 30 seconds (embedding + Groq call) |
| Response time — agent | 15–45 seconds (includes web search) |
| Context window management | Resume/JD text truncated to 3000–4000 chars before LLM calls |
| FAISS persistence | In-memory only — rebuilt per session, keeps app stateless |
| Model training | Offline once → saved .pkl → loaded at startup in < 1 second |
| Thread safety | TOKENIZERS_PARALLELISM=false + OMP/MKL/NUMEXPR=1 at startup |
| Rare skill handling | Skills with < 2 samples removed before training (stratified split safety) |

---

## 🧠 Key Technical Decisions

### Why Groq over OpenAI?
Groq provides free access to LLaMA 3.3 70B Versatile with fast inference via custom LPU hardware. OpenAI charges per token from the first request — not viable for a student project with iterative development and testing cycles.

### Why local HuggingFace embeddings over OpenAI embeddings?
A resume and JD may produce 30+ chunks each. Embedding all of these via an API call would add monetary cost and network latency for every single analysis. The local model (all-MiniLM-L6-v2, ~90MB) runs in < 1 second per chunk on CPU.

### Why FAISS over Pinecone or Chroma?
The vector store only needs to live for one user session — a new resume is uploaded each time. FAISS as an in-memory library requires zero setup and zero cost. Pinecone requires cloud account setup; Chroma adds a server dependency neither of which are necessary here.

### Why LangGraph over a simple LangChain LCEL chain?
The career coach needs to search the web, evaluate results, potentially search again with a refined query, then synthesize an answer. This requires a loop. A fixed LCEL chain executes a predetermined sequence and cannot conditionally repeat steps — LangGraph's graph-based architecture is the correct tool.

### Why XGBoost over neural networks?
The task is structured tabular classification: two categorical inputs (role + level) → one categorical output (skill). XGBoost is the industry standard for this type of problem, trains in minutes on CPU, and integrates natively with SHAP for explainability — a neural network would add complexity with no accuracy benefit given only two input features.

---

## 📊 Dataset

| Property | Details |
|---|---|
| **Source** | [LinkedIn Job Postings 2023–24 — Kaggle](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) |
| **Files used** | `job_postings.csv` + `job_skills.csv` |
| **Size** | 33,000+ job postings |
| **After processing** | 280,000+ role–level–skill triplets |
| **Job categories** | 8 standardised role categories |
| **Experience levels** | 3 standardised levels (Fresher / Mid-Level / Senior) |
| **Usage** | Train XGBoost model for skill trend prediction |

---

## 🚀 Future Improvements

- [ ] Add LinkedIn job scraping for real-time JD fetching
- [ ] Persistent vector store across sessions using ChromaDB
- [ ] Resume scoring dashboard with history tracking
- [ ] Multi-language resume support
- [ ] Deploy to Streamlit Community Cloud

---

## 👨‍💻 Author

**Karan** — 3rd Year B.Tech Computer Science, Thapar University  
Built during internship at Indian Express Data Centre

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**⭐ If this project helped you, please give it a star!**

Built with ❤️ using LangGraph · LangChain · XGBoost · Groq · FAISS · Streamlit

</div>
