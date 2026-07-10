# ============================================================
# config.py — Central Configuration for JobLens AI
# All settings loaded from .env file using python-dotenv
# ============================================================

import os
from dotenv import load_dotenv

# Load all variables from the .env file into os.environ
load_dotenv()

# ── LLM Settings ────────────────────────────────────────────
# Groq is free. Best free model right now is llama-3.3-70b-versatile
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")

# ── Embedding Settings ───────────────────────────────────────
# sentence-transformers runs locally — completely FREE, no API key needed
# all-MiniLM-L6-v2 is fast + accurate for resume/JD matching
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ── Directory Paths ──────────────────────────────────────────
RESUME_UPLOAD_DIR = os.getenv("RESUME_UPLOAD_DIR", "data/uploads/resumes")
JD_UPLOAD_DIR     = os.getenv("JD_UPLOAD_DIR",     "data/uploads/jds")
ML_DATASET_PATH   = os.getenv("ML_DATASET_PATH",   "data/raw/job_skills_dataset.csv")
MODEL_SAVE_PATH   = os.getenv("MODEL_SAVE_PATH",    "models/skill_trend_model.pkl")

# ── RAG / Chunking Settings ──────────────────────────────────
# For RESUMES: smaller chunks so skills/bullet points don't get split
RESUME_CHUNK_SIZE    = 400
RESUME_CHUNK_OVERLAP = 80

# For JOB DESCRIPTIONS: slightly larger chunks to keep context together
JD_CHUNK_SIZE    = 600
JD_CHUNK_OVERLAP = 120

# How many top chunks to retrieve when doing RAG search
RAG_TOP_K = 6

# ── ML Model Settings ────────────────────────────────────────
# XGBoost hyperparameters tuned for the skill prediction task
XGBOOST_PARAMS = {
    "n_estimators":   300,
    "max_depth":      6,
    "learning_rate":  0.05,
    "subsample":      0.8,
    "colsample_bytree": 0.8,
    "use_label_encoder": False,
    "eval_metric":    "mlogloss",
    "random_state":   42,
}

# ── Agent Settings ───────────────────────────────────────────
# Max iterations the LangGraph agent can loop before stopping
AGENT_MAX_ITERATIONS = 10

# How many web search results to fetch per query
WEB_SEARCH_RESULTS = 8

# ── Ensure all upload directories exist ─────────────────────
os.makedirs(RESUME_UPLOAD_DIR, exist_ok=True)
os.makedirs(JD_UPLOAD_DIR,     exist_ok=True)
os.makedirs("models",          exist_ok=True)
os.makedirs("data/raw",        exist_ok=True)
