# ============================================================
# utils/gap_analyzer.py
#
# PURPOSE: Utility functions for gap analysis, resume parsing,
#          and generating detailed gap reports using LLM.
# ============================================================

import os
import sys
from langchain_groq          import ChatGroq
from langchain_core.prompts  import ChatPromptTemplate

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GROQ_MODEL_NAME


def get_llm(api_key: str, temperature: float = 0.2) -> ChatGroq:
    """Initialize Groq LLM — used across all utility functions."""
    return ChatGroq(
        api_key=api_key,
        model     = GROQ_MODEL_NAME,
        temperature = temperature,
        max_tokens  = 3000,
    )


def extract_skills_from_text(text: str,doc_type: str = "resume",api_key: str = "") -> list:
    """
    Use LLM to extract all technical skills mentioned in text.

    Args:
        text:     Raw text from resume or JD
        doc_type: "resume" or "jd"

    Returns:
        List of skill strings
    """
    llm = get_llm(api_key, temperature=0.1)

    if doc_type == "resume":
        instruction = "Extract ALL technical skills, tools, frameworks, and languages from this resume."
    else:
        instruction = "Extract ALL required technical skills, tools, and technologies from this job description."

    prompt = f"""{instruction}

TEXT:
{text[:3000]}

Return ONLY a comma-separated list of skills. No explanations.
Example format: Python, Machine Learning, SQL, TensorFlow, Docker, REST APIs

SKILLS:"""

    response = llm.invoke(prompt)
    # Parse the comma-separated skills into a clean list
    skills_text = response.content.strip()
    skills      = [s.strip() for s in skills_text.split(",") if s.strip()]
    return skills


def generate_gap_report(
    resume_skills : list,
    jd_skills     : list,
    resume_text   : str,
    jd_text       : str,
    role          : str,
    level         : str,
    api_key: str
) -> str:
    """
    Generate a detailed gap analysis report comparing resume to JD.

    Returns a formatted markdown string with:
    - Matching skills (green zone)
    - Missing skills (red zone — need to learn)
    - Improvement suggestions per missing skill
    """
    llm = get_llm(api_key, temperature=0.2)

    # Compute sets for matching/missing
    resume_set  = set(s.lower() for s in resume_skills)
    jd_set      = set(s.lower() for s in jd_skills)
    matching    = sorted(resume_set & jd_set)
    missing     = sorted(jd_set - resume_set)

    prompt = f"""You are an expert career advisor. Analyze this gap between a candidate's resume and a job description.

TARGET ROLE: {role} ({level})

MATCHING SKILLS (candidate has these): {', '.join(matching) or 'None identified'}
MISSING SKILLS (JD requires but resume lacks): {', '.join(missing) or 'None identified'}

JD EXCERPT:
{jd_text[:1500]}

RESUME EXCERPT:
{resume_text[:1500]}

Generate a detailed gap analysis report in this EXACT format:

## 📊 Resume–JD Gap Analysis Report

### ✅ Your Strengths ({len(matching)} skills matched)
List each matching skill and in 1 line why it is valuable for {role}

### ❌ Critical Gaps ({len(missing)} skills missing)
For each missing skill:
**[Skill Name]** — Priority: [High/Medium/Low]
- Why it matters: Why this skill is important for {role}
- How to learn it: Specific free resource (YouTube channel or course name)
- Time to learn: Estimated time for basics

### 📈 Overall Match Assessment
Give a percentage estimate and honest feedback on candidacy.

### 🎯 Top 3 Actions to Take This Week
Specific, actionable steps the candidate should start immediately.
"""

    response = llm.invoke(prompt)
    return response.content


def rewrite_full_resume_section(
    section_text: str,
    role: str,
    api_key: str
):
    """
    Rewrite an entire resume section (e.g., Projects or Experience)
    to be more impactful for the target role.

    Args:
        section_text: The original section text from resume
        role:         Target job role

    Returns:
        Rewritten section as string
    """
    llm = get_llm(api_key, temperature=0.3)

    prompt = f"""You are an expert resume writer specializing in tech roles.
Rewrite the following resume section for a {role} position.

ORIGINAL SECTION:
{section_text}

REWRITING RULES:
1. Every bullet must start with a strong action verb
   (Built, Engineered, Developed, Optimized, Architected, Implemented, Deployed, Designed)
2. Use the STAR format: Situation → Task → Action → Result (implied)
3. Quantify everything possible (users, accuracy %, speed improvement, data size)
4. Include relevant technical keywords that ATS systems look for in {role} jobs
5. Remove vague language like "helped with", "worked on", "involved in"
6. Keep each bullet to 1-2 lines maximum

REWRITTEN SECTION:"""

    response = llm.invoke(prompt)
    return response.content


def calculate_ats_score(resume_text, jd_text, api_key):
    """
    Calculate ATS (Applicant Tracking System) compatibility score.
    ATS systems scan resumes for keyword matches with JD.

    Returns dict with score and detailed feedback.
    """
    llm = get_llm(api_key, temperature=0.1)

    prompt = f"""You are an ATS (Applicant Tracking System) simulator.
Analyze how well this resume would perform when scanned by an ATS for this job.

JOB DESCRIPTION:
{jd_text[:1500]}

RESUME:
{resume_text[:1500]}

Evaluate and return a JSON-like structured response:

ATS_SCORE: [number 0-100]
KEYWORD_MATCHES: [list of JD keywords found in resume]
KEYWORD_MISSES: [list of important JD keywords NOT in resume]
FORMAT_ISSUES: [any formatting problems that hurt ATS parsing]
TOP_RECOMMENDATIONS: [3 specific changes to improve ATS score]

Be precise and realistic with the score."""

    response = llm.invoke(prompt)
    return {"report": response.content}
