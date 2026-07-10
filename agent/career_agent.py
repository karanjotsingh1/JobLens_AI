# ============================================================
# agent/career_agent.py — v3 RECURSION FIX
#
# PROBLEM: Agent was looping infinitely because:
#   1. After tool executed, agent called ANOTHER tool instead
#      of writing the final answer — endless loop!
#   2. Recursion limit of 10 hit before final answer produced.
#
# ROOT CAUSE: System prompt said "ALWAYS call web_search first"
#   which made the agent call web_search AGAIN even after
#   generate_skill_learning_plan already fetched URLs internally.
#   Result: tool → tool → tool → tool → CRASH (recursion limit)
#
# FIXES APPLIED:
#   1. AGENT_MAX_ITERATIONS increased to 25 (safe headroom)
#   2. System prompt rewritten — agent told to call MAX 1-2 tools
#      then IMMEDIATELY write final answer using tool results
#   3. tool_executor_node handles ALL heavy work internally
#      (web search + plan generation in one step)
#      so agent never needs to call multiple tools for one query
#   4. Added tool_call_count tracking via system prompt message
#      to prevent agent from looping
# ============================================================

import os
import sys
from typing import Annotated, TypedDict, List
from langchain_groq           import ChatGroq
from langchain_core.messages  import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools     import tool
from langgraph.graph          import StateGraph, START, END
from langgraph.graph.message  import add_messages
from langgraph.prebuilt       import ToolNode, tools_condition
from duckduckgo_search        import DDGS

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GROQ_MODEL_NAME, WEB_SEARCH_RESULTS


# ─────────────────────────────────────────────────────────────
# 1. AGENT STATE
# ─────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages      : Annotated[List, add_messages]
    resume_context: str
    gap_analysis  : str
    role_target   : str
    level_target  : str
    api_key       : str


# ─────────────────────────────────────────────────────────────
# 2. TOOLS
# NOTE: Each tool is SELF-CONTAINED — does ALL the heavy work
#       internally so agent only needs to call ONE tool per query
#       and then write the final answer. No chaining needed.
# ─────────────────────────────────────────────────────────────

@tool
def web_search_learning_resources(query: str) -> str:
    """
    Search the web for FREE learning resources.
    Use this ONLY for general resource questions (not 30-day plans).
    Args:
        query: Search query string
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(
                query,
                max_results=WEB_SEARCH_RESULTS,
                region="in-en",
                safesearch="moderate"
            ))
        if not results:
            return "No results found."
        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(
                f"[{i}] TITLE: {r.get('title', 'N/A')}\n"
                f"    URL:   {r.get('href', 'N/A')}\n"
                f"    DESC:  {r.get('body', 'N/A')[:250]}\n"
            )
        return "\n".join(formatted)
    except Exception as e:
        return f"Search failed: {str(e)}"


@tool
def generate_skill_learning_plan(missing_skills: str, role: str, level: str) -> str:
    """
    Generate a complete 30-day learning plan WITH real URLs already included.
    This tool handles EVERYTHING internally — web search + plan generation.
    Use this for any 30-day plan request. Do NOT call web_search separately.
    Args:
        missing_skills: Skills to learn (comma-separated)
        role:           Target job role
        level:          Experience level
    """
    return f"PLAN_REQUEST|skills={missing_skills}|role={role}|level={level}"


@tool
def rewrite_resume_bullet(bullet_point: str, role: str) -> str:
    """
    Rewrite a weak resume bullet point into 3 strong ATS-friendly versions.
    Args:
        bullet_point: Original weak bullet point
        role:         Target job role
    """
    return f"REWRITE_REQUEST|bullet={bullet_point}|role={role}"


@tool
def analyze_skill_gaps(resume_skills: str, jd_requirements: str) -> str:
    """
    Compare resume skills vs JD requirements to find gaps.
    Args:
        resume_skills:   Comma-separated skills from resume
        jd_requirements: Comma-separated skills from JD
    """
    resume_set = set(s.strip().lower() for s in resume_skills.split(",") if s.strip())
    jd_set     = set(s.strip().lower() for s in jd_requirements.split(",") if s.strip())
    missing  = sorted(jd_set - resume_set)
    matching = sorted(jd_set & resume_set)
    extra    = sorted(resume_set - jd_set)
    return (
        f"MATCHING ({len(matching)}): {', '.join(matching) or 'None'}\n"
        f"MISSING  ({len(missing)}): {', '.join(missing)  or 'None'}\n"
        f"EXTRA    ({len(extra)}):   {', '.join(extra)    or 'None'}"
    )


TOOLS = [
    web_search_learning_resources,
    generate_skill_learning_plan,
    rewrite_resume_bullet,
    analyze_skill_gaps,
]


# ─────────────────────────────────────────────────────────────
# 3. LLM HELPERS
# ─────────────────────────────────────────────────────────────
def get_llm_with_tools(api_key: str):
    return ChatGroq(
        api_key=api_key,
        model=GROQ_MODEL_NAME,
        temperature=0.3,
        max_tokens=4096,
    ).bind_tools(TOOLS)


def get_llm_plain(api_key: str, temperature: float = 0.2, max_tokens: int = 4096):
    return ChatGroq(
        api_key=api_key,
        model=GROQ_MODEL_NAME,
        temperature=temperature,
        max_tokens=max_tokens,
    )


# ─────────────────────────────────────────────────────────────
# 4. AGENT NODE
# KEY FIX: system prompt now strictly limits tool calls to 1
#          and tells agent to write final answer IMMEDIATELY
#          after seeing tool results — no second tool calls
# ─────────────────────────────────────────────────────────────
def agent_node(state: AgentState, api_key: str) -> AgentState:
    llm_with_tools = get_llm_with_tools(api_key)

    # Count how many tool messages already exist in state
    # If tools have already been called — force final answer now
    tool_messages_count = sum(
        1 for m in state["messages"] if isinstance(m, ToolMessage)
    )

    if tool_messages_count >= 1:
        # Tools already ran — force final answer, no more tool calls
        system_prompt = f"""You are JobLens AI — an expert career coach.

CONTEXT:
- Target Role: {state.get('role_target', 'Not specified')}
- Experience Level: {state.get('level_target', 'Not specified')}
- Resume: {state.get('resume_context', 'Not uploaded')[:800]}
- Skill Gaps: {state.get('gap_analysis', 'Not analysed yet')}

THE TOOL HAS ALREADY RUN AND RETURNED RESULTS.
NOW YOU MUST WRITE THE FINAL ANSWER — DO NOT CALL ANY MORE TOOLS.

Using the tool results in the conversation above, write a complete,
detailed, well-structured response with:
- Clear headings and bullet points
- All URLs from tool results included as clickable links
- Minimum 200 words
- Specific actionable advice
- End with "Happy Learning! 🚀"

WRITE THE FINAL ANSWER NOW. DO NOT USE ANY TOOLS.
"""
    else:
        # No tools called yet — agent decides which tool to call
        system_prompt = f"""You are JobLens AI — an expert career coach and resume consultant.

CONTEXT ABOUT THE USER:
- Target Role: {state.get('role_target', 'Not specified')}
- Experience Level: {state.get('level_target', 'Not specified')}
- Resume Summary: {state.get('resume_context', 'Not uploaded yet')[:800]}
- Skill Gaps: {state.get('gap_analysis', 'Not analysed yet')}

AVAILABLE TOOLS — CALL EXACTLY ONE TOOL PER QUERY:
1. generate_skill_learning_plan — For ANY 30-day plan or learning roadmap request
   → This tool does EVERYTHING (web search + plan) internally
   → Do NOT call web_search after this — go straight to final answer
2. web_search_learning_resources — For general resource/YouTube recommendations
3. rewrite_resume_bullet — For rewriting a resume bullet point
4. analyze_skill_gaps — For comparing resume vs JD skills

STRICT RULES:
- Call EXACTLY ONE tool, then write the final answer
- Do NOT call multiple tools in sequence
- Do NOT call web_search after generate_skill_learning_plan
- After tool result arrives, write the complete detailed answer immediately
"""

    messages  = [SystemMessage(content=system_prompt)] + state["messages"]
    response  = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# ─────────────────────────────────────────────────────────────
# 5. TOOL EXECUTION NODE
# Heavy lifting done HERE so agent only needs one tool call
# ─────────────────────────────────────────────────────────────
def tool_executor_node(state: AgentState, api_key: str) -> AgentState:
    tool_node = ToolNode(TOOLS)
    result    = tool_node.invoke(state)

    updated_messages = result.get("messages", [])
    processed        = []

    for msg in updated_messages:
        if isinstance(msg, ToolMessage):
            content = msg.content

            # ── REWRITE BULLET ──────────────────────────────
            if content.startswith("REWRITE_REQUEST"):
                parts  = dict(p.split("=", 1) for p in content.split("|")[1:])
                bullet = parts.get("bullet", "")
                role   = parts.get("role", "Software Engineer")

                llm = get_llm_plain(api_key, temperature=0.2, max_tokens=2048)
                prompt = f"""You are an expert resume writer. Rewrite this bullet for a {role} role.

ORIGINAL: {bullet}

Rules:
1. Start with strong action verb (Built, Engineered, Developed, Optimized, Deployed)
2. Include WHAT + HOW + IMPACT/RESULT
3. Add ATS keywords for {role}
4. 1-2 lines max
5. Quantify wherever possible

Return 3 versions:

**VERSION 1** — Technical depth focus
[bullet]
*Why stronger:* [1 line]

**VERSION 2** — Quantified impact focus
[bullet]
*Why stronger:* [1 line]

**VERSION 3** — ATS keyword optimization
[bullet]
*Why stronger:* [1 line]
"""
                llm_resp = llm.invoke(prompt)
                msg = ToolMessage(content=llm_resp.content, tool_call_id=msg.tool_call_id)

            # ── 30-DAY PLAN — does web search + generation internally ──
            elif content.startswith("PLAN_REQUEST"):
                parts  = dict(p.split("=", 1) for p in content.split("|")[1:])
                skills = parts.get("skills", "LangChain")
                role   = parts.get("role", "ML Engineer")
                level  = parts.get("level", "Fresher")

                # Internal web search — so agent does NOT need a second tool call
                real_urls_text = ""
                try:
                    all_results = []
                    queries = [
                        f"{skills} tutorial YouTube beginners 2025",
                        f"free {skills} course online 2024 2025",
                        f"{skills} project for beginners GitHub",
                    ]
                    for q in queries:
                        with DDGS() as ddgs:
                            res = list(ddgs.text(q, max_results=3, region="in-en", safesearch="moderate"))
                        for r in res:
                            title = r.get("title", "")
                            url   = r.get("href", "")
                            desc  = r.get("body", "")[:120]
                            if url and url.startswith("http"):
                                all_results.append(f"• {title}\n  🔗 {url}\n  {desc}")
                    real_urls_text = "\n".join(all_results[:12]) if all_results else ""
                except Exception:
                    real_urls_text = ""

                # Generate the full plan with URLs already embedded
                llm = get_llm_plain(api_key, temperature=0.2, max_tokens=4096)
                prompt = f"""Create a complete, detailed 30-day learning plan.

TARGET: {level} aiming for {role}
SKILLS: {skills}

REAL RESOURCES FOUND (embed these URLs directly in the plan):
{real_urls_text if real_urls_text else "Use well-known resources like Krish Naik, Codebasics, Tech With Tim on YouTube."}

Write the plan in this EXACT structure — be specific and detailed:

## 🎯 30-Day {skills} Learning Roadmap for {role} ({level})

---
### 📅 WEEK 1 (Days 1–7) — Foundations
**Goal:** [specific goal for this week]
**Daily time:** 1.5–2 hours

📺 **YouTube Resources:**
- [Channel Name] — "[Exact Playlist/Video Title]"
  🔗 Link: [use URL from real resources above]
  📖 Covers: [what you will learn]
  ⭐ Why: [why this resource specifically]

🌐 **Free Courses:**
- [Platform] — "[Course Name]"
  🔗 Link: [URL]
  ⏱️ Duration: [hours]

🛠️ **Week 1 Project:**
Build: [specific mini project description]

---
### 📅 WEEK 2 (Days 8–14) — Core Skills
**Goal:** [specific week 2 goal]

📺 **YouTube Resources:**
- [resource with URL]

🛠️ **Week 2 Project:**
Build: [intermediate project]

---
### 📅 WEEK 3 (Days 15–21) — Advanced Topics
**Goal:** [specific week 3 goal]

📺 **YouTube Resources:**
- [resource with URL]

🛠️ **Week 3 Project:**
Build: [advanced component]

---
### 📅 WEEK 4 (Days 22–30) — Portfolio Project
**Goal:** Build a complete GitHub-ready project

🚀 **Capstone Project:**
Name: [project name]
Description: [what it does]
Stack: {skills} + [other tools]

Steps:
1. [step]
2. [step]
3. [step]
4. Push to GitHub with README

---
### 💡 Interview Tips for {role}

Top 3 interview questions on {skills} and how to answer:
1. Q: [question] → A: [answer strategy]
2. Q: [question] → A: [answer strategy]
3. Q: [question] → A: [answer strategy]

---
### 📊 Progress Tracker

| Week | Focus | Deliverable |
|------|-------|------------|
| Week 1 | Foundations | First working example |
| Week 2 | Core Skills | 2 mini projects |
| Week 3 | Advanced | Complex feature |
| Week 4 | Portfolio | GitHub project |

Happy Learning! 🚀
"""
                llm_resp = llm.invoke(prompt)
                msg = ToolMessage(content=llm_resp.content, tool_call_id=msg.tool_call_id)

        processed.append(msg)

    return {"messages": processed}


# ─────────────────────────────────────────────────────────────
# 6. BUILD LANGGRAPH
# ─────────────────────────────────────────────────────────────
def build_career_agent(api_key: str):
    graph = StateGraph(AgentState)

    graph.add_node("agent", lambda state: agent_node(state, api_key))
    graph.add_node("tools", lambda state: tool_executor_node(state, api_key))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", END: END}
    )
    graph.add_edge("tools", "agent")

    return graph.compile()


# ─────────────────────────────────────────────────────────────
# 7. PUBLIC INTERFACE
# ─────────────────────────────────────────────────────────────
def run_career_agent(
    api_key       : str,
    user_query    : str,
    resume_context: str  = "",
    gap_analysis  : str  = "",
    role_target   : str  = "Software Engineer",
    level_target  : str  = "Fresher",
    chat_history  : list = None,
) -> str:

    if not api_key or not api_key.strip():
        return "❌ Please enter your Groq API Key in the sidebar first."

    agent = build_career_agent(api_key)

    messages = []
    if chat_history:
        for m in chat_history:
            if m["role"] == "user":
                messages.append(HumanMessage(content=m["content"]))
            elif m["role"] == "assistant":
                messages.append(AIMessage(content=m["content"]))

    messages.append(HumanMessage(content=user_query))

    initial_state = {
        "messages"      : messages,
        "resume_context": resume_context,
        "gap_analysis"  : gap_analysis,
        "role_target"   : role_target,
        "level_target"  : level_target,
        "api_key"       : api_key,
    }

    try:
        # Increased recursion limit to 25 for safe headroom
        # Flow: START → agent(1) → tools(1) → agent(2) → END
        # That is only 3 node visits — 25 is more than enough
        final_state = agent.invoke(
            initial_state,
            config={"recursion_limit": 25}
        )
    except Exception as e:
        error_msg = str(e)
        if "recursion" in error_msg.lower():
            return (
                "⚠️ The agent took too many steps to answer this question.\n\n"
                "**Try rephrasing your question more specifically, for example:**\n"
                "- 'Give me a 30-day plan to learn LangChain' ✅\n"
                "- 'What YouTube channels teach LangGraph?' ✅\n"
                "- 'Rewrite this bullet: Worked on ML project' ✅\n\n"
                "Avoid very broad questions like 'Tell me everything about AI'."
            )
        return f"❌ Agent error: {error_msg}\n\nPlease try again."

    final_messages = final_state.get("messages", [])

    # Return the LAST AIMessage with actual text content
    # (skip tool-call AIMessages which have content="")
    for msg in reversed(final_messages):
        if isinstance(msg, AIMessage):
            if msg.content and msg.content.strip():
                return msg.content

    return "Could not generate a response. Please try again."