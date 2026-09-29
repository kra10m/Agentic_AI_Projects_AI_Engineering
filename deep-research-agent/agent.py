"""
Deep Research & Evidence Verification Agent
Built with LangGraph, LangChain, and DuckDuckGo Search.

This agentic workflow decomposes a complex research topic into sub-questions,
iteratively researches each question, extracts verifiable claims, checks for 
contradictions, and synthesizes a final cited report.
"""

import os
import operator
from typing import Annotated, TypedDict
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from ddgs import DDGS

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

# Load environment variables
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, ".env")
load_dotenv(dotenv_path=env_path)

# ==========================================
# Schemas
# ==========================================

class ResearchPlan(BaseModel):
    """Schema for decomposing a research topic."""
    sub_questions: list[str] = Field(min_length=2, max_length=4)

class Evidence(BaseModel):
    """Schema for an extracted factual claim."""
    claim: str
    source_url: str
    relevant_question: str

class EvidenceExtraction(BaseModel):
    """Wrapper schema for multiple evidence items."""
    items: list[Evidence]

class Contradiction(BaseModel):
    """Schema for detected conflicts in evidence."""
    description: str = Field(description="Exact conflict between sources.")
    investigate_question: str = Field(description="Specific search question to resolve the conflict.")

class VerificationResult(BaseModel):
    """Schema for the verification phase output."""
    contradictions_found: bool
    contradictions: list[Contradiction] = Field(default_factory=list)


# ==========================================
# State Definition
# ==========================================

class ResearchState(TypedDict):
    research_topic: str
    sub_questions: Annotated[list[str], operator.add] 
    answered_questions: Annotated[list[str], operator.add]
    evidence: Annotated[list[Evidence], operator.add]
    contradictions: Annotated[list[str], operator.add]
    verification_done: bool 
    final_report: str


# ==========================================
# LLM Initialization
# ==========================================

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

structured_planner = llm.with_structured_output(ResearchPlan)
structured_evaluator = llm.with_structured_output(EvidenceExtraction)
structured_verifier = llm.with_structured_output(VerificationResult)


# ==========================================
# Graph Nodes
# ==========================================

def planner_node(state: ResearchState):
    """Decomposes the main topic into actionable research tasks."""
    print(f"\n--- [NODE: PLANNER] Decomposing Topic ---")
    
    prompt = (
        f"You are a lead researcher. Decompose the following topic into specific, "
        f"verifiable sub-questions.\n"
        f"IMPORTANT: You MUST use the provided tool/schema to output the list. "
        f"Do not output markdown, bullet points, or conversational text. Return ONLY the structured data.\n\n"
        f"Topic: {state['research_topic']}"
    )
    
    plan: ResearchPlan = structured_planner.invoke(prompt)
    print(f"Generated {len(plan.sub_questions)} tasks.")
    
    return {"sub_questions": plan.sub_questions, "verification_done": False}


def researcher_node(state: ResearchState):
    """Executes web searches and extracts structured evidence for pending tasks."""
    all_q = state.get("sub_questions", [])
    answered_q = state.get("answered_questions", [])
    
    pending_q = [q for q in all_q if q not in answered_q]
    if not pending_q:
        return {} 
        
    current_q = pending_q[0]
    print(f"\n--- [NODE: RESEARCHER] Investigating: {current_q[:60]}... ---")
    
    # Query formulation
    topic = state['research_topic'].replace("What are the primary reasons why the", "").replace("fell?", "fall").strip()
    stop_words = {"what", "are", "the", "to", "did", "how", "why", "which", "and", "in", "of", "such", "as", "extent"}
    q_words = [w for w in current_q.replace("?", "").replace(",", "").split() if w.lower() not in stop_words]
    
    clean_query = f"{topic} {' '.join(q_words[:4])}"
    print(f"   -> Querying web for: '{clean_query}'...")
    
    # Web search execution
    raw_results = []
    try:
        raw_results = list(DDGS().text(clean_query, max_results=3))
    except Exception as e:
        print(f"   -> Search engine warning: {e}")

    if not raw_results:
        print("   -> No search results found. Skipping extraction.")
        return {"answered_questions": [current_q], "evidence": []}

    search_context = "\n\n".join([f"Source: {r.get('href')}\nSnippet: {r.get('body')}" for r in raw_results if r.get('body')])
        
    # Evidence extraction
    prompt = (
        f"Extract 1 to 2 key factual claims from the search results that help answer this question.\n"
        f"IMPORTANT: If the search results do not contain relevant information, return an empty list.\n\n"
        f"Question: {current_q}\n\nSearch Results:\n{search_context}"
    )
    
    try:
        extraction: EvidenceExtraction = structured_evaluator.invoke(prompt)
        items = [ev for ev in (extraction.items or []) if ev.claim.strip()]
        for ev in items:
            ev.relevant_question = current_q
    except Exception:
        items = []
        
    print(f"   -> Extracted {len(items)} validated claim(s).")
    return {"answered_questions": [current_q], "evidence": items}


def verifier_node(state: ResearchState):
    """Evaluates collected evidence for contradictions and generates resolution tasks."""
    print(f"\n--- [NODE: VERIFIER] Analyzing {len(state['evidence'])} claims for contradictions ---")
    
    evidence_text = "".join(
        [f"[{i}] {ev.claim} (Source: {ev.source_url})\n" for i, ev in enumerate(state['evidence'], 1)]
    )
        
    prompt = (
        f"You are a critical verification agent. Review the following claims for direct contradictions "
        f"or significant factual tensions.\n\n"
        f"Claims:\n{evidence_text}\n\n"
        f"If you find a contradiction, describe it and formulate exactly one new, highly targeted "
        f"sub-question to resolve it.\n"
        f"IMPORTANT: You MUST use the provided tool/schema to output your result. "
        f"Do not output markdown, essays, or conversational text. Return ONLY the structured data."
    )
    
    try:
        result: VerificationResult = structured_verifier.invoke(prompt)
    except Exception as e:
        print(f"   -> Verifier parsing warning: {e}. Assuming no contradictions.")
        result = VerificationResult(contradictions_found=False, contradictions=[])
    
    if result.contradictions_found and result.contradictions:
        print(f"   -> WARNING: Found {len(result.contradictions)} contradiction(s)!")
        new_questions = []
        conflict_descriptions = []
        
        for c in result.contradictions:
            print(f"      Conflict: {c.description}")
            print(f"      New Task: {c.investigate_question}")
            new_questions.append(c.investigate_question)
            conflict_descriptions.append(c.description)
            
        return {
            "sub_questions": new_questions, 
            "contradictions": conflict_descriptions,
            "verification_done": True 
        }
    
    print("   -> All claims align. No contradictions detected.")
    return {"verification_done": True}


def synthesizer_node(state: ResearchState):
    """Generates the final markdown report with inline citations."""
    print(f"\n--- [NODE: SYNTHESIZER] Drafting Final Report ---")
    
    evidence_text = "".join(
        [f"[{i}] {ev.claim}\n    Source: {ev.source_url}\n" for i, ev in enumerate(state['evidence'], 1)]
    )
        
    contradictions_text = "None detected."
    if state.get("contradictions"):
        contradictions_text = "\n".join(state["contradictions"])
        
    prompt = (
        f"You are an expert analyst. Write a comprehensive, highly professional final report on the topic: '{state['research_topic']}'.\n\n"
        f"You must base your report EXCLUSIVELY on the following verified evidence. "
        f"Do not include outside knowledge.\n\n"
        f"Evidence:\n{evidence_text}\n\n"
        f"Noted Contradictions/Tensions to address:\n{contradictions_text}\n\n"
        f"IMPORTANT formatting rules:\n"
        f"1. Use clear Markdown headings.\n"
        f"2. Every time you state a fact, you MUST include an inline citation to the source URL (e.g. [1], [2]).\n"
        f"3. Include a 'References' section at the bottom mapping the numbers to the URLs."
    )
    
    response = llm.invoke(prompt)
    return {"final_report": response.content}


# ==========================================
# Routing Logic
# ==========================================

def route_after_research(state: ResearchState):
    """Determines whether to continue research or move to verification."""
    all_q = state.get("sub_questions", [])
    answered_q = state.get("answered_questions", [])
    
    if len(answered_q) < len(all_q):
        print("   [ROUTER] -> Tasks remaining. Looping back...")
        return "researcher"
    
    if not state.get("verification_done", False):
        print("   [ROUTER] -> Research complete. Routing to Verifier...")
        return "verifier"
    
    return END

def route_after_verification(state: ResearchState):
    """Routes back to research if contradictions were found, otherwise synthesizes."""
    all_q = state.get("sub_questions", [])
    answered_q = state.get("answered_questions", [])
    
    if len(answered_q) < len(all_q):
        print("   [ROUTER] -> Verifier spawned new tasks. Looping back to Researcher...")
        return "researcher"
    
    print("   [ROUTER] -> Verification complete. Routing to Synthesizer...")
    return "synthesizer" 


# ==========================================
# Graph Compilation
# ==========================================

workflow = StateGraph(ResearchState)

workflow.add_node("planner", planner_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("verifier", verifier_node)
workflow.add_node("synthesizer", synthesizer_node)

workflow.add_edge(START, "planner")
workflow.add_edge("planner", "researcher")

workflow.add_conditional_edges("researcher", route_after_research)
workflow.add_conditional_edges("verifier", route_after_verification)
workflow.add_edge("synthesizer", END) 

app = workflow.compile()


if __name__ == "__main__":
    initial_state = {
        "research_topic": "Did Vikings have horns on their helmets?",
        "sub_questions": [],
        "answered_questions": [],
        "evidence": [],
        "contradictions": [],
        "verification_done": False
    }
    
    final_state = app.invoke(initial_state)
    
    print("\n" + "="*50)
    print(" FINAL RESEARCH REPORT ")
    print("="*50 + "\n")
    print(final_state.get("final_report", "Report generation failed."))