from langchain_core.prompts import ChatPromptTemplate
# Assuming schema.py is in the same directory
from schema import QualitativeFindings, QuantitativeFindings

# ---------------------------------------------------------
# 1. THE RESEARCHER NODE
# ---------------------------------------------------------
RESEARCHER_PROMPT = """You are the Qualitative Researcher for a Corporate Due Diligence team.
Your job is to provide a narrative overview, market sentiment, and key events for the requested company.

User Query: {query}

COUNTERPART DATA (Quantitative Analyst's Draft):
{other_agent_data}
*If data is provided above, ENSURE your narrative perfectly matches their financial numbers.*

CRITIC FEEDBACK TO ADDRESS:
{feedback}

Return the data exactly as requested in the QualitativeFindings format.
"""

def researcher_node(state: dict, llm):
    print("\n[Researcher] Gathering qualitative data...")
    
    feedback = "No previous feedback. This is your first draft."
    if state.get("critic_history"):
        latest_critique = state["critic_history"][-1]
        if latest_critique.target_agent == "Researcher" and not latest_critique.is_approved:
            feedback = f"FIX THIS ISSUE: {latest_critique.feedback_text}"
            print(f"[Researcher] Addressing critic feedback...")

    # NEW: Grab the Analyst's data to ensure alignment
    other_data = "Not available yet."
    if state.get("quantitative_data"):
        other_data = state["quantitative_data"].model_dump_json(indent=2)

    prompt = ChatPromptTemplate.from_template(RESEARCHER_PROMPT)
    structured_llm = llm.with_structured_output(QualitativeFindings)
    chain = prompt | structured_llm
    
    result = chain.invoke({
        "query": state.get("query"),
        "feedback": feedback,
        "other_agent_data": other_data # Pass it to the LLM
    })
    
    return {"qualitative_data": result, "revision_pending": True}

# ---------------------------------------------------------
# 2. THE ANALYST NODE
# ---------------------------------------------------------
ANALYST_PROMPT = """You are the Quantitative Analyst for a Corporate Due Diligence team.
Your job is to provide revenue trends, key financial metrics, and a financial health summary for the requested company.

User Query: {query}

COUNTERPART DATA (Qualitative Researcher's Draft):
{other_agent_data}
*If data is provided above, ENSURE your financial numbers perfectly match their narrative assumptions.*

CRITIC FEEDBACK TO ADDRESS:
{feedback}

CRITICAL TOOL INSTRUCTIONS:
1. You MUST use the provided tool/function to output your findings. Do NOT output plain markdown text.
2. Put all narrative analysis, DCF explanations, sensitivity analysis, and your final recommendation into the `financial_health_summary` string field.
3. Put all numerical metrics, forecasts, and comparable multiples into the `key_metrics` dictionary field.
4. NEVER use Python-style underscores in numbers (e.g., use "48000", not 48_000).
"""

def analyst_node(state: dict, llm):
    print("\n[Analyst] Crunching quantitative data...")
    
    feedback = "No previous feedback. This is your first draft."
    if state.get("critic_history"):
        latest_critique = state["critic_history"][-1]
        if latest_critique.target_agent == "Analyst" and not latest_critique.is_approved:
            feedback = f"FIX THIS ISSUE: {latest_critique.feedback_text}"
            print(f"[Analyst] Addressing critic feedback...")

    # NEW: Grab the Researcher's data to ensure alignment
    other_data = "Not available yet."
    if state.get("qualitative_data"):
        other_data = state["qualitative_data"].model_dump_json(indent=2)

    prompt = ChatPromptTemplate.from_template(ANALYST_PROMPT)
    structured_llm = llm.with_structured_output(QuantitativeFindings)
    chain = prompt | structured_llm
    
    result = chain.invoke({
        "query": state.get("query"),
        "feedback": feedback,
        "other_agent_data": other_data # Pass it to the LLM
    })
    
    return {"quantitative_data": result, "revision_pending": True}