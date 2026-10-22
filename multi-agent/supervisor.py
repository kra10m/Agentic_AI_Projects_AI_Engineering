from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage

# We will assume you are using ChatOpenAI or ChatGroq from your previous projects.
# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model="gpt-4o", temperature=0)

# ---------------------------------------------------------
# 1. THE ROUTING SCHEMA
# ---------------------------------------------------------
class SupervisorDecision(BaseModel):
    reasoning: str = Field(description="Explain WHY you are choosing the next node based on the current state.")
    next_node: Literal["Researcher", "Analyst", "Critic", "Synthesizer"] = Field(
        description="The next agent to route to."
    )

# ---------------------------------------------------------
# 2. THE SUPERVISOR PROMPT
# ---------------------------------------------------------
SUPERVISOR_PROMPT = """You are the Supervisor of a Corporate Due Diligence team.
Your goal is to coordinate your team to deliver an investment report.

CURRENT STATE:
- Qualitative Data present? {has_qual}
- Quantitative Data present? {has_quant}
- Is a Critic Review Pending? {revision_pending}
- Latest Critic Feedback: {critic_feedback}

STRICT ROUTING RULES:
1. If Qualitative Data is "No", you MUST route to "Researcher".
2. If Qualitative Data is "Yes" but Quantitative Data is "No", you MUST route to "Analyst".
3. If BOTH are "Yes" AND "Is a Critic Review Pending" is "Yes", you MUST route to "Critic".
4. If BOTH are "Yes" AND "Is a Critic Review Pending" is "No":
   - If the Critic approved (Approved: True), you MUST route to "Synthesizer".
   - If the Critic rejected (Approved: False), route directly to the Critic's target_agent ("Researcher" or "Analyst").

Never route to an agent whose data is already present unless the Critic explicitly rejected their work.
"""
# ---------------------------------------------------------
# 3. THE SUPERVISOR NODE FUNCTION
# ---------------------------------------------------------
def supervisor_node(state: dict, llm):
    has_qual = "Yes" if state.get("qualitative_data") else "No"
    has_quant = "Yes" if state.get("quantitative_data") else "No"
    
    # Check our new state flag
    rev_pending = "Yes" if state.get("revision_pending") else "No"
    
    recent_feedback = "None"
    if state.get("critic_history") and len(state["critic_history"]) > 0:
        latest_critique = state["critic_history"][-1]
        recent_feedback = f"Approved: {latest_critique.is_approved}. Target: {latest_critique.target_agent}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", SUPERVISOR_PROMPT),
        ("human", "User Query: {query}")
    ])
    
    router_llm = llm.with_structured_output(SupervisorDecision)
    chain = prompt | router_llm
    
    decision: SupervisorDecision = chain.invoke({
        "has_qual": has_qual,
        "has_quant": has_quant,
        "critic_feedback": recent_feedback,
        "revision_pending": rev_pending, # Inject the flag into the prompt
        "query": state.get("query")
    })
    
    print(f"[Supervisor] Routing to: {decision.next_node} | Reason: {decision.reasoning}")
    return {"next_node": decision.next_node}