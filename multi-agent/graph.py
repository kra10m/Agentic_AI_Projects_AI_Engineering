from functools import partial
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate

from langchain_groq import ChatGroq

from schema import OverallGraphState
from supervisor import supervisor_node
from specialists import researcher_node, analyst_node
from critic import critic_node

import os
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, ".env")
load_dotenv(dotenv_path=env_path)

# Instantiate your LLM
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

# ---------------------------------------------------------
# 1. THE FINAL SYNTHESIZER
# ---------------------------------------------------------
SYNTHESIZER_PROMPT = """You are the Final Synthesizer for a Corporate Due Diligence team.
The Critic has approved the following research and financial data. 
Write a crisp, professional Investment Memo for the user.

USER QUERY: {query}

--- QUALITATIVE DATA ---
{qual_data}

--- QUANTITATIVE DATA ---
{quant_data}

Do not ask for more information. Output the final markdown report.
"""

def synthesizer_node(state: dict, llm):
    print("\n[Synthesizer] Writing final investment memo...")
    
    qual_str = state["qualitative_data"].model_dump_json(indent=2)
    quant_str = state["quantitative_data"].model_dump_json(indent=2)
    
    prompt = ChatPromptTemplate.from_template(SYNTHESIZER_PROMPT)
    chain = prompt | llm
    
    result = chain.invoke({
        "query": state["query"],
        "qual_data": qual_str,
        "quant_data": quant_str
    })
    
    # We aren't saving this to structured state, just printing/returning the final text
    return {"messages": [result]}

# ---------------------------------------------------------
# 2. THE ROUTER FUNCTION
# ---------------------------------------------------------
def route_from_supervisor(state: dict):
    """
    LangGraph conditional edges need a function that returns the name of the next node.
    Our Supervisor already did the hard work and saved the decision to state["next_node"].
    """
    return state.get("next_node")

# ---------------------------------------------------------
# 3. GRAPH ASSEMBLY (THE STAR ARCHITECTURE)
# ---------------------------------------------------------

# 1. Initialize the graph with our state schema
builder = StateGraph(OverallGraphState)

# 2. Add nodes (using functools.partial to pass the LLM into our node functions)
builder.add_node("Supervisor", partial(supervisor_node, llm=llm))
builder.add_node("Researcher", partial(researcher_node, llm=llm))
builder.add_node("Analyst", partial(analyst_node, llm=llm))
builder.add_node("Critic", partial(critic_node, llm=llm))
builder.add_node("Synthesizer", partial(synthesizer_node, llm=llm))

# 3. Define the Entry Point
builder.set_entry_point("Supervisor")

# 4. Define the edges (Workers ALWAYS report back to the Supervisor)
builder.add_edge("Researcher", "Supervisor")
builder.add_edge("Analyst", "Supervisor")
builder.add_edge("Critic", "Supervisor")
builder.add_edge("Synthesizer", END) # The process stops after synthesis

# 5. Define the Conditional Routing from the Supervisor
builder.add_conditional_edges(
    "Supervisor",
    route_from_supervisor,
    {
        # If route_from_supervisor returns "Researcher", go to the "Researcher" node
        "Researcher": "Researcher",
        "Analyst": "Analyst",
        "Critic": "Critic",
        "Synthesizer": "Synthesizer"
    }
)

# 6. Compile the graph with memory
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

if __name__ == "__main__":
    initial_state = {
        "query": "Should we invest in Tesla right now? Give me a full due diligence report.",
        "qualitative_data": None,
        "quantitative_data": None,
        "critic_history": [],
        "messages": [],
        "revision_pending": False,  # <--- Added
    }
    
    # We must pass a config with a thread_id so the memory saver knows where to store this specific run
    config = {"configurable": {"thread_id": "1"}}
    
    print("Starting Multi-Agent System...\n")
    
    # Pass the config into the stream
    for output in graph.stream(initial_state, config, stream_mode="updates"):
        for node_name, state_update in output.items():
            print(f"--- Finished Node: {node_name} ---")
            
    print("\n--- FINAL RESULT ---")
    # Now this will successfully retrieve the final state from memory!
    final_state = graph.get_state(config).values
    print(final_state.get("messages")[-1].content)