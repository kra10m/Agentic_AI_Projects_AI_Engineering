import os
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from tool_retriever import search_knowledge_base, RETRIEVAL_TOOL_SCHEMA

load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY")
)
MODEL_NAME = "openai/gpt-oss-120b"

class CitedResponse(BaseModel):
    answer: str = Field(
        description="The concise answer grounded strictly in the provided chunks."
    )
    citations: list[str] = Field(
        default_factory=list,
        description="List of Chunk IDs actually used (e.g., ['cloud_architecture.txt_chunk_3'])."
    )
    information_found: bool = Field(
        description="True if the documentation had the answer; False if info was missing."
    )

AGENT_SYSTEM_PROMPT = """You are an internal technical knowledge assistant.
You have access to `search_knowledge_base` to look up architecture guides.

Rules:
1. For general conversation or greetings, answer directly without searching.
2. For specific questions about architecture, databases, or infrastructure, search the knowledge base.
3. Stop calling tools once you have found the necessary context.
"""

SYNTHESIS_SYSTEM_PROMPT = """You are a grounded verification assistant.
Your job is to produce a structured JSON response based on the dialogue history.

Strict Grounding Rules:
- Only assert facts explicitly stated in the retrieved chunks.
- If the retrieved context does not contain enough information to answer the question, set `information_found` to False and state that the internal docs do not contain this information.
- Populate `citations` ONLY with Chunk IDs that directly support your statements.
"""

def run_grounded_agent(user_query: str, max_turns: int = 3) -> CitedResponse:
    print(f"\n{'='*70}\nUSER QUERY: {user_query}\n{'='*70}")

    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]

    for turn in range(max_turns):
        print(f"[Turn {turn + 1}] Evaluating tools...")
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=[RETRIEVAL_TOOL_SCHEMA],
            tool_choice="auto"
        )
        msg = response.choices[0].message

        if msg.tool_calls:
            messages.append(msg)
            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)
                call_id = tool_call.id

                print(f"-> Calling {fn_name}({fn_args})")
                if fn_name == "search_knowledge_base":
                    result = search_knowledge_base(fn_args.get("query", ""), top_k=2)
                else:
                    result = f"Error: unknown tool {fn_name}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": result
                })
        else:
            messages.append(msg)
            break

    print("\n[Phase 2] Synthesizing grounded structured response...")

    gathered_context = []
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "tool":
            gathered_context.append(m.get("content", ""))

    context_str = "\n\n".join(gathered_context) if gathered_context else "No documentation was retrieved."

    synthesis_messages = [
        {
            "role": "system",
            "content": (
                SYNTHESIS_SYSTEM_PROMPT
                + "\n\nYou must respond ONLY with a valid JSON object matching this schema:\n"
                + json.dumps(CitedResponse.model_json_schema(), indent=2)
            ),
        },
        {
            "role": "user",
            "content": (
                f"User Query: {user_query}\n\n"
                f"Retrieved Context:\n{context_str}\n\n"
                "Synthesize the final answer and return the JSON object."
            ),
        },
    ]

    synth_response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=synthesis_messages,
        response_format={"type": "json_object"}
    )

    raw_json_str = synth_response.choices[0].message.content
    final_result = CitedResponse.model_validate_json(raw_json_str)
    return final_result

if __name__ == "__main__":
    res_a = run_grounded_agent("What database is used for transactions and why?")
    print("\nRESULT A:")
    print(f"Answer: {res_a.answer}")
    print(f"Citations: {res_a.citations}")
    print(f"Information Found: {res_a.information_found}")

    res_b = run_grounded_agent("What is our team's AWS monthly budget limit?")
    print("\nRESULT B:")
    print(f"Answer: {res_b.answer}")
    print(f"Citations: {res_b.citations}")
    print(f"Information Found: {res_b.information_found}")