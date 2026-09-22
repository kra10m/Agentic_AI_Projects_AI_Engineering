import os
import json
from dotenv import load_dotenv
from openai import OpenAI

from tool_retriever import search_knowledge_base, RETRIEVAL_TOOL_SCHEMA

load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY")
)
MODEL_NAME = "openai/gpt-oss-120b"
SYSTEM_PROMPT = """You are an expert internal engineering assistant.

You have access to a tool named `search_knowledge_base` which queries internal architecture and infrastructure documentation.

Guidelines:
1. If the user asks general conversational queries (e.g., greetings, basic math), answer directly without calling tools.
2. If the user asks questions regarding internal systems, architecture, databases, or deployment workflows, ALWAYS search the knowledge base first.
3. If the first search result does not give you enough details to fully answer the query, you may call `search_knowledge_base` a second time with a revised search query.
4. Base your answers on the retrieved documentation. Do not invent details not present in the chunks.
"""

def run_agent(user_query: str, max_turns: int = 4):
    print(f"\n{'='*70}\nUSER: {user_query}\n{'='*70}")
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]

    for turn in range(max_turns):
        print(f"\n[Turn {turn + 1}] Calling LLM...")
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=[RETRIEVAL_TOOL_SCHEMA],
            tool_choice="auto"
        )
        
        response_msg = response.choices[0].message
        
        if response_msg.tool_calls:
            messages.append(response_msg)
            
            for tool_call in response_msg.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)
                call_id = tool_call.id
                
                print(f"-> Agent requested tool: {fn_name}(args={fn_args})")
                
                if fn_name == "search_knowledge_base":
                    search_query = fn_args.get("query", "")
                    tool_result = search_knowledge_base(query=search_query, top_k=2)
                else:
                    tool_result = f"Error: Unknown tool {fn_name}"
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": tool_result
                })
                print(f"-> Tool executed and result returned to context.")
                
        else:
            print(f"\n[Agent Finished]\n")
            print(f"ASSISTANT:\n{response_msg.content}")
            return response_msg.content

    print("Agent reached maximum turn limit without finishing.")
    return None

if __name__ == "__main__":
    run_agent("Hello! Who are you and what can you help me with?")

    run_agent("What database do we use for caching session data and why?")