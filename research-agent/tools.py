import json
import os
from typing import Any, Dict, List
from tavily import TavilyClient

def web_search(query: str, max_results: int = 3) -> str:
    """
    Executes a web search and truncates snippets to fit within strict token budgets.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return json.dumps({"error": "TAVILY_API_KEY environment variable is missing."})

    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="basic",
            include_answer=False,
        )

        results: List[Dict[str, Any]] = []
        for item in response.get("results", []):
            snippet = item.get("content", "")
            # Truncate each snippet to ~350 characters to guard token limits
            if len(snippet) > 350:
                snippet = snippet[:350] + "..."
            
            results.append({
                "title": item.get("title"),
                "url": item.get("url"),
                "snippet": snippet,
            })

        if not results:
            return json.dumps({"message": f"No results found for query: '{query}'"})

        return json.dumps(results)

    except Exception as exc:
        return json.dumps({"error": f"Search execution failed: {str(exc)}"})