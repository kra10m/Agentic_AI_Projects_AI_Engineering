from typing import List
from pydantic import BaseModel, Field

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the live web for recent news, technical documentation, or factual queries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The exact search query string to submit to the search engine."
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of search results to return (default is 3).",
                        "default": 3
                    }
                },
                "required": ["query"]
            }
        }
    }
]

class Citation(BaseModel):
    title: str = Field(description="Title of the source or article")
    url: str = Field(description="Direct URL to the referenced material")

class ResearchReport(BaseModel):
    title: str = Field(description="Clear, descriptive title for the research report")
    executive_summary: str = Field(description="A high-level 2-3 sentence overview answering the user's core query")
    key_findings: List[str] = Field(description="List of 3 to 5 major discoveries, updates, or technical takeaways")
    detailed_analysis: str = Field(description="Detailed narrative analysis synthesized across all gathered sources")
    sources: List[Citation] = Field(description="List of all unique web sources cited during research")