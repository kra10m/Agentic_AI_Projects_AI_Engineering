from typing import TypedDict, Annotated, Literal, Optional, Any
from pydantic import BaseModel, Field
import operator
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class QualitativeFindings(BaseModel):
    summary: str = Field(description="Narrative overview of the company's current status.")
    sources: list[str] = Field(description="URLs or names of sources used.")
    market_sentiment: str = Field(description="Overall sentiment: Positive, Negative, or Mixed.")
    key_events: list[str] = Field(description="Recent leadership changes, product launches, or lawsuits.")

class QuantitativeFindings(BaseModel):
    revenue_trend: str = Field(description="Is revenue growing, flat, or shrinking?")
    # CHANGE 1: Use Any so the LLM can generate nested dictionaries if it wants to.
    key_metrics: dict[str, Any] = Field(description="Key financial metrics like PE ratio, YoY growth, etc.")
    financial_health_summary: str = Field(description="Brief assessment of financial stability.")

class CriticFeedback(BaseModel):
    is_approved: bool = Field(description="True if findings are solid, False if rework is needed.")
    feedback_text: str = Field(description="Specific explanation of what is missing, contradictory, or weak.")
    target_agent: Optional[Literal["Researcher", "Analyst"]] = Field(
        description="If not approved, which agent needs to redo their work?"
    )

class OverallGraphState(TypedDict):
    query: str
    qualitative_data: Optional[QualitativeFindings]
    quantitative_data: Optional[QuantitativeFindings]
    critic_history: Annotated[list[CriticFeedback], operator.add]
    messages: Annotated[list[BaseMessage], add_messages]
    next_node: str
    # CHANGE 2: Add this flag so the Supervisor knows when new data is ready for review.
    revision_pending: bool