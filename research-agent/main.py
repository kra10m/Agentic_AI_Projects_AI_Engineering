from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from agent import ResearchAgent
from schemas import ResearchReport

app = FastAPI(
    title="Autonomous Research Agent API",
    description="Production API delivering citation-backed research reports via an autonomous ReAct loop.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema
class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=5, max_length=500, description="The research topic or question.")
    max_iterations: int = Field(default=3, ge=1, le=5, description="Search exploration depth cap.")

# Dependency / Singleton
agent_instance = None

def get_agent(max_iterations: int = 3) -> ResearchAgent:
    return ResearchAgent(max_iterations=max_iterations)

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    return {"status": "healthy", "service": "research-agent"}

@app.post("/research", response_model=ResearchReport, status_code=status.HTTP_200_OK)
async def conduct_research(request: ResearchRequest):
    """
    Executes an autonomous research investigation and returns a structured Pydantic report.
    """
    try:
        agent = get_agent(max_iterations=request.max_iterations)
        # Offload blocking synchronous LLM & search calls to a worker threadpool
        report = await run_in_threadpool(agent.run, request.query)
        return report
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent execution failed: {str(exc)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)