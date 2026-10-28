# Multi-Agent Architectures in LangGraph

LangGraph excels at orchestrating multiple specialized agents. 

## The Supervisor Pattern
In a supervisor architecture, a single routing agent (the supervisor) delegates tasks to worker agents. The state schema typically includes a list of `messages` and a `next` field indicating which worker should act next.

## State Reducers
When multiple agents update the state, you must define how updates are merged. In LangGraph state schemas (often defined via `TypedDict`), you can use reducers. 
For example, to append messages rather than overwrite them, use the `add_messages` reducer from `langgraph.graph.message`:
`messages: Annotated[list, add_messages]`

## Routing Logic
The supervisor LLM is prompted to respond with the name of the next worker or "FINISH". This output is parsed and used by a conditional edge to route execution to the corresponding worker node. Once a worker finishes, it typically routes back to the supervisor via a standard edge.