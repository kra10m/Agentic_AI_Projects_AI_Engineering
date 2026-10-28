# LangGraph Checkpointing and Persistence

Stateful applications require memory. LangGraph handles this via Checkpointers, which save the state of the graph at every superstep.

## MemorySaver
For local or testing environments, LangGraph provides `MemorySaver`, an in-memory checkpointer. 
To use it, initialize `memory = MemorySaver()` and pass it when compiling the graph: `app = graph.compile(checkpointer=memory)`.

## Threads
Checkpointers organize state by threads. A thread represents a single continuous conversation or execution trace.
To execute a compiled graph within a specific thread, you must pass a `thread_id` in the configuration dictionary:
`config = {"configurable": {"thread_id": "thread_123"}}`
`app.invoke(initial_state, config=config)`

## Resuming and Human-in-the-Loop
Because state is persisted per thread, you can pause execution, ask for human input, and resume exactly where the graph left off. If a node is interrupted, invoking the graph again with the same `thread_id` will resume execution from the interrupted node using the persisted state.