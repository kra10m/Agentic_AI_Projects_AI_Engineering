# LangGraph Core Concepts

LangGraph is a library for building stateful, multi-actor applications with LLMs, built on top of LangChain. It represents applications as graphs.

## StateGraph
The core abstraction in LangGraph is the `StateGraph`. A StateGraph is parameterized by a state schema. The state is passed to each node in the graph, and nodes return updates to this state.

## Nodes
Nodes are the building blocks of a LangGraph. They can be Python functions or runnables. Every node receives the current state as input and returns a dictionary representing updates to the state.
To add a node to the graph, use `graph.add_node(name, function)`. The name must be a unique string.

## Edges
Edges define the flow of execution between nodes.
- **Normal Edges:** Connect one node to another directly using `graph.add_edge(start_node, end_node)`.
- **Conditional Edges:** Route execution dynamically based on the current state. Use `graph.add_conditional_edges(start_node, routing_function, mapping)`. The routing function returns a string, which the mapping dictionary uses to determine the next node.
- **Entry Point:** Defined via `graph.set_entry_point(node_name)`.
- **Finish Point:** The graph execution stops when it reaches the special `END` node.