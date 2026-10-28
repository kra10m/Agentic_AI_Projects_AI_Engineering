import os
from typing import List, TypedDict
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph, START

from query_transform import QueryTransformer
from retrieval_rerank import AdvancedRetriever

load_dotenv()

class GraphState(TypedDict):
    original_query: str
    current_query: str
    documents: List[str]
    retrieved_chunk_ids: List[str]
    generation: str
    attempt_count: int
    max_attempts: int
    evidence_sufficient: bool

class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

class GroundedAnswer(BaseModel):
    """Schema for the final generated response to enforce citations."""
    answer: str = Field(description="The detailed answer to the user's question.")
    citations: List[str] = Field(description="A list of the exact Chunk IDs used to answer the question.")

class AdaptiveRAGGraph:
    def __init__(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        chroma_db_dir = os.path.join(os.path.dirname(script_dir), "chroma_db_advanced")
        
        self.retriever = AdvancedRetriever(chroma_db_dir)
        self.query_transformer = QueryTransformer()
        
        self.llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0) 
        self.structured_evaluator = self.llm.with_structured_output(GradeDocuments)
        self.structured_generator = self.llm.with_structured_output(GroundedAnswer)
        
        self.app = self._build_graph()


    def retrieve_node(self, state: GraphState):
        attempt = state.get("attempt_count", 0) + 1
        print(f"\n--- RETRIEVE (Attempt {attempt}) ---")
        query = state["current_query"]
        
        retrieved_ids = self.retriever.retrieve(query, initial_k=10, final_k=3)
        
        documents = []
        for doc_id in retrieved_ids:
            raw_text = self.retriever.doc_map[doc_id]
            documents.append(f"Chunk ID: {doc_id}\nContent: {raw_text}")
        
        return {
            "documents": documents, 
            "retrieved_chunk_ids": retrieved_ids, 
            "attempt_count": attempt
        }

    def evaluate_node(self, state: GraphState):
        print("\n--- EVALUATE EVIDENCE ---")
        query = state["current_query"]
        documents = state["documents"]
        
        system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
        If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
        
        grade_prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}")
        ])
        
        evaluator = grade_prompt | self.structured_evaluator
        
        relevant_docs_found = False
        for doc in documents:
            score = evaluator.invoke({"question": query, "document": doc})
            if score.binary_score.lower() == "yes":
                relevant_docs_found = True
                break 
                
        if relevant_docs_found:
            print("✅ Evidence Sufficient.")
        else:
            print("❌ Evidence Insufficient.")
            
        return {"evidence_sufficient": relevant_docs_found}

    def transform_query_node(self, state: GraphState):
        print("\n--- TRANSFORM QUERY ---")
        new_query = self.query_transformer.transform(state["current_query"])
        return {"current_query": new_query}
        
    def generate_node(self, state: GraphState):
        print("\n--- GENERATE ---")
        
        if not state.get("evidence_sufficient", False) and state["attempt_count"] >= state["max_attempts"]:
            print("Executing clean fallback (Missing Information).")
            fallback_msg = "I could not find sufficient information in the provided technical documentation to answer this question securely."
            return {"generation": fallback_msg}

        system_prompt = """You are a technical documentation assistant. Answer the user's question using ONLY the provided retrieved context.
        If you don't know the answer, just say that you don't know. 
        You must cite the exact Chunk IDs you used to form your answer."""
        
        gen_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Retrieved Context: \n\n {context} \n\n User Question: {question}")
        ])
        
        generator = gen_prompt | self.structured_generator
        

        context_str = "\n\n".join(state["documents"])
        response = generator.invoke({"context": context_str, "question": state["original_query"]})
        
        valid_citations = [cite for cite in response.citations if cite in state["retrieved_chunk_ids"]]
        
        final_output = f"{response.answer}\n\nSources: {', '.join(valid_citations) if valid_citations else 'None'}"
        
        return {"generation": final_output}


    def route_after_evaluation(self, state: GraphState):
        print("\n--- ROUTING ---")
        if state.get("evidence_sufficient", False):
            print("-> Evidence good. Routing to Generation.")
            return "generate"
        elif state["attempt_count"] >= state["max_attempts"]:
            print("-> Max attempts reached. Routing to Generation (Best Effort).")
            return "generate"
        else:
            print("-> Evidence poor. Routing to Query Transformation for retry.")
            return "transform_query"

    def _build_graph(self):
        workflow = StateGraph(GraphState)
        
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("evaluate", self.evaluate_node)
        workflow.add_node("transform_query", self.transform_query_node)
        workflow.add_node("generate", self.generate_node)
        
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "evaluate")
        workflow.add_edge("transform_query", "retrieve")
        workflow.add_edge("generate", END)
        
        workflow.add_conditional_edges(
            "evaluate",
            self.route_after_evaluation,
            {
                "generate": "generate",
                "transform_query": "transform_query"
            }
        )
        
        return workflow.compile()

if __name__ == "__main__":
    print("Testing Adaptive RAG Pipeline...")
    rag = AdaptiveRAGGraph()
    
    initial_state = {
        "original_query": "how do I save my state to a sql database?",
        "current_query": "how do I save my state to a sql database?",
        "attempt_count": 0,
        "max_attempts": 2
    }
    
    final_state = rag.app.invoke(initial_state)
    print("\n--- FINAL STATE ---")
    print(f"Original Query: {final_state['original_query']}")
    print(f"Final Executed Query: {final_state['current_query']}")
    print(f"\nFinal Grounded Output:\n{final_state['generation']}")