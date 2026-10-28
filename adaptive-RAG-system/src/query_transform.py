import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()


class RewrittenQuery(BaseModel):
    query: str = Field(description="The rewritten, highly specific search query")

class QueryTransformer:
    def __init__(self):
        self.llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
        self.structured_llm = self.llm.with_structured_output(RewrittenQuery)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert AI assistant that optimizes user queries for a vector search engine over LangGraph documentation. "
                       "Resolve any ambiguous pronouns, add missing technical context, and make the query as precise as possible. "
                       "Do not attempt to answer the question. Your only job is to output the optimized query."),
            ("human", "User query: {query}")
        ])
        
        self.chain = self.prompt | self.structured_llm

    def transform(self, query: str) -> str:
        print(f"Original query: {query}")
        result = self.chain.invoke({"query": query})
        print(f"Rewritten query: {result.query}")
        return result.query

if __name__ == "__main__":
    transformer = QueryTransformer()
    print("\n--- Testing Query Transformation ---")
    
    transformer.transform("how do I pause it and resume later?")
    print("-" * 40)
    
    transformer.transform("What reducer should I use to append messages?")