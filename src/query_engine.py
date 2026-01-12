import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_ollama import ChatOllama


class QueryEngine:
    """Query engine for RAG-based document retrieval."""
    
    def __init__(self, persist_directory: str = "../chroma_db"):
        """Initialize the query engine with embeddings and vector store."""
        self.embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
        self.vector_store = Chroma(embedding_function=self.embeddings, persist_directory=persist_directory)
        self.query_engine = self.vector_store.as_retriever(search_kwargs={"k": 4})
        print("Retriever is ready. Connected to ChromaDB.")
        
    def _create_retrieve_tool(self):
        """Create the retrieve_context tool with access to vector_store."""
        vector_store = self.vector_store  # Capture in closure
        
        @tool(response_format="content_and_artifact")
        def retrieve_context(query: str):
            """Retrieve information to help answer a query."""
            retrieved_docs = vector_store.similarity_search(query, k=3)
            serialized = "\n\n".join(
                (f"Source: {doc.metadata}\nContent: {doc.page_content}")
                for doc in retrieved_docs
            )
            return serialized, retrieved_docs
        
        return retrieve_context
    
    def create_agent(self, model: str = "llama3.1:8b"):
        """Create a RAG agent with the retrieval tool."""
        llm = ChatOllama(model=model)
        retrieve_tool = self._create_retrieve_tool()
        
        prompt = (
            "You have access to a tool that retrieves context from a school readings and lectures. "
            "Use the tool to help answer user queries."
        )
        agent = create_agent(model=llm, tools=[retrieve_tool], system_prompt=prompt)
        return agent
    
    def query(self, agent, query: str):
        """Query the agent and return the response."""
        response = agent.invoke({"messages": [{"role": "user", "content": query}]})
        messages = response["messages"]
        
        # Find the last AI/assistant message (skip tool messages)
        # LangChain agents return messages in order: Human, AI (with tool calls), Tool, AI (final response)
        for message in reversed(messages):
            # Check the class name or type to identify AI messages
            msg_class = type(message).__name__
            msg_type = getattr(message, 'type', None) or (message.get("type") if isinstance(message, dict) else None)
            
            # Check if it's an AI message (not a tool message)
            if msg_class == "AIMessage" or msg_type == "ai":
                content = getattr(message, 'content', None) or (message.get("content") if isinstance(message, dict) else None)
                # Only return if it has actual content (not just tool calls)
                if content and content.strip():
                    return content
        
        # Fallback: try to get content from last message
        last_msg = messages[-1] if messages else None
        if last_msg:
            content = getattr(last_msg, 'content', None) or (last_msg.get("content") if isinstance(last_msg, dict) else None)
            if content:
                return content
        
        return "No response generated."


if __name__ == "__main__":
    # Initialize query engine
    qe = QueryEngine()
    
    # Create agent
    agent = qe.create_agent()
    
    # Example query
    query = (
        "What is the standard method for performing Monte Carlo dropouts?\n\n"
        "Once you get the answer, look up common extensions of that method."
    )
    
    # Get response
    response = qe.query(agent, query)
    print(response)

