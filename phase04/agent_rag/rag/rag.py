from typing import Any, Dict, Optional
from .retriever import BM25Retriever
from openai import AsyncOpenAI


class RAG:
    """RAG system that can operate in naive or agentic mode."""
    def __init__(self, llm_client: AsyncOpenAI, retriever: BM25Retriever, mode="naive", system_prompt=None, model="gpt-4o-mini", default_k=3):
        self.llm_client = llm_client
        self.retriever = retriever
        self.mode = mode.lower()
        self.model = model
        self.default_k = default_k
        self.system_prompt = system_prompt or "Answer only based on documents. Be concise.\n\nQuestion: {query}\nDocuments:\n{context}\nAnswer:"
        self._agent = None
        
        if self.mode == "agentic":
            self._setup_agent()

    def _setup_agent(self):
        """Setup agent for agentic mode."""
        try:
            from agents import Agent, function_tool
        except ImportError:
            raise ImportError("agents package required for agentic mode")

        @function_tool
        def retrieve(query: str) -> str:
            """Search Hugging Face docs for technical info, APIs, commands, and examples.
            Use exact terms (e.g., "from_pretrained", "ESPnet upload", "torchrun"). 
            Try 2-3 targeted searches: specific terms → tool names → alternatives."""
            docs = self.retriever.retrieve(query, self.default_k)
            if not docs:
                return f"No documents found for '{query}'. Try different search terms or break down the query into smaller parts."
            return "\n\n".join([f"Doc {i}: {doc.page_content}" for i, doc in enumerate(docs, 1)])

        self._agent = Agent(
            name="RAG Assistant",
            model=self.model,
            instructions="Search with exact terms first (commands, APIs, tool names). Try 2-3 different searches if needed. Only answer from retrieved documents. Preserve exact syntax and technical details.",
            tools=[retrieve]
        )

    async def _naive_query(self, question: str, top_k: int) -> Dict[str, Any]:
        """Handle naive mode: retrieve once, then generate."""
        # Retrieve documents
        docs = self.retriever.retrieve(question, top_k)
        
        if not docs:
            return {"answer": "No relevant documents found.", "retrieved_documents": [], "num_retrieved": 0}
        
        # Generate response
        context = "\n\n".join([f"Document {i}:\n{doc.page_content}" for i, doc in enumerate(docs, 1)])
        prompt = self.system_prompt.format(query=question, context=context)
        
        response = await self.llm_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )

        return {
            "answer": response.choices[0].message.content.strip(),
            "retrieved_documents": [{"content": doc.page_content, "metadata": doc.metadata, "document_id": i} for i, doc in enumerate(docs)],
            "num_retrieved": len(docs),
        }

    async def _agentic_query(self, question: str, top_k: int) -> Dict[str, Any]:
        """Handle agentic mode: agent controls retrieval strategy."""
        try:
            from agents import Runner
        except ImportError:
            raise ImportError("agents package required for agentic mode")

        # Let agent handle the retrieval and reasoning
        result = await Runner.run(self._agent, input=question)

        # In agentic mode, the agent controls retrieval internally
        # so we don't return specific retrieved documents
        return {
            "answer": result.final_output,
            "retrieved_documents": [],  # Agent handles retrieval internally
            "num_retrieved": 0,  # Cannot determine exact count from agent execution
        }

    async def query(self, question: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """Query the RAG system."""
        if top_k is None:
            top_k = self.default_k

        try:
            if self.mode == "naive":
                return await self._naive_query(question, top_k)
            elif self.mode == "agentic":
                return await self._agentic_query(question, top_k)
            else:
                raise ValueError(f"Unknown mode: {self.mode}")
        except Exception as e:
            return {
                "answer": f"Error: {str(e)}",
                "retrieved_documents": [],
                "num_retrieved": 0,
            }