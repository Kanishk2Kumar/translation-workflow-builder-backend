from nodes.document_upload import DocumentUploadNode
from nodes.rag_tm import RAGNode
from nodes.llm_agent import LLMAgentNode
from nodes.output import OutputNode

NODE_REGISTRY: dict[str, type] = {
    "document_upload": DocumentUploadNode,
    "rag_tm":          RAGNode,
    "vector_db":       RAGNode,       # same class for demo; vector_db = read-only RAG
    "llm_agent":       LLMAgentNode,
    "translation":     LLMAgentNode,  # translation node is just an LLM agent with defaults
    "output":          OutputNode,
    "document_output": OutputNode,
}