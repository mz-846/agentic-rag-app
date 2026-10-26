"""Agentic RAG app: document RAG + Tavily web search + CrewAI agents on Groq."""
import warnings

# langchain-community prints a sunset notice on import; FAISS + FastEmbed still live there.
warnings.filterwarnings("ignore", message=".*langchain-community.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_community")
