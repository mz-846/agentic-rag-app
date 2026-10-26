"""Settings, API keys and model factories.

Keys are read from environment variables (or a local `.env` file, which is git-ignored),
never hard-coded:

    GROQ_API_KEY=...      # https://console.groq.com/keys
    TAVILY_API_KEY=...    # https://app.tavily.com

Every LLM call goes through Groq's OpenAI-compatible endpoint, so CrewAI and LangChain use
the same model and the same key. To run fully locally with Ollama instead (bonus task), set
LLM_BASE_URL=http://localhost:11434/v1 and LLM_MODEL=llama3.2 (any key works).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
# CrewAI sends anonymous usage telemetry by default; switch it off unless the user opts in.
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
# CrewAI warns on every call that Groq models have no local tokenizer; harmless, so hide it.
logging.getLogger("crewai.llms.base_llm").setLevel(logging.ERROR)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# Groq's current production chat models (Oct 2026). gpt-oss-20b is fast and cheap;
# gpt-oss-120b is stronger. The Llama-3 "llama3-8b-8192" model used in the original
# starter notebook is no longer offered.
GROQ_MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "llama-3.3-70b-versatile",
               "llama-3.1-8b-instant"]


@dataclass
class Settings:
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    tavily_api_key: str = field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", GROQ_MODELS[0]))
    llm_base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", GROQ_BASE_URL))
    temperature: float = 0.1
    # Retrieval
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL",
                                                                   "BAAI/bge-small-en-v1.5"))
    # "fastembed" = real BGE embeddings (ONNX, no PyTorch needed).
    # "hash"      = offline hashing embeddings, only for tests / no-internet demos.
    embedding_backend: str = field(default_factory=lambda: os.getenv("EMBEDDING_BACKEND",
                                                                     "fastembed"))
    vector_backend: str = "faiss"          # or "chroma"
    chunk_size: int = 800
    chunk_overlap: int = 120
    k: int = 4
    data_dir: Path = ROOT / "crew_data"
    index_dir: Path = ROOT / "vectorstore"
    reports_dir: Path = ROOT / "reports"

    def require(self, *names: str):
        missing = [n for n in names if not getattr(self, n)]
        if missing:
            env = {"groq_api_key": "GROQ_API_KEY", "tavily_api_key": "TAVILY_API_KEY"}
            raise RuntimeError("Missing " + ", ".join(env.get(m, m) for m in missing)
                               + " - add it to .env or enter it in the app sidebar.")


def crew_llm(s: Settings):
    """LLM object for CrewAI agents (native OpenAI-compatible provider -> Groq)."""
    from crewai import LLM

    s.require("groq_api_key")
    return LLM(model=s.llm_model, base_url=s.llm_base_url, api_key=s.groq_api_key,
               provider="openai", temperature=s.temperature, max_tokens=2000)


def chat_model(s: Settings):
    """LangChain chat model for the plain RAG chain (same Groq endpoint)."""
    from langchain_openai import ChatOpenAI

    s.require("groq_api_key")
    return ChatOpenAI(model=s.llm_model, base_url=s.llm_base_url, api_key=s.groq_api_key,
                      temperature=s.temperature, max_tokens=1500)
