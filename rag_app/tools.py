"""CrewAI tools: the agents' hands.

- KnowledgeBaseTool : semantic search over the FAISS index built from crew_data/
- WebSearchTool     : real-time web search with Tavily
"""
from __future__ import annotations

from typing import Any

from crewai.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from .knowledge_base import cite


class QueryInput(BaseModel):
    query: str = Field(..., description="A focused search query")


class KnowledgeBaseTool(BaseTool):
    name: str = "knowledge_base_search"
    description: str = (
        "Search the private document knowledge base (PDFs and notes in crew_data/, e.g. the "
        "Sporo Health paper on clinical summarisation and patient chart review). Returns the "
        "most relevant passages with [source p.N] citations. Use it first for anything the "
        "documents might cover.")
    args_schema: type[BaseModel] = QueryInput
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _store: Any = PrivateAttr()
    _k: int = PrivateAttr(default=4)

    def __init__(self, vectorstore, k: int = 4, **kw):
        super().__init__(**kw)
        self._store, self._k = vectorstore, k

    def _run(self, query: str) -> str:
        hits = self._store.similarity_search_with_score(query, k=self._k)
        if not hits:
            return "No relevant passages found in the knowledge base."
        return "\n\n".join(f"[{cite(d)}] (similarity {score:.2f})\n{d.page_content}"
                           for d, score in hits)


class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = (
        "Search the live web with Tavily for current or general information that the private "
        "documents do not cover (recent news, other companies, definitions). Returns titles, "
        "URLs and snippets - cite the URLs.")
    args_schema: type[BaseModel] = QueryInput
    _client: Any = PrivateAttr()
    _max_results: int = PrivateAttr(default=4)

    def __init__(self, api_key: str, max_results: int = 4, client=None, **kw):
        super().__init__(**kw)
        if client is None:
            from tavily import TavilyClient

            client = TavilyClient(api_key=api_key)
        self._client, self._max_results = client, max_results

    def _run(self, query: str) -> str:
        res = self._client.search(query=query, max_results=self._max_results,
                                  search_depth="basic")
        items = res.get("results", [])
        if not items:
            return "No web results."
        return "\n\n".join(f"[{i + 1}] {r.get('title', '')}\nURL: {r.get('url', '')}\n"
                           f"{r.get('content', '')[:700]}" for i, r in enumerate(items))
