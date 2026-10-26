"""Offline smoke tests - no API keys, no internet.

They use the HashingEmbeddings stand-in, a fake Tavily client, and a tiny fake
OpenAI-compatible server (tests/fake_llm_server.py) in place of Groq, so they check the
*plumbing* (loading, chunking, indexing, retrieval, tool calls, agent hand-offs, report
export) - not answer quality.

    pip install pytest flask
    pytest -q
"""
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag_app.config import Settings, chat_model, crew_llm  # noqa: E402
from rag_app import knowledge_base as kb  # noqa: E402


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close()
    return p


@pytest.fixture(scope="session")
def fake_llm(tmp_path_factory):
    port, log = free_port(), tmp_path_factory.mktemp("llm") / "requests.jsonl"
    proc = subprocess.Popen([sys.executable, str(ROOT / "tests" / "fake_llm_server.py"),
                             str(port), str(log)])
    for _ in range(50):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close(); break
        except OSError:
            time.sleep(0.2)
    yield f"http://127.0.0.1:{port}/v1", log
    proc.terminate()


@pytest.fixture(scope="session")
def settings(tmp_path_factory, fake_llm):
    s = Settings(groq_api_key="test", tavily_api_key="test", embedding_backend="hash")
    s.llm_base_url = fake_llm[0]
    s.index_dir = tmp_path_factory.mktemp("index")
    return s


@pytest.fixture(scope="session")
def store(settings):
    return kb.get_vectorstore(settings)


class FakeTavily:
    def search(self, query, **kw):
        return {"results": [{"title": "Example news", "url": "https://example.com/a",
                             "content": f"Fake result about {query}"}]}


def test_load_and_split():
    docs = kb.load_documents(ROOT / "crew_data")
    sources = {d.metadata["source"] for d in docs}
    assert any(s.endswith(".pdf") for s in sources) and any(s.endswith(".md") for s in sources)
    chunks = kb.split_documents(docs, 800, 120)
    assert len(chunks) > len(docs)
    assert max(len(c.page_content) for c in chunks) <= 800


def test_retrieval_finds_known_fact(store):
    hits = store.similarity_search("How many hours per month do medical residents spend reading patient charts?", k=4)
    assert any("112 hours" in " ".join(h.page_content.split()) for h in hits)


def test_index_is_cached_and_reloads(settings, store):
    again = kb.get_vectorstore(settings)
    assert again.index.ntotal == store.index.ntotal


def test_chroma_backend_matches_faiss(settings, store):
    chunks = kb.split_documents(kb.load_documents(settings.data_dir))
    chroma = kb.build_vectorstore(chunks, kb.get_embeddings(settings), "chroma")
    q = "What is the SPeC framework?"
    f_ids = [d.metadata["chunk_id"] for d in store.similarity_search(q, k=3)]
    c_ids = [d.metadata["chunk_id"] for d in chroma.similarity_search(q, k=3)]
    assert f_ids[0] == c_ids[0]


def test_rag_chain(settings, store):
    from rag_app.rag_chain import build_rag_chain

    out = build_rag_chain(store, chat_model(settings)).invoke("What does Sporo Health do?")
    assert out["answer"].startswith("FAKE ANSWER") and len(out["docs"]) == 4


def test_tools(store):
    from rag_app.tools import KnowledgeBaseTool, WebSearchTool

    assert "[sporo" in KnowledgeBaseTool(store).run(query="patient chart review").lower()
    assert "https://example.com/a" in WebSearchTool(api_key="x", client=FakeTavily()).run(query="AI")


def test_research_crew_runs_and_calls_tools(settings, store, fake_llm):
    from rag_app.crew import run_research_crew
    from rag_app.tools import KnowledgeBaseTool, WebSearchTool

    res = run_research_crew("Does Sporo streamline patient chart reviews?", crew_llm(settings),
                            KnowledgeBaseTool(store), WebSearchTool("x", client=FakeTavily()))
    assert res["answer"] and res["research"] and res["draft"]
    reqs = [json.loads(l) for l in fake_llm[1].read_text().splitlines()]
    tool_msgs = [m for r in reqs for m in r["messages"] if m.get("role") == "tool"]
    assert tool_msgs, "the researcher never received a tool result"
    assert any("Sporo" in str(m.get("content")) for m in tool_msgs)


def test_trend_and_report(settings, store, tmp_path):
    from rag_app.crew import run_report_crew, run_trend_analysis
    from rag_app.report import save_report
    from rag_app.tools import KnowledgeBaseTool, WebSearchTool

    web = WebSearchTool("x", client=FakeTavily())
    assert run_trend_analysis("AI in healthcare", crew_llm(settings), web)
    md = run_report_crew("AI clinical summarisation", crew_llm(settings), KnowledgeBaseTool(store), web)
    paths = save_report("# Title\n\n## Summary\n\n- point “quoted” — dash\n\n" + md, "AI test", tmp_path)
    assert paths["markdown"].exists() and paths["pdf"].stat().st_size > 500


def test_missing_key_message():
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        crew_llm(Settings(groq_api_key=""))
