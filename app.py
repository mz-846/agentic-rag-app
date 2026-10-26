"""Streamlit front-end for the Agentic RAG app (a mini "Perplexity" for your documents).

Run:  streamlit run app.py

Modes
- Quick answer   : plain RAG (retrieve top-k chunks -> one LLM call). Fast and cheap.
- Agent crew     : Researcher (knowledge base + web) -> Writer -> Critic. Slower, more thorough.
- Trend report   : Researcher + Trend Analyst -> Report Generator; download as .md / .pdf.
"""
import os
from pathlib import Path

import streamlit as st

from rag_app.config import GROQ_MODELS, Settings, chat_model, crew_llm
from rag_app.knowledge_base import SUPPORTED, cite, get_vectorstore

st.set_page_config(page_title="Agentic RAG", page_icon="🔎", layout="wide")
settings = Settings()

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    groq_key = st.text_input("Groq API key", value=settings.groq_api_key, type="password",
                             help="Free key at console.groq.com/keys. Read from .env if set.")
    tavily_key = st.text_input("Tavily API key", value=settings.tavily_api_key, type="password",
                               help="Free key at app.tavily.com. Needed for web search.")
    model = st.selectbox("Model", GROQ_MODELS + ([settings.llm_model] if settings.llm_model not in GROQ_MODELS else []),
                         index=(GROQ_MODELS + [settings.llm_model]).index(settings.llm_model))
    k = st.slider("Chunks to retrieve (k)", 2, 10, settings.k)
    use_web = st.toggle("Allow web search (Tavily)", value=bool(tavily_key))
    settings.groq_api_key, settings.tavily_api_key, settings.llm_model, settings.k = groq_key, tavily_key, model, k

    st.divider()
    st.subheader("📚 Knowledge base")
    docs = sorted(p.name for p in settings.data_dir.glob("*") if p.suffix.lower() in SUPPORTED)
    st.markdown("\n".join(f"- `{d}`" for d in docs) or "_No documents yet_")
    uploads = st.file_uploader("Add documents", type=[s.strip(".") for s in SUPPORTED],
                               accept_multiple_files=True)
    if uploads and st.button("Add to knowledge base"):
        for f in uploads:
            (settings.data_dir / Path(f.name).name).write_bytes(f.getbuffer())
        st.cache_resource.clear()
        st.success(f"Added {len(uploads)} file(s) - the index will rebuild.")
        st.rerun()


@st.cache_resource(show_spinner="Building the vector index (first run downloads the embedding model)…")
def load_store(_fingerprint: tuple):
    return get_vectorstore(Settings())


def kb_fingerprint():
    return tuple((p.name, p.stat().st_mtime) for p in sorted(settings.data_dir.glob("*")))


def tools(store):
    from rag_app.tools import KnowledgeBaseTool, WebSearchTool

    kb_tool = KnowledgeBaseTool(store, k=settings.k)
    web_tool = WebSearchTool(api_key=settings.tavily_api_key) if (use_web and settings.tavily_api_key) else None
    return kb_tool, web_tool


def show_sources(docs):
    with st.expander(f"📄 {len(docs)} retrieved passages"):
        for d in docs:
            st.markdown(f"**{cite(d)}**")
            st.caption(d.page_content[:600] + ("…" if len(d.page_content) > 600 else ""))


# ----------------------------------------------------------------------------- main
st.title("🔎 Agentic RAG: ask your documents and the web")
st.caption("LangChain + FAISS retrieval, Tavily web search and CrewAI agents, served by Groq.")

mode = st.radio("Mode", ["Quick answer", "Agent crew", "Trend report"], horizontal=True,
                captions=["RAG, one LLM call", "Researcher → Writer → Critic", "Report → .md / .pdf"])

if not settings.groq_api_key:
    st.info("Add your **Groq API key** in the sidebar (or a `.env` file) to start.")
    st.stop()

try:
    store = load_store(kb_fingerprint())
except Exception as e:  # e.g. no internet for the first embedding-model download
    st.error(f"Couldn't build the knowledge base: {e}")
    st.stop()

if mode == "Trend report":
    topic = st.text_input("Report topic", "AI for clinical summarisation and patient chart review")
    if st.button("Generate report", type="primary"):
        if not settings.tavily_api_key:
            st.warning("The trend report needs a Tavily key for web search.")
            st.stop()
        from rag_app.crew import run_report_crew
        from rag_app.report import save_report

        kb_tool, _ = tools(store)
        from rag_app.tools import WebSearchTool
        with st.spinner("Three agents are researching and writing - this takes a minute…"):
            md = run_report_crew(topic, crew_llm(settings), kb_tool, WebSearchTool(settings.tavily_api_key))
            paths = save_report(md, topic, settings.reports_dir)
        st.markdown(paths["markdown"].read_text(encoding="utf-8"))
        c1, c2 = st.columns(2)
        c1.download_button("⬇️ Markdown", paths["markdown"].read_bytes(), paths["markdown"].name)
        c2.download_button("⬇️ PDF", paths["pdf"].read_bytes(), paths["pdf"].name, "application/pdf")
    st.stop()

# chat modes
if "history" not in st.session_state:
    st.session_state.history = []
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask e.g. 'How much time do physicians spend on patient chart review?'")
if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            if mode == "Quick answer":
                from rag_app.rag_chain import build_rag_chain

                with st.spinner("Retrieving and answering…"):
                    out = build_rag_chain(store, chat_model(settings), k=settings.k).invoke(question)
                answer = out["answer"]
                st.markdown(answer)
                show_sources(out["docs"])
            else:
                from rag_app.crew import run_research_crew

                kb_tool, web_tool = tools(store)
                with st.status("Agents at work…", expanded=False) as status:
                    res = run_research_crew(question, crew_llm(settings), kb_tool, web_tool)
                    status.update(label="Done", state="complete")
                answer = res["answer"]
                st.markdown(answer)
                with st.expander("🔬 Researcher notes"):
                    st.markdown(res["research"])
                with st.expander("✍️ Writer's first draft"):
                    st.markdown(res["draft"])
        except Exception as e:
            answer = f"⚠️ {type(e).__name__}: {e}"
            st.error(answer)
    st.session_state.history.append({"role": "assistant", "content": answer})
