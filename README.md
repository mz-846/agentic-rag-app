# 🔎 Agentic RAG App: Documents + Web Search + AI Agents

A small "Perplexity for your own documents". You can ask a question three ways:

| Mode | What happens |
|---|---|
| **Quick answer** | Classic RAG. The top-k chunks are retrieved from a FAISS index of `crew_data/`, and one LLM call writes a cited answer. |
| **Agent crew** | CrewAI agents work in sequence. The **Researcher** chooses its own tools (knowledge base, Tavily web search). The **Writer** drafts an answer. The **Critic** fact-checks the draft against the research notes and returns the final answer. |
| **Trend report** | The Researcher and a **Trend Analyst** pass their findings to a **Report Generator**. The report downloads as Markdown or PDF. |

**Stack:** LangChain · FAISS (or Chroma) · BGE embeddings via FastEmbed · Groq (OpenAI-compatible API) · Tavily · CrewAI · Streamlit · Docker

Built for the IntElligence Data Science Training & Internship *End-to-End Project: Agentic RAG App (LangChain, Groq, Tavily)*.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                     # then paste your keys into .env
streamlit run app.py                                     # http://localhost:8501
```

| Key | Where to get it (both free) |
|---|---|
| `GROQ_API_KEY` | <https://console.groq.com/keys> |
| `TAVILY_API_KEY` | <https://app.tavily.com> (only needed for web search and reports) |

You can also paste the keys into the app's sidebar. **Keys are never stored in the code or the notebook.** `.env` is git-ignored.

On first run the app downloads the embedding model (BAAI/bge-small-en-v1.5, about 65 MB) and builds the index. After that, startup is instant.

### Docker

```bash
docker build -t agentic-rag .
docker run -p 8501:8501 --env-file .env agentic-rag
```

## Project structure

```
agentic-rag-app/
├── analysis.ipynb          # step-by-step walkthrough of everything below + student tasks + bonus experiments
├── app.py                  # Streamlit UI
├── rag_app/
│   ├── config.py           # settings, keys from .env, Groq / Ollama model factories
│   ├── knowledge_base.py   # load → clean → chunk → embed → FAISS/Chroma (cached, auto-rebuilds)
│   ├── rag_chain.py        # plain RAG chain (LCEL)
│   ├── tools.py            # CrewAI tools: knowledge_base_search, web_search (Tavily)
│   ├── crew.py             # agents + crews: research, trend analyst, report generator
│   └── report.py           # Markdown → .md + .pdf export
├── crew_data/              # knowledge base: drop PDFs / .md / .txt here
│   ├── sporo_health_clinical_summarization.pdf
│   └── rag_and_agents_primer.md
├── tests/                  # offline tests (no keys, no internet)
├── requirements.txt / requirements-dev.txt
├── Dockerfile
└── .env.example
```

## Student tasks: where each one is done

| Task | Where |
|---|---|
| 1. Explore the vector store: `.similarity_search()`, PCA / t-SNE plots | notebook §5 |
| 2. Expand the knowledge base and rebuild the index | `crew_data/rag_and_agents_primer.md`, notebook §6, and the app's upload box (the index rebuilds automatically) |
| 3. New agent: Trend Analyst with the Tavily tool | `rag_app/crew.py::trend_analyst`, notebook §10 |
| 4. Report generator agent, exported to Markdown and PDF | `rag_app/crew.py::run_report_crew` + `report.py`, notebook §11, app's "Trend report" mode |
| Bonus: local LLM instead of Groq | Ollama via `LLM_BASE_URL`, notebook §12 |
| Bonus: Chroma instead of FAISS | `build_vectorstore(..., backend="chroma")`, notebook §12 |
| Bonus: chunk sizes, overlaps, embedding models | retrieval hit rate and MRR on a 12-question labelled set, notebook §12 |
| Bonus: Streamlit app + Docker | `app.py`, `Dockerfile` |

## Design notes and changes from the starter

- **Model.** The starter's `llama3-8b-8192` is no longer offered by Groq. The default is now `openai/gpt-oss-20b`, a current Groq production model. You can switch to `gpt-oss-120b` or the Llama 3.x models in the sidebar or with `LLM_MODEL`. CrewAI and LangChain both call Groq's OpenAI-compatible endpoint, so a single setting switches the whole app to a local Ollama model.
- **Embeddings.** BGE-small runs through **FastEmbed** (ONNX) instead of sentence-transformers. The results are the same, but there's no PyTorch install, so the Docker image is much smaller.
- **Secrets.** The starter asked for API keys to be pasted into notebook cells, where they would be committed to GitHub. Keys now come from `.env` or the sidebar.
- **Agents.** The brief's Researcher → Writer → Critic design replaces the starter's 5-agent router/grader chain. The Researcher chooses between tools itself rather than following a keyword `if` statement.
- **Jupyter compatibility.** CrewAI 1.x refuses to run synchronously inside Jupyter's event loop. `rag_app.crew.kickoff()` handles this, so the same functions work in the notebook, scripts and Streamlit.
- **Citations.** Every chunk carries `source` and `page` metadata. Prompts require `[file p.N]` citations, and the plain RAG prompt is told to say "I don't know" when the context doesn't contain the answer.
- **Telemetry.** CrewAI's anonymous telemetry is switched off by default.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

The 9 tests run **offline**. They use a hashing-embedding stand-in, a fake Tavily client and a tiny fake OpenAI-compatible server in place of Groq. They check the plumbing: loading, chunking, FAISS/Chroma retrieval, the RAG chain, tool calls reaching the agents, all three crews, and PDF export. They don't test answer quality.

> **Status:** the code, the app and every notebook cell have been run end-to-end against these stand-ins. They have **not** yet been run against the live Groq and Tavily APIs, because API keys and internet access weren't available where this was built. Run `analysis.ipynb` once with your keys to produce the real outputs before submitting.

## Credits

- `crew_data/sporo_health_clinical_summarization.pdf`: Lee, Vogt & Kumar (2024), *Assessing the Role of Clinical Summarization and Patient Chart Review within Communications, Medical Management, and Diagnostics*, arXiv:2407.16905. Supplied as the course's sample document.
- Based on the course starter code: [furkhan67/agentic_rag_app-perplexity-clone-](https://github.com/furkhan67/agentic_rag_app-perplexity-clone-).
