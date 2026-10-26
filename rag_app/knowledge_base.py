"""Knowledge base: load documents -> split into chunks -> embed -> vector store.

    docs   = load_documents("crew_data")
    chunks = split_documents(docs, chunk_size=800, chunk_overlap=120)
    store  = build_vectorstore(chunks, get_embeddings(settings))
    store.similarity_search_with_score("What does Sporo Health do?", k=4)

`get_vectorstore(settings)` wraps all of this and caches the FAISS index on disk, rebuilding
it automatically when files in crew_data/ change (Task 2: drop a file in, rebuild, query).
"""
from __future__ import annotations

import hashlib
import json
import re
import warnings
from pathlib import Path

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings

warnings.filterwarnings("ignore", message=".*langchain-community.*")
SUPPORTED = {".pdf", ".md", ".txt"}


# --------------------------------------------------------------------------- loading
def load_documents(data_dir: str | Path) -> list[Document]:
    """Load every PDF / Markdown / text file in `data_dir` (one Document per PDF page)."""
    from pypdf import PdfReader

    docs = []
    for path in sorted(Path(data_dir).glob("**/*")):
        if path.suffix.lower() not in SUPPORTED:
            continue
        if path.suffix.lower() == ".pdf":
            for i, page in enumerate(PdfReader(path).pages):
                text = clean_text(page.extract_text() or "")
                if text:
                    docs.append(Document(page_content=text,
                                         metadata={"source": path.name, "page": i + 1}))
        else:
            docs.append(Document(page_content=clean_text(path.read_text(encoding="utf-8")),
                                 metadata={"source": path.name, "page": None}))
    return docs


def clean_text(text: str) -> str:
    """Fix the usual PDF-extraction noise: hyphenated line breaks and ragged whitespace."""
    text = re.sub(r"-\n(\w)", r"\1", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_documents(docs: list[Document], chunk_size: int = 800,
                    chunk_overlap: int = 120) -> list[Document]:
    """Recursive splitting tries paragraph -> line -> sentence -> word boundaries in turn,
    so chunks end at natural breaks. Overlap keeps a fact that straddles a boundary
    retrievable from either side."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                              separators=["\n\n", "\n", ". ", " ", ""])
    chunks = splitter.split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
    return chunks


# --------------------------------------------------------------------------- embeddings
class HashingEmbeddings(Embeddings):
    """Offline stand-in for a real embedding model (no download, no internet).

    Hashes word uni/bi-grams into a fixed-size vector (like sklearn's HashingVectorizer).
    It only captures word overlap, not meaning, so use it for tests and demos without
    internet - real answers should use the BGE model via FastEmbed.
    """

    def __init__(self, dim: int = 768):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.vec = HashingVectorizer(n_features=dim, ngram_range=(1, 2), stop_words="english",
                                     alternate_sign=False, norm="l2")

    def _embed(self, texts):
        return self.vec.transform(texts).toarray().astype(np.float32).tolist()

    def embed_documents(self, texts):
        return self._embed(texts)

    def embed_query(self, text):
        return self._embed([text])[0]


def get_embeddings(s: Settings) -> Embeddings:
    if s.embedding_backend == "hash":
        return HashingEmbeddings()
    from langchain_community.embeddings import FastEmbedEmbeddings

    # Downloads the ~65 MB ONNX model from Hugging Face on first use, then caches it.
    return FastEmbedEmbeddings(model_name=s.embedding_model)


# --------------------------------------------------------------------------- vector stores
def build_vectorstore(chunks: list[Document], embeddings: Embeddings, backend: str = "faiss",
                      persist_dir: str | Path | None = None):
    if backend == "chroma":
        from langchain_chroma import Chroma

        return Chroma.from_documents(chunks, embeddings, collection_name="crew_data",
                                     collection_metadata={"hnsw:space": "cosine"},
                                     persist_directory=str(persist_dir) if persist_dir else None)
    from langchain_community.vectorstores import FAISS
    from langchain_community.vectorstores.utils import DistanceStrategy

    # Cosine similarity: embeddings are normalised, so max-inner-product == cosine.
    return FAISS.from_documents(chunks, embeddings,
                                distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
                                normalize_L2=True)


def _fingerprint(s: Settings) -> str:
    files = [(p.name, p.stat().st_size, int(p.stat().st_mtime))
             for p in sorted(s.data_dir.glob("**/*")) if p.suffix.lower() in SUPPORTED]
    key = json.dumps([files, s.embedding_backend, s.embedding_model, s.chunk_size,
                      s.chunk_overlap])
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def get_vectorstore(s: Settings, force_rebuild: bool = False):
    """Load the cached FAISS index, or (re)build it if crew_data/ or settings changed."""
    from langchain_community.vectorstores import FAISS
    from langchain_community.vectorstores.utils import DistanceStrategy

    embeddings = get_embeddings(s)
    fp = _fingerprint(s)
    index_path = s.index_dir / fp
    if index_path.exists() and not force_rebuild:
        # Safe: this pickle was written by us, from our own documents.
        return FAISS.load_local(str(index_path), embeddings, allow_dangerous_deserialization=True,
                                distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
                                normalize_L2=True)
    chunks = split_documents(load_documents(s.data_dir), s.chunk_size, s.chunk_overlap)
    if not chunks:
        raise RuntimeError(f"No .pdf/.md/.txt documents found in {s.data_dir}")
    store = build_vectorstore(chunks, embeddings, "faiss")
    index_path.mkdir(parents=True, exist_ok=True)
    store.save_local(str(index_path))
    return store


def format_docs(docs: list[Document]) -> str:
    """Render retrieved chunks with a citation tag the LLM can quote, e.g. [doc.pdf p.3]."""
    return "\n\n".join(f"[{cite(d)}]\n{d.page_content}" for d in docs)


def cite(d: Document) -> str:
    page = d.metadata.get("page")
    return f"{d.metadata.get('source', '?')}" + (f" p.{page}" if page else "")
