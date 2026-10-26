"""Plain (non-agentic) Retrieval-Augmented QA with LangChain.

question -> retriever (top-k chunks) -> prompt with the chunks as context -> LLM -> answer

This is the modern LCEL equivalent of the old `RetrievalQA.from_chain_type(...)` shown in
the course brief (RetrievalQA was removed from LangChain 1.x).
"""
from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

from .knowledge_base import format_docs

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You answer questions using ONLY the context below. Cite sources in square brackets "
     "exactly as they appear in the context, e.g. [doc.pdf p.2]. If the context does not "
     "contain the answer, say you don't know - do not make anything up.\n\n"
     "Context:\n{context}"),
    ("human", "{question}"),
])


def build_rag_chain(vectorstore, llm, k: int = 4):
    """Returns a runnable: chain.invoke(question) -> {"answer": str, "docs": [Document]}."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    answer = (
        {"context": RunnableLambda(lambda x: format_docs(x["docs"])),
         "question": RunnableLambda(lambda x: x["question"])}
        | RAG_PROMPT | llm | StrOutputParser()
    )
    return RunnableParallel(docs=retriever, question=RunnablePassthrough()).assign(answer=answer)
