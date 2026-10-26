# Primer: Retrieval-Augmented Generation and AI Agents

*Study notes written for this project. Added to the knowledge base to practise Task 2 (expanding the knowledge base).*

## Why large language models hallucinate

A large language model (LLM) produces text by predicting the next token from patterns learned in training. It has no built-in way to check a claim against a source. When a question goes beyond what it saw in training, or asks for something specific like a figure, a date or a citation, the model still produces fluent text, and that text can be plausible but wrong. This is called hallucination. Two more limits make it worse. Knowledge is frozen at the training cutoff. Private documents (a company's reports, a patient's chart) were never in the training data at all.

## Retrieval-Augmented Generation (RAG)

RAG reduces hallucination by giving the model the relevant evidence at question time. The pipeline has two phases.

**Indexing (done once, or whenever documents change):**
1. *Load* the documents (PDF pages, web pages, notes).
2. *Chunk* them into passages of a few hundred to roughly a thousand characters. Chunks overlap a little so that a fact sitting on a boundary is not lost.
3. *Embed* each chunk: an embedding model maps text to a vector, so that passages with similar meaning end up close together. BAAI/bge-small-en-v1.5 is a small, strong open-source embedding model that produces 384-dimensional vectors.
4. *Store* the vectors in a vector index such as FAISS or Chroma. These support fast nearest-neighbour search.

**Querying (every question):**
1. Embed the question with the same embedding model.
2. *Retrieve* the k most similar chunks, using cosine similarity.
3. *Augment* the prompt: put the retrieved chunks into the prompt as context.
4. *Generate*: the LLM answers using that context and cites it.

## Choosing chunk size and overlap

Small chunks (around 300 characters) are precise, but a single chunk may not hold enough context to answer the question. Large chunks (around 1,500 characters) carry more context, but they dilute the embedding with unrelated sentences and use up the prompt budget. A common starting point is 500–1,000 characters with 10–20% overlap. The best way to choose is to measure: write a few questions whose answers you know, and check how often the right passage appears in the top-k results (retrieval hit rate).

## FAISS versus Chroma

FAISS (Facebook AI Similarity Search) is a library. It is very fast, runs in memory and can be saved to disk, but it is not a database: there is no server and no built-in metadata filtering. Chroma is an embedded vector database with persistence, metadata filters and collections. For a few thousand chunks, both return the same neighbours in milliseconds. The choice matters more for scale, filtering and operations than for accuracy.

## Web search as a retrieval tool

A document index only knows what is in its documents. A web-search API such as Tavily returns current results, already cleaned for LLM use, so an assistant can answer questions about recent events. Search results still need checking. Citing the URL lets the reader verify the answer.

## What makes a system "agentic"

An AI agent is an LLM placed in a loop. It is given a role, a goal and a set of tools. At each step it decides what to do next (call a tool, look at the result, call another tool, or give a final answer). This contrasts with a fixed pipeline, where the developer hard-codes the steps.

Multi-agent frameworks such as CrewAI split a job across specialised agents. Each agent has a role, a goal, a backstory and its own tools. A typical research crew:
- **Researcher**: gathers evidence from the document index and the web.
- **Writer**: turns the evidence into a clear answer.
- **Critic (reviewer)**: checks the draft against the evidence, removes unsupported claims and suggests improvements.

Splitting the work this way makes each prompt simpler. It also adds a self-checking step, at the cost of more LLM calls (more time and more tokens).

## Evaluating a RAG system

- **Retrieval quality**: hit rate@k (does the right passage appear in the top k?) and mean reciprocal rank (how high does it appear?).
- **Faithfulness**: is every claim in the answer supported by the retrieved context?
- **Answer relevance**: does the answer actually address the question?
- **Cost and latency**: tokens and seconds per answer. These matter for multi-agent workflows.

## Risks in healthcare settings

Clinical uses of RAG need extra care. Patient data must be handled in line with privacy law (for example GDPR in the UK and EU, HIPAA in the US). Answers must be traceable to their sources. A clinician must stay responsible for the final decision. Retrieval cannot fix errors in the source documents, so the quality of the underlying records still matters.
