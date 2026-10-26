"""Multi-agent workflows with CrewAI.

1. Research crew  (core)   : Researcher -> Writer -> Critic
2. Trend analyst  (Task 3) : one agent with the Tavily tool, "summarise the latest news on X"
3. Report crew    (Task 4) : Researcher + Trend Analyst -> Report Generator -> .md / .pdf

Each agent = role + goal + backstory + tools + LLM. Each Task says what to do and what the
output must look like; `context=[...]` hands earlier tasks' outputs to later ones, which is
how agents collaborate in a sequential process.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import date

from crewai import Agent, Crew, Process, Task


def kickoff(crew: Crew, inputs: dict):
    """crew.kickoff() that also works inside Jupyter.

    CrewAI refuses to run synchronously when an asyncio event loop is already running -
    which is always the case in a Jupyter notebook. Running it in a worker thread (no
    event loop there) sidesteps that, and is a no-op difference in scripts / Streamlit.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return crew.kickoff(inputs=inputs)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(crew.kickoff, inputs=inputs).result()

# ----------------------------------------------------------------------------- agents


def researcher(llm, tools) -> Agent:
    return Agent(
        role="Researcher",
        goal="Find accurate, well-sourced facts that answer the user's question.",
        backstory=(
            "You are a meticulous research analyst. You ALWAYS search the private knowledge "
            "base first; you use web search only for things the documents don't cover or that "
            "need up-to-date information. You never invent facts and you keep every source."),
        tools=tools, llm=llm, allow_delegation=False, max_iter=6, verbose=False)


def writer(llm) -> Agent:
    return Agent(
        role="Content Writer",
        goal="Turn research notes into a clear, well-structured answer for a smart non-expert.",
        backstory=(
            "You are a science communicator. You write concisely, structure answers with short "
            "paragraphs or bullet points, and keep the researcher's citations next to each claim."),
        llm=llm, allow_delegation=False, verbose=False)


def critic(llm) -> Agent:
    return Agent(
        role="Reviewer",
        goal="Make sure the final answer is accurate, grounded in the sources, complete and clear.",
        backstory=(
            "You are a sceptical editor and fact-checker. You compare every claim in the draft "
            "with the research notes, remove anything unsupported, fix gaps and unclear wording, "
            "and return the improved final version."),
        llm=llm, allow_delegation=False, verbose=False)


def trend_analyst(llm, web_tool) -> Agent:
    """Task 3: a new agent role with the Tavily tool."""
    return Agent(
        role="Trend Analyst",
        goal="Summarise the most important recent developments on a topic, with dates and links.",
        backstory=(
            "You track technology news for an executive briefing. You search the web for the "
            "latest developments, ignore hype and duplicates, and report what changed and why it "
            "matters - always with the source URL."),
        tools=[web_tool], llm=llm, allow_delegation=False, max_iter=5, verbose=False)


def report_generator(llm) -> Agent:
    """Task 4: collects insights from other agents and writes a structured report."""
    return Agent(
        role="Report Generator",
        goal="Compile research and trend findings into a polished, structured Markdown report.",
        backstory=(
            "You are a consultant who writes crisp briefing documents. You organise material "
            "under clear headings, lead with the key takeaways, and end with a sources list."),
        llm=llm, allow_delegation=False, verbose=False)


# ----------------------------------------------------------------------------- crews


def run_research_crew(question: str, llm, kb_tool, web_tool=None) -> dict:
    """Researcher -> Writer -> Critic. Returns the final answer plus each stage's output."""
    tools = [kb_tool] + ([web_tool] if web_tool else [])
    r, w, c = researcher(llm, tools), writer(llm), critic(llm)

    research = Task(
        description=("Research this question: {question}\n"
                     "Search the knowledge base first; use web search only if needed. "
                     "Collect the relevant facts, figures and quotes."),
        expected_output=("Bullet-point research notes. Every bullet ends with its source in "
                         "square brackets, e.g. [doc.pdf p.3] or [URL]. Note anything you "
                         "could not find."),
        agent=r)
    draft = Task(
        description="Write an answer to the question '{question}' using ONLY the research notes.",
        expected_output=("A clear answer of 100-250 words in Markdown, with the citations from "
                         "the notes kept next to the claims they support."),
        agent=w, context=[research])
    review = Task(
        description=("Fact-check the draft answer to '{question}' against the research notes. "
                     "Remove or correct anything not supported by the notes, fill obvious gaps, "
                     "and improve clarity."),
        expected_output=("The final improved answer in Markdown with citations, followed by a "
                         "line '---' and a section '**Reviewer notes:**' listing what you "
                         "changed (or 'No changes needed')."),
        agent=c, context=[research, draft])

    crew = Crew(agents=[r, w, c], tasks=[research, draft, review], process=Process.sequential,
                verbose=False)
    out = kickoff(crew, {"question": question})
    return {"answer": out.raw, "research": research.output.raw, "draft": draft.output.raw,
            "usage": getattr(out, "token_usage", None)}


def trend_task(agent: Agent) -> Task:
    return Task(
        description=("Today is {today}. Search the web for the most important developments in "
                     "the last few months on: {topic}. Run 2-3 different searches."),
        expected_output=("5-7 bullet points, newest first. Each: what happened, when, why it "
                         "matters, and the source URL in square brackets."),
        agent=agent)


def run_trend_analysis(topic: str, llm, web_tool) -> str:
    """Task 3: e.g. run_trend_analysis('AI in healthcare', ...)"""
    a = trend_analyst(llm, web_tool)
    t = trend_task(a)
    out = kickoff(Crew(agents=[a], tasks=[t], verbose=False),
                  {"topic": topic, "today": date.today().isoformat()})
    return out.raw


def run_report_crew(topic: str, llm, kb_tool, web_tool) -> str:
    """Task 4: Researcher + Trend Analyst feed a Report Generator. Returns Markdown."""
    r = researcher(llm, [kb_tool, web_tool])
    ta = trend_analyst(llm, web_tool)
    rg = report_generator(llm)
    research = Task(
        description=("Research the topic '{topic}' in depth using the knowledge base first and "
                     "the web second. Gather key facts, numbers, challenges and open questions."),
        expected_output="Detailed bullet-point notes, each with a [source] citation.",
        agent=r)
    trends = trend_task(ta)
    report = Task(
        description=("Write a briefing report on '{topic}' (date: {today}) from the research "
                     "notes and the trend summary."),
        expected_output=(
            "A Markdown report with exactly these sections:\n"
            "# <Title>\n## Executive summary (3-4 sentences)\n## Key findings (bullets)\n"
            "## Latest developments (bullets with dates)\n## Challenges and open questions\n"
            "## Sources (numbered list of every document/URL cited)\n"
            "Use citations like [1] in the text that match the Sources list. Do not wrap the "
            "report in a code block."),
        agent=rg, context=[research, trends])
    out = kickoff(Crew(agents=[r, ta, rg], tasks=[research, trends, report],
                       process=Process.sequential, verbose=False),
                  {"topic": topic, "today": date.today().isoformat()})
    return out.raw
