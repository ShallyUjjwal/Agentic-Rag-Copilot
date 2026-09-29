import logging
from typing import Literal

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from tavily import TavilyClient

from app.core.config import get_settings
from app.rag.state import AgentState
from app.rag.vectorstore import get_retriever


logger = logging.getLogger(__name__)

settings = get_settings()

_llm = None
_tavily = None


def llm():
    """
    Return a cached Groq chat model.

    Returns:
        ChatGroq: Configured Groq LLM.
    """

    global _llm

    if _llm is None:

        if not settings.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is missing"
            )

        _llm = ChatGroq(
            model=settings.groq_model,
            temperature=0,
            api_key=settings.groq_api_key,
        )

    return _llm


def web_search_client():
    """
    Return a cached Tavily client.

    Returns:
        TavilyClient: Tavily search client.
    """

    global _tavily

    if _tavily is None:

        if not settings.tavily_api_key:
            raise RuntimeError(
                "TAVILY_API_KEY is missing"
            )

        _tavily = TavilyClient(
            api_key=settings.tavily_api_key
        )

    return _tavily


def add_trace(
    state: AgentState,
    message: str,
):
    """
    Append an execution message to the agent trace.

    Returns:
        List[str]: Updated trace.
    """

    return [
        *state.get("trace", []),
        message,
    ]


# ============================================================
# ROUTER
# ============================================================

def route_question(state: AgentState):
    """
    Decide whether the question requires company knowledge
    or can be answered directly.

    Returns:
        dict containing source_used and trace.
    """

    prompt = f"""
You are the router for an enterprise HR policy assistant.

Classify the user's message into exactly one category:

kb
direct

Use "kb" for questions about:
- company HR policies
- leave
- holidays
- benefits
- payroll
- remote work
- attendance
- onboarding
- performance
- expenses
- travel
- employee conduct
- company procedures
- employee support

Use "direct" only for:
- greetings
- thanks
- casual conversation
- simple messages that do not require company knowledge

User question:
{state["question"]}

Return exactly one word:
kb
or
direct
"""

    result = llm().invoke(prompt)

    route = result.content.strip().lower()

    if route not in {"kb", "direct"}:
        route = "kb"

    return {
        "source_used": route,
        "trace": add_trace(
            state,
            f"Router → {route.upper()}",
        ),
    }


def route_after_router(
    state: AgentState,
) -> Literal[
    "retrieve_kb",
    "direct_answer",
]:

    if state["source_used"] == "kb":
        return "retrieve_kb"

    return "direct_answer"


# ============================================================
# PRIVATE KNOWLEDGE BASE
# ============================================================

def retrieve_kb(state: AgentState):
    """
    Retrieve relevant chunks from the private HR knowledge base.

    Returns:
        dict containing retrieved documents and trace.
    """

    retriever = get_retriever()

    documents = retriever.invoke(
        state["current_query"]
    )

    return {
        "kb_docs": documents,
        "trace": add_trace(
            state,
            f"Private KB retrieval → {len(documents)} chunks",
        ),
    }


def grade_kb(state: AgentState):
    """
    Grade whether retrieved company documents contain
    enough evidence to answer the question.

    Returns:
        dict containing kb_grade and trace.
    """

    if not state["kb_docs"]:
        return {
            "kb_grade": "weak",
            "trace": add_trace(
                state,
                "KB evidence grade → WEAK",
            ),
        }

    context = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')}\n"
        f"{doc.page_content}"
        for doc in state["kb_docs"]
    )

    prompt = f"""
You are an HR document relevance grader.

Determine whether the private company HR documents
contain enough information to answer the employee's question.

Employee question:
{state["question"]}

Private HR documents:
{context}

Return exactly one word:

good

if the documents contain sufficient relevant evidence.

Return:

weak

if the documents do not contain enough evidence.

Do not explain your answer.
Return only:
good
or
weak
"""

    result = llm().invoke(prompt)

    grade = result.content.strip().lower()

    if grade not in {"good", "weak"}:
        grade = "weak"

    return {
        "kb_grade": grade,
        "trace": add_trace(
            state,
            f"KB evidence grade → {grade.upper()}",
        ),
    }


def after_kb(
    state: AgentState,
) -> Literal[
    "generate_from_kb",
    "search_web",
]:

    if state["kb_grade"] == "good":
        return "generate_from_kb"

    return "search_web"


# ============================================================
# WEB SEARCH FALLBACK
# ============================================================

def search_web(state: AgentState):
    """
    Search the public web using Tavily when private
    company knowledge is insufficient.

    Returns:
        dict containing web results, citations and trace.
    """

    response = web_search_client().search(
        query=state["current_query"],
        search_depth="advanced",
        max_results=4,
    )

    results = response.get(
        "results",
        [],
    )

    lines = []
    citations = []

    for result in results:

        title = result.get(
            "title",
            "",
        )

        url = result.get(
            "url",
            "",
        )

        content = result.get(
            "content",
            "",
        )

        lines.append(
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Content: {content}"
        )

        if url:
            citations.append(
                {
                    "title": title or url,
                    "url": url,
                    "type": "web",
                }
            )

    return {
        "web_results": "\n\n".join(lines),
        "citations": citations,
        "source_used": "web",
        "trace": add_trace(
            state,
            "Web fallback → Tavily search",
        ),
    }


def grade_web(state: AgentState):
    """
    Determine whether web search returned enough
    relevant evidence.

    Returns:
        dict containing web_grade and trace.
    """

    if not state["web_results"].strip():
        return {
            "web_grade": "weak",
            "trace": add_trace(
                state,
                "Web evidence grade → WEAK",
            ),
        }

    prompt = f"""
You are an evidence grader for an HR support assistant.

Employee question:
{state["question"]}

Public web evidence:
{state["web_results"]}

Determine whether the web evidence is directly relevant
and sufficient to provide a useful answer.

Return exactly one word:

good

or:

weak

Do not explain.
"""

    result = llm().invoke(prompt)

    grade = result.content.strip().lower()

    if grade not in {"good", "weak"}:
        grade = "weak"

    return {
        "web_grade": grade,
        "trace": add_trace(
            state,
            f"Web evidence grade → {grade.upper()}",
        ),
    }


def after_web(
    state: AgentState,
) -> Literal[
    "generate_from_web",
    "rewrite_query",
    "insufficient",
]:

    if state["web_grade"] == "good":
        return "generate_from_web"

    if state["retry_count"] < settings.max_retries:
        return "rewrite_query"

    return "insufficient"


# ============================================================
# QUERY REWRITING
# ============================================================

def rewrite_query(state: AgentState):
    """
    Rewrite the user's question to improve retrieval.

    Returns:
        dict containing updated query, retry count and trace.
    """

    prompt = f"""
You are an HR search query optimizer.

Rewrite the following employee question into a clearer
search query for an HR knowledge base and public web search.

Preserve the original intent.

Add useful HR or policy terminology where appropriate.

Do not answer the question.

Return only the rewritten search query.

Original question:
{state["question"]}
"""

    result = llm().invoke(prompt)

    rewritten = result.content.strip()

    return {
        "current_query": rewritten,
        "retry_count": state["retry_count"] + 1,
        "trace": add_trace(
            state,
            f"Query rewrite → {rewritten}",
        ),
    }


# ============================================================
# ANSWER GENERATION
# ============================================================

def generate_from_kb(state: AgentState):
    """
    Generate an answer using only private company documents.

    Returns:
        dict containing answer, citations and trace.
    """

    context = "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n"
        f"{doc.page_content}"
        for doc in state["kb_docs"]
    )

    prompt = f"""
You are an enterprise HR policy and employee support copilot.

Answer the employee's question ONLY using the private
company HR knowledge base below.

Rules:
- Do not invent policy details.
- Do not use information that is not in the supplied documents.
- Be concise and practical.
- Be respectful.
- If the documents do not specify something, say so.
- If there are steps, present them clearly.
- State that the answer is based on the company's private
  knowledge base.

Employee question:
{state["question"]}

Private company HR knowledge base:
{context}
"""

    answer = llm().invoke(prompt).content.strip()

    citations = []
    seen = set()

    for document in state["kb_docs"]:

        source = document.metadata.get(
            "source",
            "Private KB",
        )

        if source not in seen:

            seen.add(source)

            citations.append(
                {
                    "title": source.split("/")[-1],
                    "url": "",
                    "type": "private_kb",
                }
            )

    return {
        "answer": answer,
        "source_used": "private_kb",
        "citations": citations,
        "trace": add_trace(
            state,
            "Answer generation → PRIVATE KB",
        ),
    }


def generate_from_web(state: AgentState):
    """
    Generate an answer using public web evidence.

    Returns:
        dict containing answer, source and trace.
    """

    prompt = f"""
You are an enterprise HR policy and employee support copilot.

The private company HR knowledge base was insufficient.

Answer ONLY from the public web evidence below.

Important:
- Clearly state that the information comes from external
  public sources.
- Do not present external information as company policy.
- Tell the employee that company HR should validate the
  information when appropriate.
- Do not invent facts.

Employee question:
{state["question"]}

Public web evidence:
{state["web_results"]}
"""

    answer = llm().invoke(prompt).content.strip()

    return {
        "answer": answer,
        "source_used": "web_search",
        "trace": add_trace(
            state,
            "Answer generation → WEB SEARCH",
        ),
    }


def direct_answer(state: AgentState):
    """
    Handle greetings and casual conversation without retrieval.

    Returns:
        dict containing direct answer and trace.
    """

    answer = llm().invoke(
        f"""
Respond briefly and naturally to:

{state["question"]}
"""
    ).content.strip()

    return {
        "answer": answer,
        "source_used": "direct",
        "trace": add_trace(
            state,
            "Direct response → no retrieval",
        ),
    }


def insufficient(state: AgentState):
    """
    Return a safe response when neither private KB nor
    web search provides enough evidence.

    Returns:
        dict containing fallback answer and trace.
    """

    return {
        "answer": (
            "I couldn't find enough reliable evidence in "
            "the company HR knowledge base or external "
            "search to answer confidently. Please contact "
            "the HR team or provide more details."
        ),
        "source_used": "insufficient_evidence",
        "trace": add_trace(
            state,
            "Stopped → insufficient reliable evidence",
        ),
    }


# ============================================================
# LANGGRAPH
# ============================================================

def build_graph():
    """
    Build and compile the Agentic RAG LangGraph.

    Returns:
        Compiled LangGraph agent.
    """

    graph = StateGraph(
        AgentState
    )

    graph.add_node(
        "route_question",
        route_question,
    )

    graph.add_node(
        "retrieve_kb",
        retrieve_kb,
    )

    graph.add_node(
        "grade_kb",
        grade_kb,
    )

    graph.add_node(
        "search_web",
        search_web,
    )

    graph.add_node(
        "grade_web",
        grade_web,
    )

    graph.add_node(
        "rewrite_query",
        rewrite_query,
    )

    graph.add_node(
        "generate_from_kb",
        generate_from_kb,
    )

    graph.add_node(
        "generate_from_web",
        generate_from_web,
    )

    graph.add_node(
        "direct_answer",
        direct_answer,
    )

    graph.add_node(
        "insufficient",
        insufficient,
    )

    # START → Router
    graph.add_edge(
        START,
        "route_question",
    )

    # Router → KB or Direct
    graph.add_conditional_edges(
        "route_question",
        route_after_router,
        {
            "retrieve_kb": "retrieve_kb",
            "direct_answer": "direct_answer",
        },
    )

    # KB retrieval → grading
    graph.add_edge(
        "retrieve_kb",
        "grade_kb",
    )

    # KB grading → answer or web
    graph.add_conditional_edges(
        "grade_kb",
        after_kb,
        {
            "generate_from_kb": "generate_from_kb",
            "search_web": "search_web",
        },
    )

    # Web → grading
    graph.add_edge(
        "search_web",
        "grade_web",
    )

    # Web grading → answer/rewrite/failure
    graph.add_conditional_edges(
        "grade_web",
        after_web,
        {
            "generate_from_web": "generate_from_web",
            "rewrite_query": "rewrite_query",
            "insufficient": "insufficient",
        },
    )

    # Rewrite → retrieve again
    graph.add_edge(
        "rewrite_query",
        "retrieve_kb",
    )

    # Terminal nodes
    graph.add_edge(
        "generate_from_kb",
        END,
    )

    graph.add_edge(
        "generate_from_web",
        END,
    )

    graph.add_edge(
        "direct_answer",
        END,
    )

    graph.add_edge(
        "insufficient",
        END,
    )

    return graph.compile()


agent_graph = build_graph()


# ============================================================
# PUBLIC API
# ============================================================

def ask(question: str):
    """
    Run the Agentic RAG workflow for a user question.

    Input:
        question: Employee's question.

    Returns:
        dict: Final agent state containing answer,
              source, trace and citations.
    """

    initial: AgentState = {
        "question": question,
        "current_query": question,
        "kb_docs": [],
        "web_results": "",
        "kb_grade": "",
        "web_grade": "",
        "answer": "",
        "source_used": "",
        "retry_count": 0,
        "trace": [],
        "citations": [],
    }

    return agent_graph.invoke(
        initial
    )