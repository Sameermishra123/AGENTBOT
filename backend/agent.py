# rag_agent_app/backend/agent.py

import os
from typing import List, Literal, TypedDict
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

# Import API keys from config
from .config import GROQ_API_KEY, TAVILY_API_KEY
from .vectorstore import get_retriever

# --- Tools ---
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
tavily = TavilySearch(max_results=3, topic="general")

@tool
def web_search_tool(query: str) -> str:
    """Up-to-date web info via Tavily"""
    try:
        result = tavily.invoke({"query": query})
        if isinstance(result, dict) and 'results' in result:
            formatted_results = []
            for item in result['results']:
                title = item.get('title', 'No title')
                content = item.get('content', 'No content')
                url = item.get('url', '')
                formatted_results.append(f"Title: {title}\nContent: {content}\nURL: {url}")
            return "\n\n".join(formatted_results) if formatted_results else "No results found"
        else:
            return str(result)
    except Exception as e:
        return f"WEB_ERROR::{e}"

@tool
def rag_search_tool(query: str) -> str:
    """Top-K chunks from KB (empty string if none)"""
    try:
        retriever_instance = get_retriever()
        docs = retriever_instance.invoke(query, k=5)
        return "\n\n".join(d.page_content for d in docs) if docs else ""
    except Exception as e:
        return f"RAG_ERROR::{e}"

# --- Pydantic schemas for structured output ---
class RouteDecision(BaseModel):
    route: Literal["rag", "web", "answer", "end"]
    reply: str | None = Field(None, description="Filled only when route == 'end'")

class RagJudge(BaseModel):
    sufficient: bool = Field(..., description="True if retrieved information is sufficient to answer the user's question, False otherwise.")

# --- LLM instances with structured output where needed ---
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

router_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0).with_structured_output(RouteDecision)
judge_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0).with_structured_output(RagJudge)
answer_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)

# --- Shared state type ---
class AgentState(TypedDict, total=False):
    messages: List[BaseMessage]
    route: Literal["rag", "web", "answer", "end"]
    rag: str
    web: str
    web_search_enabled: bool

# --- Node 1: router ---
def router_node(state: AgentState, config: RunnableConfig) -> AgentState:
    print("\n--- Entering router_node ---")
    query = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    web_search_enabled = config.get("configurable", {}).get("web_search_enabled", True)
    print(f"Router received web search info : {web_search_enabled}")

    system_prompt = (
        "You are an intelligent routing agent designed to direct user queries to the most appropriate tool."
        " Prioritize using the internal knowledge base (RAG) for factual information that is likely "
        "to be contained within pre-uploaded documents."
    )

    if web_search_enabled:
        system_prompt += (
            " You can use web search for queries that require current, real-time, or broad general knowledge."
        )
    else:
        system_prompt += (
            " Web search is disabled. Do not choose the 'web' route."
        )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query)
    ]

    result: RouteDecision = router_llm.invoke(messages)
    initial_router_decision = result.route
    router_override_reason = None

    if not web_search_enabled and result.route == "web":
        result.route = "rag"
        router_override_reason = "Web search disabled; redirected to RAG."
        print(f"Router decision overridden: changed from 'web' to 'rag'.")

    print(f"Router final decision: {result.route}, Reply (if 'end'): {result.reply}")

    out = {
        "messages": state["messages"],
        "route": result.route,
        "web_search_enabled": web_search_enabled
    }
    if router_override_reason:
        out["initial_router_decision"] = initial_router_decision
        out["router_override_reason"] = router_override_reason
    if result.route == "end":
        out["messages"] = state["messages"] + [AIMessage(content=result.reply or "Hello!")]
    print("--- Exiting router_node ---")
    return out

# --- Node 2: RAG lookup ---
def rag_node(state: AgentState, config: RunnableConfig) -> AgentState:
    print("\n--- Entering rag_node ---")
    query = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    web_search_enabled = config.get("configurable", {}).get("web_search_enabled", True)
    print(f"RAG query: {query}")

    chunks = rag_search_tool.invoke(query)
    if chunks.startswith("RAG_ERROR::"):
        print(f"RAG Error: {chunks}. Checking web search enabled status.")
        next_route = "web" if web_search_enabled else "answer"
        return {**state, "rag": "", "route": next_route}

    if not chunks:
        print("No RAG chunks retrieved.")
        next_route = "web" if web_search_enabled else "answer"
        return {**state, "rag": "", "route": next_route, "web_search_enabled": web_search_enabled}

    print(f"Retrieved RAG chunks (first 500 chars): {chunks[:500]}...")

    judge_system = (
        "You are a judge evaluating if the retrieved information is sufficient and relevant to fully "
        "answer the user's question. Return EXACT JSON matching: {\"sufficient\": true} or {\"sufficient\": false}."
    )
    judge_messages = [
        SystemMessage(content=judge_system),
        HumanMessage(content=f"Question: {query}\nRetrieved info: {chunks}\nIs this sufficient?")
    ]

    try:
        verdict: RagJudge = judge_llm.invoke(judge_messages)
        print(f"RAG Judge verdict: {verdict.sufficient}")
    except Exception as e:
        print("Error calling judge_llm:", repr(e))
        try:
            print("Exception details:", getattr(e, "response", str(e)))
        except Exception:
            pass
        return {**state, "rag": chunks, "route": "web" if web_search_enabled else "answer", "web_search_enabled": web_search_enabled}

    next_route = "answer" if verdict.sufficient else ("web" if web_search_enabled else "answer")
    if not verdict.sufficient:
        print(f"RAG not sufficient. Next route: {next_route}")

    print("--- Exiting rag_node ---")
    return {**state, "rag": chunks, "route": next_route, "web_search_enabled": web_search_enabled}

# --- Node 3: web search ---
def web_node(state: AgentState, config: RunnableConfig) -> AgentState:
    print("\n--- Entering web_node ---")
    query = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    web_search_enabled = config.get("configurable", {}).get("web_search_enabled", True)
    print(f"Web search enabled: {web_search_enabled}")
    if not web_search_enabled:
        return {**state, "web": "Web search was disabled by the user.", "route": "answer"}

    snippets = web_search_tool.invoke(query)
    if snippets.startswith("WEB_ERROR::"):
        print(f"Web Error: {snippets}. Proceeding to answer with limited info.")
        return {**state, "web": "", "route": "answer"}

    print(f"Web snippets retrieved: {snippets[:200]}...")
    print("--- Exiting web_node ---")
    return {**state, "web": snippets, "route": "answer"}

# --- Node 4: final answer ---
def answer_node(state: AgentState) -> AgentState:
    print("\n--- Entering answer_node ---")
    user_q = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")

    ctx_parts = []
    if state.get("rag"):
        ctx_parts.append("Knowledge Base Information:\n" + state["rag"])
    if state.get("web") and not state["web"].startswith("Web search was disabled"):
        ctx_parts.append("Web Search Results:\n" + state["web"])
    context = "\n\n".join(ctx_parts) or "No external context available."

    prompt = f"""Please answer the user's question using the provided context.

Question: {user_q}

Context:
{context}

Provide a helpful, accurate, and concise response based on the available information."""

    print(f"Prompt sent to answer_llm: {prompt[:500]}...")
    ans = answer_llm.invoke([HumanMessage(content=prompt)]).content
    print(f"Final answer generated: {ans[:200]}...")
    print("--- Exiting answer_node ---")
    return {**state, "messages": state["messages"] + [AIMessage(content=ans)]}

# --- Routing helpers ---
def from_router(st: AgentState) -> Literal["rag", "web", "answer", "end"]:
    return st["route"]

def after_rag(st: AgentState) -> Literal["answer", "web"]:
    return st["route"]

def after_web(_) -> Literal["answer"]:
    return "answer"

# --- Build graph ---
def build_agent():
    g = StateGraph(AgentState)
    g.add_node("router", router_node)
    g.add_node("rag_lookup", rag_node)
    g.add_node("web_search", web_node)
    g.add_node("answer", answer_node)

    g.set_entry_point("router")

    g.add_conditional_edges(
        "router",
        from_router,
        {"rag": "rag_lookup", "web": "web_search", "answer": "answer", "end": END}
    )

    g.add_conditional_edges(
        "rag_lookup",
        after_rag,
        {"answer": "answer", "web": "web_search"}
    )

    g.add_edge("web_search", "answer")
    g.add_edge("answer", END)

    agent = g.compile(checkpointer=MemorySaver())
    return agent

rag_agent = build_agent()
