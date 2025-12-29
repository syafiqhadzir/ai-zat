"""Agent module for AI-Zat."""
import logging
import os
from typing import TypedDict, cast, Dict, Any

import streamlit as st
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq.chat_models import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from .rag import rag_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- TOOLS ---

@tool
def lookup_journal(query: str) -> str:
    """
    Search the Jurnal Arkeologi Malaysia for relevant information.
    Use this tool FIRST for any questions about archaeology, history, or the journal content.
    """
    docs = rag_manager.retrieve(query)
    if not docs:
        return "No relevant information found in the journal."
    
    # Format retrieval
    return "\n\n".join([f"[Source: Chunk {i}] {doc.page_content}" for i, doc in enumerate(docs)])

@tool
def search_web(query: str) -> str:
    """
    Search the general web for information NOT found in the journal.
    Use this only if 'lookup_journal' fails or for modern context.
    """
    # Check API key first
    if not os.getenv("TAVILY_API_KEY"):
        return "Web search is disabled (TAVILY_API_KEY missing)."
        
    try:
        search_tool = TavilySearchResults(max_results=3)
        return str(search_tool.invoke(query))
    except Exception as e:
        return f"Web search failed: {e}"

# --- STATE ---

class GraphState(TypedDict):
    """Deep Research Agent State."""
    messages: list[BaseMessage]

# --- NODES ---

def reasoner_node(state: GraphState) -> Dict[str, Any]:
    """The 'Brain': Decides whether to call a tool or answer."""
    messages = state["messages"]
    
    if "llm" not in st.session_state:
        return {"messages": [SystemMessage(content="LLM not initialized.")]}

    llm = cast(ChatGroq, st.session_state.llm)
    
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools([lookup_journal, search_web])
    
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# --- WORKFLOW ---

def build_workflow() -> CompiledStateGraph:
    """Build the Agentic RAG Workflow."""
    workflow = StateGraph(GraphState)
    
    # Tools Node (Prebuilt by LangGraph)
    tools = [lookup_journal, search_web]
    tool_node = ToolNode(tools)
    
    # Add Nodes
    workflow.add_node("reasoner", reasoner_node)
    workflow.add_node("tools", tool_node)
    
    # Edges
    workflow.set_entry_point("reasoner")
    
    # Conditional Edge: If tool call -> tools, else -> End
    workflow.add_conditional_edges(
        "reasoner",
        tools_condition,
    )
    
    workflow.add_edge("tools", "reasoner")
    
    return workflow.compile()

def initialize_agent(model_name: str) -> CompiledStateGraph:
    """Initialize agent with model."""
    if "llm" not in st.session_state or st.session_state.get("selected_model") != model_name:
        st.session_state.llm = ChatGroq(model=model_name, temperature=0.5)
        st.session_state.selected_model = model_name
        
    return build_workflow()
