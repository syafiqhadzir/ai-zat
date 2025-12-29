"""Agent module for AI-Zat with Memory Persistence and Streaming."""
import logging
import os
from typing import Any, Dict, cast

import streamlit as st
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq.chat_models import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from .rag import rag_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- MEMORY PERSISTENCE ---
DB_PATH = ".memory.db"

def get_checkpointer() -> SqliteSaver:
    """Get or create the SQLite checkpointer for memory persistence."""
    return SqliteSaver.from_conn_string(DB_PATH)

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
    
    # Format retrieval with page numbers
    return "\n\n".join([
        f"[Source: Page {doc.metadata.get('page', 'Unknown')}] {doc.page_content}" 
        for doc in docs
    ])

@tool
def search_web(query: str) -> str:
    """
    Search the general web for information NOT found in the journal.
    Use this only if 'lookup_journal' fails or for modern context.
    """
    if not os.getenv("TAVILY_API_KEY"):
        return "Web search is disabled (TAVILY_API_KEY missing)."
        
    try:
        search_tool = TavilySearchResults(max_results=3)
        return str(search_tool.invoke(query))
    except Exception as e:
        return f"Web search failed: {e}"

# --- STATE ---

class GraphState(Dict[str, Any]):
    """Deep Research Agent State with message history."""
    messages: list[BaseMessage]

# --- NODES ---

SYSTEM_PROMPT = """You are an expert archaeologist assistant for Jurnal Arkeologi Malaysia.

INSTRUCTIONS:
1. ALWAYS use the 'lookup_journal' tool first to find relevant information.
2. If the journal doesn't have the answer, use 'search_web' for recent context.
3. Cite your sources clearly (e.g., "According to Chunk 2...").
4. Maintain an academic yet accessible tone.
5. If you cannot find the answer, say so honestly."""

def reasoner_node(state: GraphState) -> Dict[str, Any]:
    """The 'Brain': Decides whether to call a tool or answer."""
    messages = state["messages"]
    
    if "llm" not in st.session_state:
        return {"messages": [SystemMessage(content="LLM not initialized.")]}

    llm = cast(ChatGroq, st.session_state.llm)
    
    # Prepend system message if not present
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools([lookup_journal, search_web])
    
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# --- WORKFLOW ---

def build_workflow(with_memory: bool = True) -> CompiledStateGraph:
    """Build the Agentic RAG Workflow with optional memory."""
    workflow = StateGraph(GraphState)
    
    # Tools Node
    tools = [lookup_journal, search_web]
    tool_node = ToolNode(tools)
    
    # Add Nodes
    workflow.add_node("reasoner", reasoner_node)
    workflow.add_node("tools", tool_node)
    
    # Edges
    workflow.set_entry_point("reasoner")
    workflow.add_conditional_edges("reasoner", tools_condition)
    workflow.add_edge("tools", "reasoner")
    
    # Compile with or without memory
    if with_memory:
        checkpointer = get_checkpointer()
        return workflow.compile(checkpointer=checkpointer)
    
    return workflow.compile()

def initialize_agent(model_name: str, with_memory: bool = True) -> CompiledStateGraph:
    """Initialize agent with model and optional memory persistence."""
    if "llm" not in st.session_state or st.session_state.get("selected_model") != model_name:
        # Enable LangSmith tracing if configured
        if os.getenv("LANGCHAIN_TRACING_V2"):
            logger.info("LangSmith tracing enabled.")
        
        st.session_state.llm = ChatGroq(model=model_name, temperature=0.5)
        st.session_state.selected_model = model_name
        
    return build_workflow(with_memory=with_memory)
