"""Agent module for AI-Zat with Memory Persistence and Streaming."""
import logging
import os
from typing import Any, Literal, cast

import streamlit as st
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool
from langchain_groq.chat_models import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
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
    Use this tool FIRST for any questions about archaeology, history,
    or the journal content.
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

class GraphState(dict[str, Any]):
    """Deep Research Agent State with message history and iteration tracking."""
    messages: list[BaseMessage]
    retry_count: int

# --- GRADERS (Structured Output) ---

class GradeRetrieval(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="Relevance score 'yes' or 'no'")

class GradeHallucination(BaseModel):
    """Binary score for hallucination check."""
    binary_score: str = Field(description="Is answer grounded in facts? 'yes'/'no'")

# --- NODES ---

SYSTEM_PROMPT = """You are an expert archaeologist assistant for \
Jurnal Arkeologi Malaysia.

INSTRUCTIONS:
1. ALWAYS use the 'lookup_journal' tool first to find relevant information.
2. If the journal doesn't have the answer, use 'search_web' for recent context.
3. Cite your sources clearly (e.g., "According to Chunk 2...").
4. Maintain an academic yet accessible tone.
5. If you cannot find the answer, say so honestly."""

def reasoner_node(state: GraphState) -> dict[str, Any]:
    """The 'Brain': Decides whether to call a tool or answer."""
    messages = state["messages"]
    retry_count = state.get("retry_count", 0)
    
    if "llm" not in st.session_state:
        return {"messages": [SystemMessage(content="LLM not initialized.")]}

    llm = cast(ChatGroq, st.session_state.llm)
    
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT), *messages]
    
    llm_with_tools = llm.bind_tools([lookup_journal, search_web])
    response = llm_with_tools.invoke(messages)
    
    return {"messages": [response], "retry_count": retry_count}

def grade_retrieval_node(state: GraphState) -> Literal["generate", "rewrite"]:
    """Determines if retrieval is relevant or needs a rewrite."""
    last_message = state["messages"][-1]
    
    # If not a tool output, just generate
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return "generate"

    # For simplicity in this demo, we grade the 'lookup_journal' results
    # In a full Level 5 system, we'd invoke the LLM to score each chunk
    # Here we skip if results are populated or force rewrite if empty
    if "No relevant information" in last_message.content:
        return "rewrite"
    
    return "generate"

def query_rewriter_node(state: GraphState) -> dict[str, Any]:
    """Rewrites the query for better RAG results."""
    messages = state["messages"]
    retry_count = state.get("retry_count", 0) + 1
    
    llm = cast(ChatGroq, st.session_state.llm)
    
    # Simple rewriter prompt
    prompt = (
        "The previous retrieval failed. Rewrite the original user question to "
        "be more searchable in an archaeology database. "
        f"Original: {messages[-2].content}"
    )
    response = llm.invoke(prompt)
    
    # Replace the last human message or append a new one
    return {
        "messages": [HumanMessage(content=f"Retry {retry_count}: {response.content}")],
        "retry_count": retry_count
    }

# --- WORKFLOW ---

def build_workflow(with_memory: bool = True) -> CompiledStateGraph:
    """Build the Self-Correcting Agentic RAG Workflow."""
    workflow = StateGraph(GraphState)
    
    # Nodes
    workflow.add_node("reasoner", reasoner_node)
    workflow.add_node("tools", ToolNode([lookup_journal, search_web]))
    workflow.add_node("rewriter", query_rewriter_node)
    
    # Entry
    workflow.set_entry_point("reasoner")
    
    # Flow
    workflow.add_conditional_edges(
        "reasoner",
        tools_condition,
        {
            "tools": "tools",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "tools",
        grade_retrieval_node,
        {
            "generate": "reasoner",
            "rewrite": "rewriter"
        }
    )
    
    workflow.add_edge("rewriter", "reasoner")
    
    if with_memory:
        return workflow.compile(checkpointer=get_checkpointer())
    
    return workflow.compile()

def initialize_agent(model_name: str, with_memory: bool = True) -> CompiledStateGraph:
    """Initialize agent with model and optional memory persistence."""
    current_model = st.session_state.get("selected_model")
    if "llm" not in st.session_state or current_model != model_name:
        # Enable LangSmith tracing if configured
        if os.getenv("LANGCHAIN_TRACING_V2"):
            logger.info("LangSmith tracing enabled.")
        
        st.session_state.llm = ChatGroq(model=model_name, temperature=0.5)
        st.session_state.selected_model = model_name
        
    return build_workflow(with_memory=with_memory)
