import logging
import os
from pathlib import Path
import warnings

import streamlit as st
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

from .rag import rag_manager  # Import RAG manager relative to package

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
load_dotenv()

# Universal import for agent
try:
    from .agent import initialize_agent
except ImportError:
    try:
        from agent import initialize_agent
    except ImportError:
        import sys
        sys.path.append(str(Path(__file__).parent))
        from agent import initialize_agent

# Page Config
st.set_page_config(
    page_title="AI-Zat | Archery Journal Assistant",
    page_icon="🏺",
    layout="wide",
)

# Constants
LOGO_PATH = Path(__file__).parent.parent.parent / "ikatan_logo.png"
PDF_PATH = Path(__file__).parent.parent.parent / "journal.pdf"

def main():
    st.title("🏺 AI-Zat: Deep Research Agent")

    # --- SIDEBAR ---
    with st.sidebar:
        if LOGO_PATH.exists():
            st.logo(str(LOGO_PATH), link="https://stg-jam.vercel.app/")
            st.image(str(LOGO_PATH), width=150)
        
        st.header("Deep Configuration")
        
        model_options = ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"]
        selected_model = st.selectbox("Select Model", model_options, index=0)
        
        # Knowledge Base Control
        st.subheader("Knowledge Base")
        if st.button("📚 Re-Index Journal (Vector DB)", help="Chunks and embeds the PDF"):
            with st.spinner("Processing PDF into Vector Database..."):
                try:
                    # Load text first (simple way for this demo)
                    import pypdf
                    text = ""
                    if PDF_PATH.exists():
                        with open(PDF_PATH, "rb") as f:
                            pdf = pypdf.PdfReader(f)
                            for page in pdf.pages:
                                text += page.extract_text() + "\n"
                        rag_manager.clear_database()
                        rag_manager.ingest_text(text)
                        st.success("Indexing Complete! ✅")
                    else:
                        st.error("No journal.pdf found!")
                except Exception as e:
                    st.error(f"Indexing Failed: {e}")

        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    # --- CHAT UI ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display History
    for msg in st.session_state.messages:
        # Check if it's a dict or LangChain message object
        if isinstance(msg, dict):
            role = msg["role"]
            content = msg["content"]
        else:
            role = "user" if msg.type == "human" else "assistant"
            content = msg.content
            
        with st.chat_message(role):
            st.markdown(content)

    # Input
    if user_request := st.chat_input("Ask the Research Agent..."):
        # Add User Message
        user_msg = HumanMessage(content=user_request)
        st.session_state.messages.append(user_msg)
        
        with st.chat_message("user"):
            st.markdown(user_request)

        # Agent Loop
        app = initialize_agent(selected_model)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking & Searching..."):
                response_placeholder = st.empty()
                
                inputs = {"messages": st.session_state.messages}
                final_answer = ""
                
                try:
                    for chunk in app.stream(inputs):
                        # The agent returns a dict with keys like 'reasoner' or 'tools'
                        # We are interested in the final output from 'reasoner' which is an AIMessage
                        if "reasoner" in chunk:
                            msg = chunk["reasoner"]["messages"][-1]
                            if msg.content: # Only show if it has text (not just tool calls)
                                final_answer = msg.content
                                response_placeholder.markdown(final_answer)
                            
                except Exception as e:
                    st.error(f"Agent Error: {e}")
                
                if final_answer:
                    st.session_state.messages.append(msg)
                else:
                    if not st.session_state.get("error"):
                        response_placeholder.markdown("Searching tools... (Check terminal for details)")

if __name__ == "__main__":
    main()