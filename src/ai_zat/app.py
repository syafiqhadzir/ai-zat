"""AI-Zat: Production-Grade Voice Agent with Streaming & Memory."""
import io
import logging
import os
import tempfile
import uuid
from pathlib import Path
import warnings

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from groq import Groq
from gtts import gTTS

from .rag import rag_manager

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
    page_title="AI-Zat | Archaeological Voice Agent",
    page_icon="🏺",
    layout="wide",
)

# Constants
LOGO_PATH = Path(__file__).parent.parent.parent / "ikatan_logo.png"
PDF_PATH = Path(__file__).parent.parent.parent / "journal.pdf"

# --- SESSION STATE INITIALIZATION ---
def init_session_state() -> None:
    """Initialize all session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())

# --- VOICE FUNCTIONS ---

def transcribe_audio(audio_bytes: bytes) -> str:
    """Transcribe audio using Groq Whisper."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "[Error: GROQ_API_KEY not set]"
    
    try:
        client = Groq(api_key=api_key)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        
        with open(tmp_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=(tmp_path, audio_file.read()),
                model="whisper-large-v3-turbo",
                response_format="text"
            )
        
        os.unlink(tmp_path)
        return transcription
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return f"[Transcription Error: {e}]"

def text_to_speech(text: str) -> bytes:
    """Convert text to speech using gTTS."""
    try:
        tts = gTTS(text=text[:500], lang='en')
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer.read()
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return b""

# --- MAIN APP ---

def main() -> None:
    init_session_state()
    
    st.title("🏺 AI-Zat: Production Research Agent")

    # --- SIDEBAR ---
    with st.sidebar:
        if LOGO_PATH.exists():
            st.logo(str(LOGO_PATH), link="https://stg-jam.vercel.app/")
            st.image(str(LOGO_PATH), width=150)
        
        st.header("🎛️ Configuration")
        
        model_options = ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"]
        selected_model = st.selectbox("Select LLM", model_options, index=0)
        
        enable_voice_output = st.toggle("🔊 Voice Output", value=False)
        enable_memory = st.toggle("🧠 Memory Persistence", value=True, help="Remember conversations across sessions")
        
        # Knowledge Base Control
        st.subheader("📚 Knowledge Base")
        if st.button("Re-Index Journal", help="Chunks and embeds the PDF into Vector DB", type="primary"):
            with st.spinner("Processing PDF into Vector Database..."):
                try:
                    import pypdf
                    from langchain_core.documents import Document
                    
                    if PDF_PATH.exists():
                        docs = []
                        with open(PDF_PATH, "rb") as f:
                            pdf = pypdf.PdfReader(f)
                            for i, page in enumerate(pdf.pages):
                                page_text = page.extract_text()
                                if page_text:
                                    docs.append(Document(
                                        page_content=page_text,
                                        metadata={"source": "journal.pdf", "page": i + 1}
                                    ))
                        
                        rag_manager.clear_database()
                        rag_manager.ingest_documents(docs)
                        st.success(f"Indexed {len(docs)} pages! ✅")
                    else:
                        st.error("No journal.pdf found!")
                except Exception as e:
                    st.error(f"Indexing Failed: {e}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat"):
                st.session_state.messages = []
                st.session_state.thread_id = str(uuid.uuid4())
                st.rerun()
        with col2:
            if st.button("🔄 New Thread"):
                st.session_state.thread_id = str(uuid.uuid4())
                st.session_state.messages = []
                st.rerun()

        st.caption(f"Thread: `{st.session_state.thread_id[:8]}...`")
        st.markdown("---")
        st.caption("AFL-3.0: 2025 © Ikatan Ahli Arkeologi Malaysia")

    # --- CHAT UI ---
    for msg in st.session_state.messages:
        if isinstance(msg, dict):
            role = msg["role"]
            content = msg["content"]
        elif hasattr(msg, 'type'):
            role = "user" if msg.type == "human" else "assistant"
            content = msg.content
        else:
            continue
            
        with st.chat_message(role):
            st.markdown(content)

    # --- INPUT ---
    col1, col2 = st.columns([3, 1])
    
    with col1:
        text_input = st.chat_input("Ask the Research Agent...")
    
    with col2:
        audio_input = st.audio_input("🎤", label_visibility="collapsed")
    
    user_request = None
    
    if text_input:
        user_request = text_input
    elif audio_input:
        with st.spinner("Transcribing..."):
            user_request = transcribe_audio(audio_input.read())
            if user_request.startswith("[Error") or user_request.startswith("[Transcription"):
                st.error(user_request)
                user_request = None
            else:
                st.info(f"🎤 *{user_request}*")

    if user_request:
        user_msg = HumanMessage(content=user_request)
        st.session_state.messages.append(user_msg)
        
        with st.chat_message("user"):
            st.markdown(user_request)

        # Initialize Agent
        app = initialize_agent(selected_model, with_memory=enable_memory)
        
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            # Config for memory thread
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            inputs = {"messages": st.session_state.messages}
            
            with st.spinner("Thinking..."):
                try:
                    # Stream the response
                    for chunk in app.stream(inputs, config=config, stream_mode="values"):
                        if "messages" in chunk:
                            last_msg = chunk["messages"][-1]
                            if hasattr(last_msg, 'content') and last_msg.content:
                                if not hasattr(last_msg, 'tool_calls') or not last_msg.tool_calls:
                                    full_response = last_msg.content
                                    response_placeholder.markdown(full_response + "▌")
                    
                    # Final display
                    response_placeholder.markdown(full_response)
                    
                except Exception as e:
                    st.error(f"Agent Error: {e}")
                    full_response = f"Error: {e}"
                
                if full_response:
                    st.session_state.messages.append(AIMessage(content=full_response))
                    
                    if enable_voice_output:
                        with st.spinner("Generating audio..."):
                            audio_bytes = text_to_speech(full_response)
                            if audio_bytes:
                                st.audio(audio_bytes, format="audio/mp3")

if __name__ == "__main__":
    main()