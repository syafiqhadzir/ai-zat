"""AI-Zat: Deep Research Agent with Voice Interaction."""
import io
import logging
import os
import tempfile
from pathlib import Path
import warnings

import streamlit as st
from langchain_core.messages import HumanMessage
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

# --- VOICE FUNCTIONS ---

def transcribe_audio(audio_bytes: bytes) -> str:
    """Transcribe audio using Groq Whisper (Bleeding Edge Speed)."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "[Error: GROQ_API_KEY not set for transcription]"
    
    try:
        client = Groq(api_key=api_key)
        
        # Write to temp file (Groq API requires file path)
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
        tts = gTTS(text=text[:500], lang='en')  # Limit for speed
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer.read()
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return b""

# --- MAIN APP ---

def main() -> None:
    st.title("🏺 AI-Zat: Voice-Enabled Research Agent")

    # --- SIDEBAR ---
    with st.sidebar:
        if LOGO_PATH.exists():
            st.logo(str(LOGO_PATH), link="https://stg-jam.vercel.app/")
            st.image(str(LOGO_PATH), width=150)
        
        st.header("🎛️ Configuration")
        
        model_options = ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"]
        selected_model = st.selectbox("Select LLM", model_options, index=0)
        
        enable_voice_output = st.toggle("🔊 Enable Voice Output", value=False)
        
        # Knowledge Base Control
        st.subheader("📚 Knowledge Base")
        if st.button("Re-Index Journal", help="Chunks and embeds the PDF into Vector DB", type="primary"):
            with st.spinner("Processing PDF into Vector Database..."):
                try:
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

        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

        st.markdown("---")
        st.caption("AFL-3.0: 2025 © Ikatan Ahli Arkeologi Malaysia")

    # --- CHAT UI ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display History
    for msg in st.session_state.messages:
        if isinstance(msg, dict):
            role = msg["role"]
            content = msg["content"]
        else:
            role = "user" if msg.type == "human" else "assistant"
            content = msg.content
            
        with st.chat_message(role):
            st.markdown(content)

    # --- INPUT: Voice OR Text ---
    col1, col2 = st.columns([3, 1])
    
    with col1:
        text_input = st.chat_input("Ask the Research Agent...")
    
    with col2:
        audio_input = st.audio_input("🎤 Voice", label_visibility="collapsed")
    
    user_request = None
    
    # Priority: Text > Voice
    if text_input:
        user_request = text_input
    elif audio_input:
        with st.spinner("Transcribing voice..."):
            user_request = transcribe_audio(audio_input.read())
            if user_request.startswith("[Error") or user_request.startswith("[Transcription"):
                st.error(user_request)
                user_request = None
            else:
                st.info(f"🎤 You said: *{user_request}*")

    if user_request:
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
                        if "reasoner" in chunk:
                            msg = chunk["reasoner"]["messages"][-1]
                            if msg.content:
                                final_answer = msg.content
                                response_placeholder.markdown(final_answer)
                            
                except Exception as e:
                    st.error(f"Agent Error: {e}")
                
                if final_answer:
                    st.session_state.messages.append(msg)
                    
                    # Voice Output
                    if enable_voice_output:
                        with st.spinner("Generating audio..."):
                            audio_bytes = text_to_speech(final_answer)
                            if audio_bytes:
                                st.audio(audio_bytes, format="audio/mp3")
                else:
                    if not st.session_state.get("error"):
                        response_placeholder.markdown("Searching tools... (Check terminal for details)")

if __name__ == "__main__":
    main()