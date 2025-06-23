import os
import pypdf 
import streamlit as st
from agent import initialize_app
from langchain_groq.chat_models import ChatGroq

st.title("Welcome to AI-Zat")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []
    
with st.sidebar:
    st.logo("ikatan_logo.png", link="https://stg-jam.vercel.app/")
    st.image("ikatan_logo.png", width=200)
    st.header("AI-Zat (AI Agent for Jurnal Arkeologi Malaysia)")
    st.subheader("Configuration", divider="gray")

    model_options = [
        "llama-3.3-70b-versatile"
    ]
    
    # Initialize session state for the model if it doesn't exist
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "llama-3.3-70b-versatile"
            
    selected_model = st.selectbox("Select Model", model_options, key="selected_model", index=model_options.index(st.session_state.selected_model))
    
    # Update the model in session state when changed
    if selected_model != st.session_state.selected_model:
        st.session_state.selected_model = selected_model
        # Create a new LLM instance when model changes
        st.session_state.llm = ChatGroq(model=selected_model, temperature=0.0)
    
    # Initialize LLM if it doesn't exist
    if "llm" not in st.session_state:
        st.session_state.llm = ChatGroq(model=selected_model, temperature=0.0)
    
    reset_button = st.button("🔄 Reset Conversation", key="reset_button")
    if reset_button:
        st.session_state.messages = []
        st.rerun()

    st.text("AFL-3.0: 2025 © Ikatan Ahli Arkeologi Malaysia.")
    

# Initialize the app
app = initialize_app(model_name=st.session_state.selected_model)

# Add an expander for examples and help
with st.expander("👋 Hi! Welcome to AI-Zat, your personal academic assistant!"):
    st.markdown("""
    AI-Zat is designed to help you with your academic needs, especially in the context of Jurnal Arkeologi Malaysia. 
    You can ask questions about the journal content, request summaries, or seek assistance with specific topics related to archaeology.
    """)

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get journal content
journal_content = ""
if os.path.exists("./journal.pdf"):
    try:
        # Use binary mode for PDF
        with open("./journal.pdf", "rb") as f:
            pdf_reader = pypdf.PdfReader(f)
            for page in pdf_reader.pages:
                journal_content += page.extract_text()
    except Exception as e:
        st.error(f"Error reading journal file: {e}")
                         
user_request = st.chat_input("Enter your request (e.g., 'Help me find something in Jurnal Arkeologi Malaysia'):")

if user_request:
    # Add user request to chat history
    st.session_state.messages.append({"role": "user", "content": user_request})
    with st.chat_message("user"):
        st.markdown(user_request)

    # Process with AI and get response
    with st.chat_message("assistant"):
        with st.spinner("Generating test cases..."):
            # Include settings in the inputs
            inputs = {
                "user_request": user_request,
                "journal_content": journal_content
            }
            
            # Stream the results
            response_placeholder = st.empty()
            total_answer = ""
            
            progress_bar = st.progress(0)
            progress_text = st.empty()
            
            # Track stages of processing
            stages = ["Analyzing journal content"]
            current_stage = 0
            
            for output in app.stream(inputs):
                for node_name, state in output.items():
                    # Update the progress based on the current node
                    if node_name == "summary_node":
                        current_stage = 0
                    
                    progress_value = (current_stage + 1) / len(stages)
                    progress_bar.progress(progress_value)
                    progress_text.text(f"Step {current_stage+1}/{len(stages)}: {stages[current_stage]}")
                    
                    
                    if 'answer' in state:
                        journal_summary = state.get('journal_elab_summary', '')
                        answer = state['answer']

                        total_answer = ""
                        if journal_summary:
                            total_answer += f"**Requirements Summary:**\n{journal_summary}\n\n"
                        total_answer += answer

                        response_placeholder.markdown(total_answer)

            # Clear progress indicators when done
            progress_bar.empty()
            progress_text.empty()
            
            # Add final response to chat history
            st.session_state.messages.append({"role": "assistant", "content": total_answer})