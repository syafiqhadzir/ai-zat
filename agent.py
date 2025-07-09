import streamlit as st
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langchain_groq.chat_models import ChatGroq
from dotenv import load_dotenv 

load_dotenv()

## Define the GraphState 
class GraphState(TypedDict):
    user_request: str
    journal_content: str
    answer: str

## To generate Summary of the journal content
def generate_summary_node_function(state: GraphState) -> GraphState:
    journal_content = state.get("journal_content", "")
    if "llm" not in st.session_state:
        raise RuntimeError("LLM not initialized. Please call initialize_app first.")

    prompt = (
        "You are an expert archeologist.\n"
        "You should response to the user by referring to the given 'academic journal'."
        f"Journal Content: {journal_content}\n"
    )
    
    try:
        response = st.session_state.llm.invoke(prompt)
        summary = response.content
    except Exception as e:
        summary = f"Error generating answer: {str(e)}"
        print("Exception 300\n", e)

    state['answer'] = summary
    return state

def generate_answer(user_request, llm):

    prompt = (
        "Based on the summary, look at the original user request to make sure that you are answering the request properly!\n" +
        f"User Request: {user_request}\n"
    )
    
    prompt += "\nAnswer:"
    
    try:
        response = llm.invoke(prompt)
    except Exception as e:
        response = f"Error generating answer: {str(e)}"
        
    return response.content

## Build the LangGraph pipeline
def build_workflow():
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("summary_node", generate_summary_node_function)
    
    # Set entry point and add edges
    workflow.set_entry_point("summary_node")
    
    return workflow

## The initialize_app function
def initialize_app(model_name: str):
    """
    Initialize the app with the given model name, avoiding redundant initialization.
    """
    # Check if the LLM is already initialized
    if "selected_model" in st.session_state and st.session_state.selected_model == model_name:
        return build_workflow().compile()  # Return the compiled workflow

    # Initialize the LLM for the first time or switch models
    st.session_state.llm = ChatGroq(model=model_name, temperature=1.0)
    st.session_state.selected_model = model_name
    print(f"Using model: {model_name}")
    return build_workflow().compile()