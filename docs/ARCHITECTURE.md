# Architecture

## Overview

AI-Zat is a RAG-light application that processes archaeological journal APIs (currently static PDF) and uses an LLM Agent to answer user queries.

## Components

### 1. Frontend (Streamlit)
- `src/ai_zat/app.py`
- Handles UI, Session State, and basic input validation.
- Caches PDF content.

### 2. Agent Layer (LangGraph)
- `src/ai_zat/agent.py`
- Defines the state machine for the conversation.
- **State**: `GraphState` (TypedDict) holds request, context, and answer.
- **Nodes**: `generate_response_node` constructs the specific prompt for the Journal.
- **Model**: Uses `ChatGroq` for high-speed inference.

### 3. Data Layer
- **Input**: `journal.pdf` (Raw text extraction via `pypdf`).
- **Storage**: In-memory (Session State).

## Data Flow

1. User inputs query in Streamlit.
2. `app.py` invokes `agent.stream()`.
3. Agent initializes `GraphState` with query + journal content.
4. `generate_response_node` calls Groq LLM with context-rich prompt.
5. Response is streamed back to `app.py`.
6. UI updates.
