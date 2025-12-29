# AI-Zat: Archaeological Assistant

[![CI](https://github.com/syafiqhadzir/ai-zat/actions/workflows/ci.yml/badge.svg)](https://github.com/syafiqhadzir/ai-zat/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.13-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.52.2-FF4B4B.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-AFL--3.0-green.svg)](LICENSE)
[![Code Style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**AI-Zat** is an advanced AI agent designed to assist researchers and enthusiasts with the *Jurnal Arkeologi Malaysia*. Built with bleeding-edge technology, it leverages Large Language Models (LLMs) to analyze, summarize, and answer questions based on archaeological texts.

![Logo](ikatan_logo.png)

## 🚀 Features

- **Context-Aware Analysis**: Uses Retrieval Augmented Generation (RAG) concepts to answer questions based on specific journal content.
- **Bleeding Edge Tech Stack**: Built on LangGraph v1.0, Streamlit 1.52, and Groq's high-speed inference.
- **Academic Focus**: Tailored prompts for archaeological context.
- **Modern Architecture**: Clean `src` layout, typed Python code, and robust error handling.

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Agent Orchestration**: LangGraph, LangChain
- **LLM/Inference**: Groq (Llama 3, Mixtral)
- **Search/Tools**: Tavily (optional integration)
- **PDF Processing**: pypdf

## ⚡ Quick Start

### Prerequisites

- Python 3.11+ (3.13 recommended)
- [Groq API Key](https://console.groq.com/)

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/syafiqhadzir/ai-zat.git
   cd ai-zat
   ```

2. **Set up environment**
   ```bash
   cp .env.example .env
   # Edit .env and add your GROQ_API_KEY
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   # OR for dev tools as well:
   pip install -e ".[dev]"
   ```

4. **Run the application**
   ```bash
   streamlit run src/ai_zat/app.py
   ```

### Docker

```bash
docker-compose up --build
```

### VS Code

Open the project in VS Code. If you have the Dev Containers extension installed, click "Reopen in Container" for a fully configured environment.

## 🧪 Testing

```bash
pytest
```

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

Academic Free License v3.0 (AFL-3.0). See [LICENSE](LICENSE).

---
*2025 © Ikatan Ahli Arkeologi Malaysia*