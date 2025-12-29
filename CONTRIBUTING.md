# Contributing to AI-Zat

Thank you for your interest in contributing to AI-Zat! We welcome contributions from the community to help improve this project.

## Development Process

1.  **Fork** the repository and create your branch from `main`.
2.  **Install** development dependencies: `pip install -e ".[dev]"`
3.  **Make changes** and ensure code quality:
    - Run linting: `ruff check .`
    - Run type checking: `mypy .`
    - Run tests: `pytest`
4.  **Commit** your changes using conventional commits (e.g., `feat: add new model`, `fix: PDF parsing error`).
5.  **Push** to your fork and submit a Pull Request.

## Coding Standards

- **Python Version**: 3.11+
- **Style**: We use [Ruff](https://github.com/astral-sh/ruff) for formatting and linting.
- **Typing**: All code must be fully typed (mypy strictness).
- **Documentation**: Add docstrings to all public functions and classes.

## Pull Request Process

- Ensure CI passes.
- Provide a clear description of changes.
- Update documentation if necessary.
