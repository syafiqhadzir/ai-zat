# Build stage
FROM python:3.13-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# Runtime stage
FROM python:3.13-slim

WORKDIR /app

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy installed packages
COPY --from=builder /install /usr/local

# Copy application code and assets
COPY src/ src/
COPY ikatan_logo.png .
COPY journal.pdf .
COPY .streamlit/ .streamlit/
# Create secrets file if not exists to prevent mount errors if expected
RUN mkdir -p .streamlit

# Set ownership
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8501

ENV PYTHONPATH="${PYTHONPATH}:/app/src"
ENV PYTHONUNBUFFERED=1

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "src/ai_zat/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
