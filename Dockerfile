FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install build deps for some packages (kept minimal)
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# Copy application code
COPY backend ./backend

WORKDIR /app/backend

# App listens on port 5001 (as in backend/app.py)
EXPOSE 5001

# Run the Flask app
CMD ["uv", "run", "gunicorn", "-w", "2", "-b", "0.0.0.0:5001", "app:app"]