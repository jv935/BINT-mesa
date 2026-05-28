FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agents.py model.py profiles.py app.py ./
COPY *.ipynb ./

RUN mkdir -p results

# 8765 — Solara dashboard
# 8888 — JupyterLab
EXPOSE 8765 8888
