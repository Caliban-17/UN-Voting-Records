FROM python:3.12-slim

WORKDIR /app

# gcc/g++ for the compiled wheels (fastparquet, python-louvain);
# libcairo2 for the newsletter's SVG -> PNG chart export (cairosvg).
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libcairo2 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# No `playwright install`: the data refresh uses UN Digital Library's
# MARC-XML API over plain HTTP, so the image needs no browser.

COPY . .

RUN mkdir -p .cache data

# web_app.py listens on $PORT (default 5001).
ENV PYTHONUNBUFFERED=1 \
    PORT=5001
EXPOSE 5001

# python:*-slim ships no curl; probe /health with the stdlib instead.
# start-period covers the initial CSV/parquet load.
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://localhost:5001/health', timeout=8).status == 200 else 1)"

CMD ["python", "web_app.py"]
