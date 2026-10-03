# mycamu/Dockerfile - TCE MyCamu portal (Flask, port 8080)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MYCAMU_DATA_DIR=/app/data \
    MYCAMU_DB_PATH=/app/data/mycamu.db

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app
RUN mkdir -p /app/data

EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=4s --start-period=10s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/api/auth/demo-accounts', timeout=3)" || exit 1

CMD ["python", "app.py"]
