# servers/Dockerfile
# Backend Flask server image. Built with build context "." (the project
# root) so it can share the single root-level requirements.txt with every
# other Python component (see docker-compose.yml: `context: .`).

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY servers/ /app/servers/
COPY common/ /app/common/

WORKDIR /app/servers

EXPOSE 5000

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health', timeout=2)" || exit 1

CMD ["python", "app.py"]
