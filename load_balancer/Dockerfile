# load_balancer/Dockerfile
# One image shared by all four balancer services (round_robin,
# least_connections, aco_balancer, dqn_balancer) - docker-compose.yml
# selects which script to run via the `command:` override per service,
# so there is exactly one image to build and keep in sync.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY requirements.txt /app/requirements.txt
# CPU-only PyTorch wheel first (small, no CUDA runtime), then the rest
# of the shared requirements - see requirements.txt for why.
RUN pip install --no-cache-dir torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /app/requirements.txt

COPY common/ /app/common/
COPY dqn/ /app/dqn/
COPY load_balancer/ /app/load_balancer/

EXPOSE 8000 8001 8002 8003 9000 9001 9002 9003

CMD ["python", "load_balancer/round_robin.py"]
