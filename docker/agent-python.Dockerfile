# Sandbox image for "python" entries: Python with numpy, PyTorch (CPU) and ONNX Runtime, and the agent SDK.
# Build from the repository root: docker build -f docker/agent-python.Dockerfile -t aimoba-agent-python .
FROM python:3.12-slim
RUN pip install --no-cache-dir numpy onnxruntime \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY python/aimoba_agent.py /usr/local/lib/python3.12/site-packages/aimoba_agent.py
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
USER 65534:65534
WORKDIR /agent
