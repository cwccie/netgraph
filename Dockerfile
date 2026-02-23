FROM python:3.14-slim

LABEL maintainer="Corey A. Wade <cwccie@gmail.com>"
LABEL description="NetGraph — GNN-based Network Topology Intelligence Platform"

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY pyproject.toml .
COPY src/ src/
COPY README.md .
COPY LICENSE .

RUN pip install --no-cache-dir ".[all]"

# Copy remaining files
COPY sample_data/ sample_data/
COPY tests/ tests/

# Expose dashboard port
EXPOSE 5000
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import netgraph; print('ok')" || exit 1

# Default: run the dashboard
ENTRYPOINT ["netgraph"]
CMD ["demo"]
