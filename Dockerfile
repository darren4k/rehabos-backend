# ---- Builder stage ----
FROM python:3.12-slim AS builder

WORKDIR /build

COPY pyproject.toml README.md ./
COPY rehab_os/ rehab_os/

RUN pip install --no-cache-dir --prefix=/install .

# ---- Runtime stage ----
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY rehab_os/ rehab_os/

# Create non-root user
RUN useradd -r -s /bin/false appuser && \
    mkdir -p /app/data/chroma && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

CMD ["uvicorn", "rehab_os.api.app:app", "--host", "0.0.0.0", "--port", "8080"]
