FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY switchboard ./switchboard
COPY config ./config
RUN addgroup --system --gid 10001 switchboard \
    && adduser --system --uid 10001 --ingroup switchboard switchboard \
    && mkdir -p /data \
    && python -m pip install --no-cache-dir -e . \
    && chown -R switchboard:switchboard /app /data

EXPOSE 8000
USER switchboard
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()"
CMD ["uvicorn", "switchboard.app.main:app", "--host", "127.0.0.1", "--port", "8000"]
