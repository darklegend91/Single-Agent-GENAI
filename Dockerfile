FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY scripts ./scripts
COPY tests/fixtures ./tests/fixtures
RUN useradd --uid 10001 --create-home assistant && mkdir -p /app/output && chown -R assistant:assistant /app
USER assistant
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=4)"
CMD ["uvicorn","app.main:create_app","--factory","--host","0.0.0.0","--port","8000","--workers","1"]
