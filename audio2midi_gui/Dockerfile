# Minimal Dockerfile for CPU-only batch processing (no GUI)
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Example batch command
# python -c "from backend.separation import separate; separate('input.wav', 'output/')"

CMD ["python", "--version"]
