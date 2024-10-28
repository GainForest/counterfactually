# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Use Python to handle the PORT environment variable
CMD python -c "import os; from subprocess import run; port = int(os.getenv('PORT', '8000')); run(['uvicorn', 'src.main:app', '--host', '0.0.0.0', '--port', str(port)])"