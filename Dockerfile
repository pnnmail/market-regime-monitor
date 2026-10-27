FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Encoding setup
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV TZ=America/New_York

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements.txt
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy ALL files from the current directory
COPY . .

# Check that the files were copied
RUN ls -la /app/
RUN ls -la /app/daily_strategy_check.py || echo "WARNING: daily_strategy_check.py not found!"

# Create the required directories
RUN mkdir -p /app/data/strategy
RUN mkdir -p /app/data
RUN mkdir -p /app/results

# Set write permissions
RUN chmod -R 777 /app/data
RUN chmod -R 777 /app/results

# Default command
CMD ["python", "runner.py"]