# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Install uv for dependency management
RUN pip install uv

# Copy project files
COPY . /app

# Install dependencies via uv (system-wide)
RUN uv pip install --system -r requirements.txt

# Expose the FastAPI poxrt
EXPOSE 8000

# Run the FastAPI app with Uvicorn
CMD ["python", "main.py"]