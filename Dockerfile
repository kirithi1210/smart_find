# Base Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Configure APT retries and install required system packages
RUN apt-get update -o Acquire::Retries=5 && \
    apt-get install -y --no-install-recommends --fix-missing \
    build-essential \
    libgl1 \
    libglib2.0-0t64 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements definition
COPY requirements.txt .

# Install CPU-only PyTorch first to save memory and space
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and assets
COPY . .

# Expose port for FastAPI
EXPOSE 8000

# Start application server with single worker to conserve RAM
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]