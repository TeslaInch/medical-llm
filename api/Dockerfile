# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Setup a non-root user as required by HuggingFace Spaces
RUN useradd -m -u 1000 user
USER user

# Set home to the user's home directory
ENV HOME=/home/user
ENV PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy requirements and install
COPY --chown=user requirements.txt $HOME/app/
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY --chown=user . $HOME/app/

# Expose port 7860 which is required by HF Spaces
EXPOSE 7860

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
