FROM python:3.10-slim

WORKDIR /app

# Install system dependencies including FFmpeg for audio processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Create directories for downloads and transcripts
RUN mkdir -p /app/downloads /app/transcripts

# Set environment variables
ENV OLLAMA_HOST=http://ollama:11434
ENV LMSTUDIO_HOST=http://lmstudio:1234
ENV DOWNLOAD_DIR=/app/downloads
ENV TRANSCRIPT_DIR=/app/transcripts

# Expose Streamlit port
EXPOSE 8501

# Set volume for persistent storage
VOLUME ["/app/downloads", "/app/transcripts"]

# Run the application
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]