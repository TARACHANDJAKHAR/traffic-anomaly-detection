FROM python:3.11-slim

# Install system dependencies required for OpenCV and FFmpeg
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app/ ./app/
COPY models/ ./models/
COPY yolov8n.pt .

# Create a non-root user and grant permissions
RUN useradd -m -u 1000 user
RUN chown -R user:user /app
USER user

# Expose port (Render overrides this with $PORT)
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run Gunicorn binding to Render's dynamic port
CMD gunicorn -w 1 --timeout 120 -b 0.0.0.0:$PORT app.app:app
