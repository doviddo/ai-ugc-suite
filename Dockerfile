FROM python:3.12-slim

# Install FFmpeg and ffprobe
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create working directories
RUN mkdir -p temp output

EXPOSE 5000

CMD ["python", "app.py"]
