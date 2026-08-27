# Hinglish ASR — HF Space Docker image
# Free CPU basic space: 2 vCPU, 16 GB RAM.
# Model is downloaded from the HF Hub at boot (fast on HF's internal network),
# which keeps the image small and the build quick.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY backend /app/backend

# faster-whisper downloads "small" from the Hub on first load and caches it.
ENV WHISPER_MODEL=small
ENV WHISPER_COMPUTE=int8
ENV PYTHONUNBUFFERED=1
# keep the model cache warm across Space restarts
ENV HF_HOME=/data/hf_cache

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--app-dir", "/app/backend", "--host", "0.0.0.0", "--port", "7860"]
