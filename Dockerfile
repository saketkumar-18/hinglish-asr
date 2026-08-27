# Hinglish ASR — Render Docker image (free tier, 512 MB RAM)
# Uses the smaller `base` model to fit free-tier memory. Model is downloaded
# from the HF Hub at BUILD time (Render's build network is fast) and baked
# into the image, so there is no boot-time download or cold-start penalty.
FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY backend /app/backend
COPY scripts/download_model.py /app/download_model.py

# Bake the base model into the image at build time.
# (python:3.11-slim has no curl/wget, so download with Python.)
RUN python /app/download_model.py Systran/faster-whisper-base /app/models/faster-whisper-base

ENV WHISPER_MODEL=/app/models/faster-whisper-base
ENV WHISPER_COMPUTE=int8
ENV PYTHONUNBUFFERED=1

EXPOSE 10000

CMD ["uvicorn", "app.main:app", "--app-dir", "/app/backend", "--host", "0.0.0.0", "--port", "10000"]
