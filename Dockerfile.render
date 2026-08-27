# Hinglish ASR — Render Docker image (free tier, 512 MB RAM)
# Uses the smaller `base` model to fit free-tier memory. Model is downloaded
# from the HF Hub at BUILD time (Render's build network is fast) and baked
# into the image, so there is no boot-time download or cold-start penalty.
FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY backend /app/backend

# Bake the base model into the image at build time.
RUN mkdir -p /app/models/faster-whisper-base \
    && for f in model.bin config.json tokenizer.json vocabulary.txt; do \
         curl -fsSL --retry 3 -o /app/models/faster-whisper-base/$f \
           "https://huggingface.co/Systran/faster-whisper-base/resolve/main/$f"; \
       done

ENV WHISPER_MODEL=/app/models/faster-whisper-base
ENV WHISPER_COMPUTE=int8
ENV PYTHONUNBUFFERED=1

EXPOSE 10000

CMD ["uvicorn", "app.main:app", "--app-dir", "/app/backend", "--host", "0.0.0.0", "--port", "10000"]
