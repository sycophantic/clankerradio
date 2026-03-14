#!/usr/bin/env python3

import os
import uuid
import shutil
import asyncio
from datetime import datetime, timezone

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel

# ======================
# CONFIG
# ======================

MODEL_SIZE = os.getenv("WHISPER_MODEL", "distil-large-v3")
DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE", "float16")

MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "4"))
TMP_DIR = "/tmp/whisper_uploads"

os.makedirs(TMP_DIR, exist_ok=True)

# ======================
# INIT
# ======================

print(f"[INIT] Loading model: {MODEL_SIZE} ({DEVICE}, {COMPUTE_TYPE})")
model = WhisperModel(
    MODEL_SIZE,
    device=DEVICE,
    compute_type=COMPUTE_TYPE,
)

semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
app = FastAPI(title="Whisper Backend")

# ======================
# HELPERS
# ======================

def iso_utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

# ======================
# ROUTES
# ======================

@app.post("/v1/audio/transcriptions")
async def transcribe_audio(
    file: UploadFile = File(...),
):
    # Fast reject if overloaded
    if semaphore.locked() and semaphore._value == 0:
        raise HTTPException(
            status_code=429,
            detail="Transcription server busy, retry later"
        )

    job_id = str(uuid.uuid4())
    tmp_path = os.path.join(TMP_DIR, f"{job_id}_{file.filename}")

    # Save upload
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    async with semaphore:
        try:
            segments, info = model.transcribe(
                tmp_path,
                beam_size=10,
                language="en",
                condition_on_previous_text=False,
            )

            text = []
            segments_out = []

            for s in segments:
                text.append(s.text)
                segments_out.append({
                    "start": s.start,
                    "end": s.end,
                    "text": s.text,
                })

            full_text = " ".join(text).strip()

            response = {
                "id": job_id,
                "object": "transcription",
                "created": int(datetime.now(tz=timezone.utc).timestamp()),
                "model": MODEL_SIZE,
                "text": full_text,
                "segments": segments_out,
                "language": info.language,
                "duration": info.duration,
            }

            return JSONResponse(response)

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Transcription failed: {e}"
            )

        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_SIZE,
        "device": DEVICE,
        "max_concurrent_jobs": MAX_CONCURRENT_JOBS,
    }

