import os
import time
import asyncio
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager
import json

from .api import TTSApi
from .core import ModelConfig

# Global TTS API instance
_tts_api = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _tts_api
    print("🚀 Initializing Optimized TTS Engine for H100...")
    config = ModelConfig(
        use_tensorrt=True,  # Re-enable for H100 performance
        use_fp16=False,  # Keep FP32 for maximum quality
        use_io_binding=True,  # Re-enable for faster H100 inference
        use_cuda_graph=True,  # Enable for extra speed
        micro_chunking_words=0,
        first_chunk_nfe_step=None,
        nfe_step=32,
    )
    _tts_api = TTSApi(config)
    # Trigger model loading
    _ = _tts_api.engine
    print("✅ System Ready")
    yield
    if _tts_api:
        _tts_api.engine.cleanup()


app = FastAPI(title="VietVoice TTS High-Performance Server", lifespan=lifespan)


def get_tts_api():
    return _tts_api


@app.get("/health")
async def health_check():
    return {"status": "healthy", "device": "H100/GPU-Ready"}


@app.websocket("/ws/tts")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    api = get_tts_api()

    try:
        while True:
            # User gửi text thuần
            text = await websocket.receive_text()

            if not text or not text.strip():
                continue

            # Sử dụng giọng trong RAM mặc định
            start_time = time.time()
            chunk_count = 0

            for audio_chunk in api.synthesize_stream(text=text):
                # Ensure audio is in int16 for bytes transmission
                # to_int16_safe is robust for streaming (no per-chunk normalization artifacts)
                audio_int16 = api.engine.audio_processor.to_int16_safe(audio_chunk)

                await websocket.send_bytes(audio_int16.tobytes())
                chunk_count += 1
                await asyncio.sleep(0)

            elapsed = time.time() - start_time
            await websocket.send_text(
                json.dumps(
                    {
                        "status": "completed",
                        "chunks": chunk_count,
                        "time": f"{elapsed:.2f}s",
                    }
                )
            )

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        await websocket.send_text(json.dumps({"error": str(e)}))
        await websocket.close()


if __name__ == "__main__":
    import uvicorn

    # Optimized for H100 server
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
