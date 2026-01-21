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
        use_tensorrt=True,
        use_fp16=True,
        use_io_binding=True,
        use_cuda_graph=True,
        micro_chunking_words=5,
        first_chunk_nfe_step=16,
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
                if audio_chunk.dtype != np.int16:
                    # Assume it's float32 in [-1.0, 1.0]
                    audio_int16 = (audio_chunk * 32767).astype(np.int16)
                else:
                    audio_int16 = audio_chunk

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
