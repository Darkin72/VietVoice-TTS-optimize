"""WebSocket inference server for VietVoice TTS using FastAPI.

Protocol (binary frames only):
1) Client sends one binary frame: UTF-8 encoded text.
2) Server synthesizes speech and streams WAV bytes in binary chunks.
3) Server sends an empty binary frame b"" to mark end-of-response.

Error handling:
- On error, server sends one binary frame with prefix b"ERR:" + utf8_message,
  then sends b"" as end-of-response marker.
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from vietvoicetts import ModelConfig, TTSApi


# Locked production profile (A100/H100) for stable latency/quality.
LOCKED_SPEED = 1.0
LOCKED_NFE_STEP = 28
LOCKED_FUSE_NFE = 1
LOCKED_RANDOM_SEED = 9527
LOCKED_INTER_OP_THREADS = 0
LOCKED_INTRA_OP_THREADS = 0
LOCKED_CUDA_DEVICE_ID = 0
LOCKED_CUDA_CONV_ALGO_SEARCH = "HEURISTIC"
LOCKED_CUDA_CONV_USE_MAX_WORKSPACE = True
LOCKED_CUDA_COPY_IN_DEFAULT_STREAM = True
LOCKED_ENABLE_CUDA_GRAPH = False
LOCKED_CHUNK_SIZE = 16384


def create_app() -> FastAPI:
    config = ModelConfig(
        speed=LOCKED_SPEED,
        nfe_step=LOCKED_NFE_STEP,
        fuse_nfe=LOCKED_FUSE_NFE,
        random_seed=LOCKED_RANDOM_SEED,
        inter_op_num_threads=LOCKED_INTER_OP_THREADS,
        intra_op_num_threads=LOCKED_INTRA_OP_THREADS,
        cuda_device_id=LOCKED_CUDA_DEVICE_ID,
        cuda_conv_algo_search=LOCKED_CUDA_CONV_ALGO_SEARCH,
        cuda_conv_use_max_workspace=LOCKED_CUDA_CONV_USE_MAX_WORKSPACE,
        cuda_copy_in_default_stream=LOCKED_CUDA_COPY_IN_DEFAULT_STREAM,
        enable_cuda_graph=LOCKED_ENABLE_CUDA_GRAPH,
    )
    api = TTSApi(config)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # Preload model/session at startup.
        engine = api.engine
        print(
            "Model config: "
            f"nfe_step={config.nfe_step}, "
            f"fuse_nfe={config.fuse_nfe}, "
            f"speed={config.speed}, "
            f"sample_rate={config.sample_rate}, "
            f"inter_op_threads={config.inter_op_num_threads}, "
            f"intra_op_threads={config.intra_op_num_threads}, "
            f"cuda_device_id={config.cuda_device_id}, "
            f"cuda_conv_algo_search={config.cuda_conv_algo_search}, "
            f"cuda_conv_use_max_workspace={config.cuda_conv_use_max_workspace}, "
            f"cuda_copy_in_default_stream={config.cuda_copy_in_default_stream}, "
            f"enable_cuda_graph={config.enable_cuda_graph}, "
            f"providers={engine.model_session_manager.providers}"
        )
        # Run warm-up infer with short/medium/long text so first real request is ready.
        warmup_texts = (
            "Xin chao.",
            "Xin chao, day la cau warm-up trung binh de khoi tao on dinh runtime.",
            "Day la cau warm-up dai hon de mo truoc cache shape va duong suy dien cho workload thuc te,"
            " giup request dau tien phan hoi nhanh va it bien dong hon.",
        )
        for warmup_text in warmup_texts:
            await asyncio.to_thread(api.synthesize_to_bytes, warmup_text)
        try:
            yield
        finally:
            api.cleanup()

    app = FastAPI(title="VietVoice TTS WebSocket Server", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.websocket("/ws")
    async def ws_infer(websocket: WebSocket) -> None:
        await websocket.accept()

        while True:
            try:
                payload = await websocket.receive_bytes()
            except WebSocketDisconnect:
                break

            if not payload:
                await websocket.send_bytes(b"ERR:empty request")
                await websocket.send_bytes(b"")
                continue

            try:
                text = payload.decode("utf-8").strip()
            except UnicodeDecodeError:
                await websocket.send_bytes(
                    b"ERR:request must be utf-8 encoded text bytes"
                )
                await websocket.send_bytes(b"")
                continue

            if not text:
                await websocket.send_bytes(b"ERR:text is empty")
                await websocket.send_bytes(b"")
                continue

            try:
                print(f"Processing text: {text}")
                wav_bytes, _ = await asyncio.to_thread(api.synthesize_to_bytes, text)

                for start in range(0, len(wav_bytes), LOCKED_CHUNK_SIZE):
                    await websocket.send_bytes(
                        wav_bytes[start : start + LOCKED_CHUNK_SIZE]
                    )

                # End-of-response marker
                await websocket.send_bytes(b"")

            except Exception as exc:  # pragma: no cover - runtime-dependent
                err = f"ERR:{exc}".encode("utf-8", errors="replace")
                await websocket.send_bytes(err)
                await websocket.send_bytes(b"")

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VietVoice TTS WebSocket server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8765, help="Bind port")
    parser.add_argument("--log-level", default="info", help="Uvicorn log level")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    app = create_app()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
