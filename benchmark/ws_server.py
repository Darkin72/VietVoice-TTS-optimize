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
from contextlib import asynccontextmanager, suppress
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
LOCKED_ENGINE_COUNT = 4
LOCKED_CHUNK_SIZE = 16384
LOCKED_MAX_ACTIVE_WEBSOCKETS = LOCKED_ENGINE_COUNT
LOCKED_PER_CLIENT_QUEUE_SIZE = 32


def create_app() -> FastAPI:
    def build_model_config() -> ModelConfig:
        return ModelConfig(
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

    engine_apis = [TTSApi(build_model_config()) for _ in range(LOCKED_ENGINE_COUNT)]
    engine_inference_locks = [asyncio.Lock() for _ in range(LOCKED_ENGINE_COUNT)]

    # Guard websocket slot accounting and engine assignment.
    active_ws_count = 0
    engine_ws_counts = [0 for _ in range(LOCKED_ENGINE_COUNT)]
    connection_assignment_lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # Preload all model sessions at startup.
        for idx, api in enumerate(engine_apis):
            engine = api.engine
            config = api.config
            print(
                f"Engine[{idx}] config: "
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

        # Warm-up with short/medium/long text so first real request is ready.
        warmup_texts = (
            "Xin chao.",
            "Xin chao, day la cau warm-up trung binh de khoi tao on dinh runtime.",
            "Day la cau warm-up dai hon de mo truoc cache shape va duong suy dien cho workload thuc te,"
            " giup request dau tien phan hoi nhanh va it bien dong hon.",
        )
        for api in engine_apis:
            for warmup_text in warmup_texts:
                await asyncio.to_thread(api.synthesize_to_bytes, warmup_text)

        try:
            yield
        finally:
            for api in engine_apis:
                api.cleanup()

    app = FastAPI(title="VietVoice TTS WebSocket Server", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.websocket("/ws")
    async def ws_infer(websocket: WebSocket) -> None:
        nonlocal active_ws_count

        assigned_engine_idx = -1
        has_ws_slot = False
        async with connection_assignment_lock:
            if active_ws_count < LOCKED_MAX_ACTIVE_WEBSOCKETS:
                # Keep websocket pinned to one engine for stable low TTFB.
                assigned_engine_idx = min(
                    range(LOCKED_ENGINE_COUNT), key=lambda idx: engine_ws_counts[idx]
                )
                engine_ws_counts[assigned_engine_idx] += 1
                active_ws_count += 1
                has_ws_slot = True

        if not has_ws_slot:
            await websocket.accept()
            await websocket.send_bytes(b"ERR:too many websocket connections")
            await websocket.send_bytes(b"")
            await websocket.close(code=1013)
            return

        await websocket.accept()
        assigned_api = engine_apis[assigned_engine_idx]
        assigned_engine_lock = engine_inference_locks[assigned_engine_idx]
        client_queue: asyncio.Queue[str | None] = asyncio.Queue(
            maxsize=LOCKED_PER_CLIENT_QUEUE_SIZE
        )
        send_lock = asyncio.Lock()

        async def send_frame(data: bytes) -> None:
            async with send_lock:
                await websocket.send_bytes(data)

        async def send_error(err_msg: bytes) -> None:
            await send_frame(err_msg)
            await send_frame(b"")

        async def run_inference_locked(text: str) -> tuple[bytes, float]:
            # Never release engine lock before background inference thread is done.
            async with assigned_engine_lock:
                infer_task = asyncio.create_task(
                    asyncio.to_thread(assigned_api.synthesize_to_bytes, text)
                )
                try:
                    return await asyncio.shield(infer_task)
                except asyncio.CancelledError:
                    await infer_task
                    raise

        async def consume_client_queue() -> None:
            while True:
                text = await client_queue.get()
                if text is None:
                    client_queue.task_done()
                    break

                try:
                    print(f"Engine[{assigned_engine_idx}] processing text: {text}")
                    wav_bytes, _ = await run_inference_locked(text)

                    for start in range(0, len(wav_bytes), LOCKED_CHUNK_SIZE):
                        await send_frame(wav_bytes[start : start + LOCKED_CHUNK_SIZE])

                    # End-of-response marker
                    await send_frame(b"")

                except Exception as exc:  # pragma: no cover - runtime-dependent
                    err = f"ERR:{exc}".encode("utf-8", errors="replace")
                    with suppress(Exception):
                        await send_error(err)
                    break
                finally:
                    client_queue.task_done()

        consumer_task = asyncio.create_task(consume_client_queue())

        try:
            while True:
                try:
                    payload = await websocket.receive_bytes()
                except WebSocketDisconnect:
                    break

                if not payload:
                    await send_error(b"ERR:empty request")
                    continue

                try:
                    text = payload.decode("utf-8").strip()
                except UnicodeDecodeError:
                    await send_error(b"ERR:request must be utf-8 encoded text bytes")
                    continue

                if not text:
                    await send_error(b"ERR:text is empty")
                    continue

                # Queue per client: concurrent messages are buffered and processed in order.
                await client_queue.put(text)

        finally:
            if not consumer_task.done():
                await client_queue.put(None)
            with suppress(Exception):
                await consumer_task

            async with connection_assignment_lock:
                if active_ws_count > 0:
                    active_ws_count -= 1
                if assigned_engine_idx >= 0:
                    engine_ws_counts[assigned_engine_idx] = max(
                        0, engine_ws_counts[assigned_engine_idx] - 1
                    )

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
