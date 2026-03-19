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


def create_app(
    speed: float = 1.0,
    nfe_step: int = 28,
    fuse_nfe: int = 1,
    random_seed: int = 9527,
    inter_op_threads: int = 0,
    intra_op_threads: int = 0,
    cuda_device_id: int = 0,
    cuda_conv_algo_search: str = "HEURISTIC",
    cuda_conv_use_max_workspace: bool = True,
    cuda_copy_in_default_stream: bool = True,
    enable_cuda_graph: bool = False,
    chunk_size: int = 16384,
) -> FastAPI:
    config = ModelConfig(
        speed=speed,
        nfe_step=nfe_step,
        fuse_nfe=fuse_nfe,
        random_seed=random_seed,
        inter_op_num_threads=inter_op_threads,
        intra_op_num_threads=intra_op_threads,
        cuda_device_id=cuda_device_id,
        cuda_conv_algo_search=cuda_conv_algo_search,
        cuda_conv_use_max_workspace=cuda_conv_use_max_workspace,
        cuda_copy_in_default_stream=cuda_copy_in_default_stream,
        enable_cuda_graph=enable_cuda_graph,
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

                for start in range(0, len(wav_bytes), chunk_size):
                    await websocket.send_bytes(wav_bytes[start : start + chunk_size])

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
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed")
    parser.add_argument("--nfe-step", type=int, default=28, help="Number of flow steps")
    parser.add_argument("--fuse-nfe", type=int, default=1, help="Fused NFE steps")
    parser.add_argument("--random-seed", type=int, default=9527, help="Random seed")
    parser.add_argument(
        "--inter-op-threads", type=int, default=0, help="ORT inter-op threads"
    )
    parser.add_argument(
        "--intra-op-threads", type=int, default=0, help="ORT intra-op threads"
    )
    parser.add_argument("--cuda-device-id", type=int, default=0, help="CUDA device id")
    parser.add_argument(
        "--cuda-conv-algo-search",
        choices=["HEURISTIC", "EXHAUSTIVE", "DEFAULT"],
        default="HEURISTIC",
        help="cuDNN convolution algorithm search mode",
    )
    parser.add_argument(
        "--disable-cuda-max-workspace",
        action="store_true",
        help="Disable cuDNN max workspace for CUDAExecutionProvider",
    )
    parser.add_argument(
        "--disable-cuda-copy-in-default-stream",
        action="store_true",
        help="Disable copy in default stream for CUDAExecutionProvider",
    )
    parser.add_argument(
        "--enable-cuda-graph",
        action="store_true",
        help="Enable CUDA graph in CUDAExecutionProvider (if supported)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=16384, help="Chunk size for streaming bytes"
    )
    parser.add_argument("--log-level", default="info", help="Uvicorn log level")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    app = create_app(
        speed=args.speed,
        nfe_step=args.nfe_step,
        fuse_nfe=args.fuse_nfe,
        random_seed=args.random_seed,
        inter_op_threads=args.inter_op_threads,
        intra_op_threads=args.intra_op_threads,
        cuda_device_id=args.cuda_device_id,
        cuda_conv_algo_search=args.cuda_conv_algo_search,
        cuda_conv_use_max_workspace=not args.disable_cuda_max_workspace,
        cuda_copy_in_default_stream=not args.disable_cuda_copy_in_default_stream,
        enable_cuda_graph=args.enable_cuda_graph,
        chunk_size=args.chunk_size,
    )

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
