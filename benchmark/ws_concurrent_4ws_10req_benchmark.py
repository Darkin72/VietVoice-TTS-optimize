"""Concurrent WebSocket benchmark for VietVoice TTS server.

Scenario:
1) Open exactly 4 websocket connections.
2) Dispatch 10 requests concurrently at nearly the same time.
3) Requests are assigned round-robin to the 4 connections.
4) Each connection processes its own assigned requests sequentially
   to keep request/response framing synchronized per websocket.
5) Report per-request metrics and aggregate latency stats.
"""

from __future__ import annotations

import asyncio
import csv
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Sequence

import websockets


WS_URL = "ws://127.0.0.1:8765/ws"
OUTPUT_CSV = "benchmark/ws_concurrent_4ws_10req_results.csv"

CONCURRENT_WEBSOCKETS = 4
TOTAL_REQUESTS = 10

# Hardcoded Vietnamese sentences for deterministic benchmark behavior.
TEST_TEXTS: Sequence[str] = (
    "Xin chao ban, toi dang thu nghiem he thong callBot.",
    "He thong can phan hoi nhanh va on dinh khi co nhieu nguoi goi cung luc.",
    "Ban hay doc cau nay voi nhip dieu tu nhien va ro rang.",
    "Do tre phan hoi dau tien la chi so rat quan trong cho trai nghiem thoai.",
    "Neu hang doi hop ly, chat luong am thanh van co the giu on dinh.",
    "Moi yeu cau duoc xu ly tuan tu theo tung ket noi websocket rieng.",
    "Kich ban nay mo phong bon ket noi dong thoi gui nhieu thong diep.",
    "Muc tieu la tranh race condition va van dam bao do tre chap nhan duoc.",
    "Ban benchmark nay su dung du lieu co dinh de ket qua de so sanh qua cac lan chay.",
    "Sau khi chay xong, chung ta xem p50 va p95 cua TTFB de danh gia hieu nang.",
)


@dataclass
class RequestResult:
    request_id: int
    connection_id: int
    words: int
    request_bytes: int
    response_bytes: int
    ttfb_ms: float
    total_ms: float
    ok: bool
    error: str


def percentile(sorted_values: List[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * p
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    frac = rank - low
    return sorted_values[low] * (1.0 - frac) + sorted_values[high] * frac


async def run_single_query(
    ws: websockets.ClientConnection, request_id: int, connection_id: int, text: str
) -> RequestResult:
    req = text.encode("utf-8")

    start = time.perf_counter()
    await ws.send(req)

    ttfb_ms = 0.0
    first_chunk_at: float | None = None
    response_bytes = 0
    error = ""

    while True:
        chunk = await ws.recv()
        now = time.perf_counter()

        if not isinstance(chunk, (bytes, bytearray)):
            error = "received non-binary websocket frame"
            break

        if first_chunk_at is None:
            first_chunk_at = now
            ttfb_ms = (first_chunk_at - start) * 1000.0

        bchunk = bytes(chunk)

        # End-of-response marker.
        if len(bchunk) == 0:
            break

        if bchunk.startswith(b"ERR:"):
            error = bchunk[4:].decode("utf-8", errors="replace")
            # Continue until end marker to keep framing synchronized.
            continue

        response_bytes += len(bchunk)

    total_ms = (time.perf_counter() - start) * 1000.0

    return RequestResult(
        request_id=request_id,
        connection_id=connection_id,
        words=len(text.split()),
        request_bytes=len(req),
        response_bytes=response_bytes,
        ttfb_ms=ttfb_ms,
        total_ms=total_ms,
        ok=(error == ""),
        error=error,
    )


async def connection_worker(
    ws: websockets.ClientConnection,
    connection_id: int,
    assigned_jobs: list[tuple[int, str]],
    start_event: asyncio.Event,
    done_map: dict[int, RequestResult],
) -> None:
    await start_event.wait()
    for request_id, text in assigned_jobs:
        result = await run_single_query(ws, request_id, connection_id, text)
        done_map[request_id] = result


def summarize(results: List[RequestResult]) -> None:
    total = len(results)
    oks = [r for r in results if r.ok]
    fails = [r for r in results if not r.ok]

    print("\n===== Concurrent Benchmark Summary =====")
    print(f"Connections: {CONCURRENT_WEBSOCKETS}")
    print(f"Total concurrent requests: {TOTAL_REQUESTS}")
    print(f"Success: {len(oks)}")
    print(f"Failed: {len(fails)}")

    if not oks:
        print("No successful requests to summarize.")
        return

    ttfb_values = sorted(r.ttfb_ms for r in oks)
    total_values = sorted(r.total_ms for r in oks)

    print("\nLatency (ms)")
    print(f"TTFB p50: {percentile(ttfb_values, 0.50):.2f}")
    print(f"TTFB p95: {percentile(ttfb_values, 0.95):.2f}")
    print(f"TTFB mean: {statistics.mean(ttfb_values):.2f}")
    print(f"Total p50: {percentile(total_values, 0.50):.2f}")
    print(f"Total p95: {percentile(total_values, 0.95):.2f}")
    print(f"Total mean: {statistics.mean(total_values):.2f}")


def save_csv(path: str, results: List[RequestResult]) -> None:
    fieldnames = (
        list(asdict(results[0]).keys())
        if results
        else list(RequestResult.__annotations__.keys())
    )
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


async def run_benchmark() -> int:
    if CONCURRENT_WEBSOCKETS != 4:
        raise ValueError("This benchmark is fixed to exactly 4 websocket connections")
    if TOTAL_REQUESTS != 10:
        raise ValueError("This benchmark is fixed to exactly 10 requests")

    print(f"Connecting to {WS_URL} with {CONCURRENT_WEBSOCKETS} websocket clients...")

    try:
        async with (
            websockets.connect(WS_URL, max_size=None) as ws0,
            websockets.connect(WS_URL, max_size=None) as ws1,
            websockets.connect(WS_URL, max_size=None) as ws2,
            websockets.connect(WS_URL, max_size=None) as ws3,
        ):
            sockets = [ws0, ws1, ws2, ws3]
            print("All 4 websocket connections established.")

            jobs_per_connection: list[list[tuple[int, str]]] = [[], [], [], []]
            for request_id in range(1, TOTAL_REQUESTS + 1):
                text = TEST_TEXTS[request_id - 1]
                conn_id = (request_id - 1) % CONCURRENT_WEBSOCKETS
                jobs_per_connection[conn_id].append((request_id, text))

            start_event = asyncio.Event()
            done_map: dict[int, RequestResult] = {}

            workers = [
                asyncio.create_task(
                    connection_worker(
                        ws=sockets[idx],
                        connection_id=idx,
                        assigned_jobs=jobs_per_connection[idx],
                        start_event=start_event,
                        done_map=done_map,
                    )
                )
                for idx in range(CONCURRENT_WEBSOCKETS)
            ]

            start_at = time.perf_counter()
            start_event.set()
            await asyncio.gather(*workers)
            wall_ms = (time.perf_counter() - start_at) * 1000.0

            ordered_results = [done_map[i] for i in range(1, TOTAL_REQUESTS + 1)]

            print("\n===== Per Request =====")
            for r in ordered_results:
                status = "OK" if r.ok else f"FAIL ({r.error})"
                print(
                    f"req={r.request_id:02d} | conn={r.connection_id} | {status} | "
                    f"TTFB={r.ttfb_ms:.2f}ms | Total={r.total_ms:.2f}ms | "
                    f"bytes={r.response_bytes}"
                )

            summarize(ordered_results)
            print(f"\nWall-clock time for whole run: {wall_ms:.2f}ms")

            save_csv(OUTPUT_CSV, ordered_results)
            print(f"Saved detailed results to: {OUTPUT_CSV}")

            return 0

    except Exception as exc:
        print(f"Benchmark failed: {exc}")
        return 1


def main() -> None:
    exit_code = asyncio.run(run_benchmark())
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
