"""WebSocket benchmark client for VietVoice TTS server.

Benchmark flow:
1) Connect to websocket server.
2) If connection succeeds, generate many texts from 3 to 100 words.
3) For each text, send UTF-8 bytes and receive streamed WAV bytes.
4) Measure TTFB, total time, throughput, audio duration and real-time factor.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import random
import statistics
import time
import wave
from dataclasses import dataclass, asdict
from typing import List, Sequence

import websockets


VI_WORDS: Sequence[str] = (
    "xin",
    "chao",
    "tat",
    "ca",
    "cac",
    "ban",
    "toi",
    "la",
    "mot",
    "he",
    "thong",
    "tong",
    "hop",
    "giong",
    "noi",
    "tieng",
    "viet",
    "chat",
    "luong",
    "cao",
    "nhanh",
    "on",
    "dinh",
    "du",
    "lieu",
    "tham",
    "chieu",
    "am",
    "thanh",
    "van",
    "ban",
    "noi",
    "dung",
    "benchmark",
    "thu",
    "nghiem",
    "hieu",
    "nang",
    "do",
    "latency",
    "throughput",
    "thoi",
    "gian",
    "phan",
    "hoi",
    "dau",
    "tien",
    "tong",
    "thoi",
    "gian",
    "xu",
    "ly",
    "server",
    "client",
    "websocket",
    "mo",
    "hinh",
    "onnx",
    "runtime",
    "pipeline",
    "chunk",
    "cross",
    "fade",
    "toc",
    "do",
    "doc",
    "tu",
    "nhien",
    "de",
    "nghe",
    "ro",
    "rang",
    "chinh",
    "xac",
    "nhe",
    "nhang",
    "mang",
    "internet",
    "truyen",
    "tai",
    "goi",
    "du",
    "lieu",
    "moi",
    "lan",
    "kiem",
    "tra",
    "gia",
    "tri",
    "trung",
    "binh",
    "phuong",
    "sai",
    "chuan",
    "tuong",
    "doi",
    "khung",
    "mau",
    "tan",
    "so",
    "hai",
    "muoi",
    "bon",
    "nghin",
    "phu",
    "hop",
    "ung",
    "dung",
    "thuc",
    "te",
    "san",
    "sang",
    "trien",
    "khai",
)


@dataclass
class QueryResult:
    index: int
    words: int
    request_bytes: int
    response_bytes: int
    ttfb_ms: float
    total_ms: float
    throughput_kib_s: float
    audio_duration_s: float
    rtf: float
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


def wav_duration_seconds(wav_bytes: bytes) -> float:
    if not wav_bytes:
        return 0.0
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            frames = wf.getnframes()
            framerate = wf.getframerate()
            return float(frames) / float(framerate) if framerate > 0 else 0.0
    except wave.Error:
        return 0.0


def generate_texts(
    min_words: int,
    max_words: int,
    samples_per_length: int,
    seed: int,
) -> List[str]:
    rnd = random.Random(seed)
    texts: List[str] = []
    for n in range(min_words, max_words + 1):
        for _ in range(samples_per_length):
            words = [rnd.choice(VI_WORDS) for _ in range(n)]
            sentence = " ".join(words)
            sentence = sentence.capitalize() + "."
            texts.append(sentence)
    rnd.shuffle(texts)
    return texts


async def run_single_query(ws, index: int, text: str) -> QueryResult:
    req = text.encode("utf-8")

    start = time.perf_counter()
    await ws.send(req)

    ttfb_ms = 0.0
    first_chunk_at: float | None = None
    chunks: List[bytes] = []
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

        # End-of-response marker from server.
        if len(bchunk) == 0:
            break

        if bchunk.startswith(b"ERR:"):
            error = bchunk[4:].decode("utf-8", errors="replace")
            # Continue until end marker so protocol stays synchronized.
            continue

        chunks.append(bchunk)

    end = time.perf_counter()

    wav_bytes = b"".join(chunks)
    total_ms = (end - start) * 1000.0
    resp_bytes = len(wav_bytes)
    throughput = (resp_bytes / 1024.0) / (total_ms / 1000.0) if total_ms > 0 else 0.0
    audio_duration = wav_duration_seconds(wav_bytes)
    rtf = (total_ms / 1000.0) / audio_duration if audio_duration > 0 else 0.0

    return QueryResult(
        index=index,
        words=len(text.split()),
        request_bytes=len(req),
        response_bytes=resp_bytes,
        ttfb_ms=ttfb_ms,
        total_ms=total_ms,
        throughput_kib_s=throughput,
        audio_duration_s=audio_duration,
        rtf=rtf,
        ok=(error == ""),
        error=error,
    )


def summarize_results(results: List[QueryResult]) -> None:
    total = len(results)
    oks = [r for r in results if r.ok]
    fails = [r for r in results if not r.ok]

    print("\n===== Benchmark Summary =====")
    print(f"Total queries: {total}")
    print(f"Success: {len(oks)}")
    print(f"Failed: {len(fails)}")

    if not oks:
        print("No successful queries to summarize.")
        return

    ttfb = sorted(r.ttfb_ms for r in oks)
    total_ms = sorted(r.total_ms for r in oks)
    thr = sorted(r.throughput_kib_s for r in oks)
    rtf_vals = sorted(r.rtf for r in oks if r.rtf > 0)

    print("\nLatency (ms)")
    print(f"TTFB p50: {percentile(ttfb, 0.50):.2f}")
    print(f"TTFB p95: {percentile(ttfb, 0.95):.2f}")
    print(f"TTFB mean: {statistics.mean(ttfb):.2f}")
    print(f"Total p50: {percentile(total_ms, 0.50):.2f}")
    print(f"Total p95: {percentile(total_ms, 0.95):.2f}")
    print(f"Total mean: {statistics.mean(total_ms):.2f}")

    print("\nThroughput")
    print(f"Response throughput mean (KiB/s): {statistics.mean(thr):.2f}")

    if rtf_vals:
        print("\nReal-Time Factor (lower is better)")
        print(f"RTF p50: {percentile(rtf_vals, 0.50):.3f}")
        print(f"RTF p95: {percentile(rtf_vals, 0.95):.3f}")
        print(f"RTF mean: {statistics.mean(rtf_vals):.3f}")


def save_csv(path: str, results: List[QueryResult]) -> None:
    fieldnames = (
        list(asdict(results[0]).keys())
        if results
        else list(QueryResult.__annotations__.keys())
    )
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(asdict(row))


async def run_benchmark(args: argparse.Namespace) -> int:
    texts = generate_texts(
        min_words=args.min_words,
        max_words=args.max_words,
        samples_per_length=args.samples_per_length,
        seed=args.seed,
    )

    print(
        f"Prepared {len(texts)} queries (word range: {args.min_words}-{args.max_words})."
    )
    print(f"Connecting to: {args.url}")

    try:
        async with websockets.connect(args.url, max_size=None) as ws:
            print("WebSocket connected. Benchmark starts now.")

            results: List[QueryResult] = []

            for idx, text in enumerate(texts, start=1):
                result = await run_single_query(ws, idx, text)
                results.append(result)

                status = "OK" if result.ok else f"FAIL ({result.error})"
                print(
                    f"[{idx}/{len(texts)}] {status} | words={result.words} | "
                    f"TTFB={result.ttfb_ms:.2f}ms | Total={result.total_ms:.2f}ms | "
                    f"Bytes={result.response_bytes}"
                )

            summarize_results(results)

            if args.output_csv:
                save_csv(args.output_csv, results)
                print(f"\nSaved per-query results to: {args.output_csv}")

            return 0

    except Exception as exc:
        print(f"Failed to connect or run benchmark: {exc}")
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark VietVoice TTS websocket server"
    )
    parser.add_argument("--url", default="ws://127.0.0.1:8765/ws", help="WebSocket URL")
    parser.add_argument(
        "--min-words", type=int, default=3, help="Minimum words per query"
    )
    parser.add_argument(
        "--max-words", type=int, default=100, help="Maximum words per query"
    )
    parser.add_argument(
        "--samples-per-length", type=int, default=2, help="Queries per each word length"
    )
    parser.add_argument("--seed", type=int, default=20260319, help="Random seed")
    parser.add_argument(
        "--output-csv",
        default="benchmark/ws_benchmark_results.csv",
        help="CSV result path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_words < 1:
        raise ValueError("--min-words must be >= 1")
    if args.max_words < args.min_words:
        raise ValueError("--max-words must be >= --min-words")
    if args.samples_per_length < 1:
        raise ValueError("--samples-per-length must be >= 1")

    raise_code = asyncio.run(run_benchmark(args))
    raise SystemExit(raise_code)


if __name__ == "__main__":
    main()
