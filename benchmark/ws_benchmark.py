"""WebSocket benchmark client for VietVoice TTS server.

Benchmark flow:
1) Connect to websocket server.
2) If connection succeeds, generate many texts from 3 to 100 words.
3) For each text, send UTF-8 bytes and receive streamed WAV bytes.
4) Measure TTFB, total time, throughput, audio duration and real-time factor.
"""

from __future__ import annotations

import asyncio
import csv
import io
from pathlib import Path
import statistics
import time
import wave
from dataclasses import dataclass, asdict
from typing import List, Sequence, Tuple

import websockets


WS_URL = "ws://127.0.0.1:8765/ws"
OUTPUT_CSV = "benchmark/ws_benchmark_results.csv"
OUTPUT_AUDIO = "benchmark/ws_longest_query.wav"
OUTPUT_LONGEST_TEXT = "benchmark/ws_longest_query.txt"

# 50 câu tiếng Việt có dấu, sắp xếp từ ngắn đến dài để benchmark nhất quán.
BENCHMARK_TEXTS: Sequence[str] = (
    "Xin chào.",
    "Tôi đang thử.",
    "Đây là bài đo.",
    "Bạn nghe rõ chứ?",
    "Máy chủ đã sẵn sàng.",
    "Âm thanh đầu ra ổn định.",
    "Tôi gửi thêm một câu ngắn.",
    "Hệ thống phản hồi khá nhanh.",
    "Đoạn này dùng để kiểm tra.",
    "Chất giọng hiện tại nghe tự nhiên.",
    "Chúng ta bắt đầu chạy benchmark nhé.",
    "Mỗi truy vấn sẽ được đo độ trễ.",
    "Tôi muốn kiểm tra cả tốc độ đọc.",
    "Câu này có dấu phẩy, nghe mượt hơn.",
    "Dữ liệu đầu vào cần rõ nghĩa và sạch.",
    "Kết quả tốt phải ổn định qua nhiều lần.",
    "Máy khách sẽ ghi nhận thời gian phản hồi đầu tiên.",
    "Sau đó chương trình tính tổng thời gian xử lý toàn bộ.",
    "Nếu mạng dao động, thông lượng có thể giảm nhẹ theo thời điểm.",
    "Chúng tôi vẫn ưu tiên giọng đọc rõ ràng hơn tốc độ tuyệt đối.",
    "Câu kiểm thử này được viết để nghe tự nhiên trong ngữ cảnh thực tế.",
    "Bạn có thể mở file âm thanh sau khi chạy xong để tự đánh giá nhanh.",
    "Trong báo cáo, chỉ số TTFB giúp quan sát độ trễ phản hồi ban đầu.",
    "Chỉ số tổng thời gian sẽ phản ánh mức độ ổn định của pipeline tổng hợp.",
    "Nếu một truy vấn lỗi, hệ thống vẫn cần đồng bộ luồng dữ liệu đến cuối.",
    "Câu này dài hơn một chút để kiểm tra cách mô hình xử lý nhịp ngắt tự nhiên.",
    "Khi câu văn có đủ dấu câu, nhịp điệu phát âm thường mềm mại và dễ nghe hơn.",
    "Bản benchmark cố ý dùng nội dung rõ ràng để hạn chế sai lệch do từ vô nghĩa gây ra.",
    "Trong thực tế triển khai, độ trễ ổn định thường quan trọng không kém chất lượng phát âm.",
    "Sau mỗi lượt chạy, chúng ta cần lưu log chi tiết để tiện so sánh giữa các phiên bản mô hình.",
    "Nếu âm đầu ra bị méo hoặc hụt hơi, cần kiểm tra lại tiền xử lý văn bản và chuẩn hóa dấu câu.",
    "Đoạn văn này được thiết kế để nghe như lời nói tự nhiên, không phải chuỗi từ ghép ngẫu nhiên.",
    "Một hệ thống TTS tốt phải cân bằng giữa độ rõ, độ tự nhiên, tốc độ phản hồi và mức dùng tài nguyên.",
    "Khi tăng độ dài câu, chúng ta sẽ quan sát xem thời gian xử lý tăng tuyến tính hay có điểm nghẽn bất thường.",
    "Để đảm bảo nhất quán, toàn bộ danh sách câu được hardcode thay vì sinh ngẫu nhiên theo từng lần chạy.",
    "Câu benchmark này có cấu trúc đầy đủ chủ ngữ và vị ngữ, giúp mô hình đọc đúng nhịp điệu hơn đáng kể.",
    "Ngoài tốc độ, người dùng cuối thường quan tâm chất lượng giọng đọc có tự nhiên, rõ chữ và dễ nghe hay không.",
    "Nếu cần phân tích sâu hơn, bạn có thể đối chiếu thêm thời lượng audio thực tế với tổng thời gian xử lý hệ thống.",
    "Trong môi trường mạng không ổn định, việc theo dõi thông lượng trung bình theo từng truy vấn là rất cần thiết.",
    "Chúng tôi lưu riêng câu dài nhất để nghe kiểm định, vì đoạn dài thường bộc lộ lỗi nhịp ngắt rõ ràng nhất.",
    "Bài đo này giả lập kịch bản người dùng gửi liên tục nhiều câu với độ dài tăng dần từ ngắn đến dài rõ rệt.",
    "Khi chỉ số RTF nhỏ hơn một, mô hình thường có khả năng tổng hợp nhanh hơn tốc độ phát lại âm thanh thực tế.",
    "Nếu bạn thấy phát âm chưa chuẩn ở một số từ, hãy rà soát bộ từ điển, chuẩn hóa văn bản và bộ tiền xử lý đầu vào.",
    "Bằng cách dùng các câu có nghĩa và đầy đủ dấu câu, kết quả benchmark sẽ phản ánh đúng chất lượng sử dụng ngoài đời.",
    "Mỗi câu trong danh sách được viết có chủ đích để bao phủ nhiều mẫu ngữ điệu, từ câu trần thuật ngắn đến câu dài phức hợp.",
    "Trong giai đoạn tối ưu, chúng ta nên giữ nguyên dữ liệu benchmark cố định để mọi thay đổi hiệu năng đều có thể so sánh công bằng.",
    "Nếu một bản cập nhật giúp giảm TTFB nhưng làm giọng đọc kém tự nhiên, quyết định triển khai cần cân nhắc theo ưu tiên sản phẩm.",
    "Kết quả đáng tin cậy không chỉ đến từ một lần chạy, mà còn từ việc lặp lại cùng dữ liệu chuẩn và theo dõi xu hướng qua thời gian.",
    "Câu dài áp chót này được thêm để kiểm tra khả năng duy trì chất lượng phát âm khi văn bản tăng dần độ phức tạp theo nhiều mệnh đề.",
    "Câu dài nhất trong bộ benchmark được dùng để lưu lại âm thanh kiểm định cuối cùng, giúp bạn nghe trực tiếp và đánh giá toàn diện độ tự nhiên, độ rõ chữ, nhịp ngắt, cũng như tính ổn định của hệ thống tổng hợp tiếng nói.",
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


def generate_texts() -> List[str]:
    return list(BENCHMARK_TEXTS)


async def run_single_query(ws, index: int, text: str) -> Tuple[QueryResult, bytes]:
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

    return (
        QueryResult(
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
        ),
        wav_bytes,
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


def save_audio(path: str, audio_bytes: bytes) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(audio_bytes)


def save_text(path: str, content: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


async def run_benchmark() -> int:
    texts = generate_texts()

    if len(texts) != 50:
        raise ValueError(f"Expected exactly 50 benchmark sentences, got {len(texts)}")

    print(f"Prepared {len(texts)} hardcoded queries (short to long).")
    print(f"Connecting to: {WS_URL}")

    try:
        async with websockets.connect(WS_URL, max_size=None) as ws:
            print("WebSocket connected. Benchmark starts now.")

            results: List[QueryResult] = []
            longest_success_words = -1
            longest_success_audio = b""
            longest_success_text = ""

            for idx, text in enumerate(texts, start=1):
                result, wav_bytes = await run_single_query(ws, idx, text)
                results.append(result)

                if result.ok and wav_bytes and result.words > longest_success_words:
                    longest_success_words = result.words
                    longest_success_audio = wav_bytes
                    longest_success_text = text

                status = "OK" if result.ok else f"FAIL ({result.error})"
                print(
                    f"[{idx}/{len(texts)}] {status} | words={result.words} | "
                    f"TTFB={result.ttfb_ms:.2f}ms | Total={result.total_ms:.2f}ms | "
                    f"Bytes={result.response_bytes}"
                )

            summarize_results(results)

            save_csv(OUTPUT_CSV, results)
            print(f"\nSaved per-query results to: {OUTPUT_CSV}")

            if longest_success_audio:
                save_audio(OUTPUT_AUDIO, longest_success_audio)
                save_text(OUTPUT_LONGEST_TEXT, longest_success_text)
                print(
                    "Saved longest-query audio "
                    f"({longest_success_words} words) to: {OUTPUT_AUDIO}"
                )
                print(f"Saved longest query text to: {OUTPUT_LONGEST_TEXT}")
            else:
                print(
                    "Did not save longest-query audio because no successful WAV was produced."
                )

            return 0

    except Exception as exc:
        print(f"Failed to connect or run benchmark: {exc}")
        return 1


def main() -> None:
    raise_code = asyncio.run(run_benchmark())
    raise SystemExit(raise_code)


if __name__ == "__main__":
    main()
