# VietVoice TTS - WebSocket Server + Benchmark

Repository này tập trung vào 3 phần chính:

- Core model inference: vietvoicetts/
- WebSocket server cho suy luận TTS: benchmark/ws_server.py
- Bộ benchmark đo hiệu năng: benchmark/ws_benchmark.py và benchmark/ws_concurrent_6ws_30req_benchmark.py

## 1) Môi trường

- Python 3.8+
- Windows/Linux/macOS
- Khuyến nghị dùng virtual environment

Lưu ý: pydub cần ffmpeg để đọc nhiều định dạng audio. Nếu máy chưa có ffmpeg, hãy cài trước.

## 2) Cài đặt

Tại thư mục gốc project:

### CPU

```bash
pip install -e ".[cpu,server,benchmark]"
```

### GPU (CUDA)

```bash
pip install -e ".[gpu,server,benchmark]"
```

## 3) Những tối ưu đã triển khai

Các thay đổi lớn đã áp dụng cho server và benchmark:

1. Cố định profile chạy production (speed, nfe_step, provider options) để kết quả ổn định.
2. Preload model + warm-up nhiều câu ngay lúc startup để giảm cold start.
3. Bảo vệ engine dùng chung bằng khóa suy luận (inference lock), tránh truy cập đồng thời gây race condition.
4. Giới hạn tối đa 6 kết nối WebSocket đang hoạt động cùng lúc.
5. Mỗi client có queue riêng, message concurrent từ cùng client sẽ được xếp hàng và xử lý FIFO.
6. Gửi frame WebSocket theo khóa gửi riêng (send lock) để tránh ghi chồng frame khi có nhiều coroutine.
7. Thêm benchmark concurrent mới: mở 6 websocket và bắn đồng thời 30 request để mô phỏng tải callBot.

## 4) Chạy WebSocket server

```bash
python benchmark/ws_server.py --host 127.0.0.1 --port 8765 --log-level info
```

Server sẽ:

- preload model khi startup
- warm-up trước khi nhận request thật
- mở health check: http://127.0.0.1:8765/health
- mở websocket endpoint: ws://127.0.0.1:8765/ws
- giới hạn 6 websocket active
- tạo queue riêng cho từng websocket client

Các ngưỡng vận hành hiện được khóa trong mã nguồn tại benchmark/ws_server.py:

- LOCKED_MAX_ACTIVE_WEBSOCKETS = 6
- LOCKED_PER_CLIENT_QUEUE_SIZE = 40
- LOCKED_CHUNK_SIZE = 16384

## 5) Protocol WebSocket (binary)

1. Client gửi 1 binary frame chứa text.encode("utf-8").
2. Server suy luận và stream WAV bytes qua nhiều binary frame.
3. Server gửi binary frame rỗng b"" để đánh dấu kết thúc response.

Nếu lỗi:

- Server gửi 1 frame bắt đầu bằng b"ERR:" + message
- Sau đó vẫn gửi b"" để đóng response

## 6) Chạy benchmark tuần tự (50 câu cố định)

```bash
python benchmark/ws_benchmark.py
```

Mặc định script sẽ:

- dùng danh sách 50 câu tiếng Việt cố định (deterministic)
- gửi tuần tự trên cùng 1 websocket
- lưu kết quả chi tiết vào benchmark/ws_benchmark_results.csv
- lưu audio câu dài nhất thành công vào benchmark/ws_longest_query.wav
- lưu text câu dài nhất vào benchmark/ws_longest_query.txt

## 7) Chạy benchmark concurrent (6 WS + 30 request)

```bash
python benchmark/ws_concurrent_6ws_30req_benchmark.py
```

Kịch bản:

1. Mở đúng 6 kết nối websocket.
2. Tạo đúng 30 request và phát đồng thời.
3. Phân phối request theo round-robin vào 6 kết nối.
4. Mỗi kết nối vẫn xử lý tuần tự theo queue phía server.

Output mặc định:

- benchmark/ws_concurrent_6ws_30req_results.csv

## 8) Bảng tổng hợp tối ưu từ các file CSV

Nguồn dữ liệu:

- benchmark/original_ws_benchmark_results.csv
- benchmark/io_result.csv
- benchmark/hyper_param_tuning+io.csv
- benchmark/final_heuristic.csv

Tất cả các file trên đều có 50 mẫu và success rate 100%.
Phần cứng Colab A100 (40GB).

| Profile                                      | TTFB p50 (ms) | TTFB p95 (ms) | TTFB mean (ms) | Total p50 (ms) | Total p95 (ms) | Total mean (ms) | Cải thiện mean TTFB vs baseline | Cải thiện mean Total vs baseline |
| -------------------------------------------- | ------------: | ------------: | -------------: | -------------: | -------------: | --------------: | ------------------------------: | -------------------------------: |
| Baseline (original_ws_benchmark_results.csv) |       1896.02 |       2752.54 |        1919.95 |        1900.86 |        2764.90 |         1925.79 |                           0.00% |                            0.00% |
| IO optimized (io_result.csv)                 |       1466.72 |       2010.24 |        1429.28 |        1472.61 |        2022.19 |         1435.30 |                          25.56% |                           25.47% |
| Hyper-param + IO (hyper_param_tuning+io.csv) |       1305.12 |       1770.43 |        1267.34 |        1311.21 |        1781.88 |         1273.34 |                          33.99% |                           33.88% |
| Final heuristic (final_heuristic.csv)        |       1093.00 |       1597.41 |        1092.70 |        1100.64 |        1610.31 |         1099.19 |                          43.09% |                           42.92% |

## 9) Kiểm tra nhanh server

```bash
curl http://127.0.0.1:8765/health
```

Kỳ vọng:

```json
{ "status": "ok" }
```

## 10) Cấu trúc quan trọng

- vietvoicetts/: model config, preprocessing, inference engine
- benchmark/ws_server.py: websocket inference server
- benchmark/ws_benchmark.py: benchmark tuần tự 50 câu
- benchmark/ws_concurrent_6ws_30req_benchmark.py: benchmark concurrent 6 websocket + 30 request
- pyproject.toml: dependencies và extras
