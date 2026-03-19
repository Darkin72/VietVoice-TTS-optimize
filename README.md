# VietVoice TTS WebSocket Server + Benchmark

Repository này hiện tập trung vào 3 phần:

- Model inference code: `vietvoicetts/`
- WebSocket server: `benchmark/ws_server.py`
- Benchmark client: `benchmark/ws_benchmark.py`

## 1) Yêu cầu môi trường

- Python 3.8+
- Windows/Linux/macOS
- Khuyến nghị dùng virtual environment

Luu y: `pydub` can `ffmpeg` de doc nhieu dinh dang audio. Neu may ban chua co `ffmpeg`, hay cai dat truoc.

## 2) Cai dat

Tai thu muc goc project, cai dat editable mode:

### CPU

```bash
pip install -e ".[cpu,server,benchmark]"
```

### GPU (CUDA)

```bash
pip install -e ".[gpu,server,benchmark]"
```

## 3) Chay WebSocket server infer

```bash
python benchmark/ws_server.py --host 127.0.0.1 --port 8765
```

Server se:

- Preload model ngay khi app startup (lifespan)
- Chay warm-up infer 1 cau ngan de request dau tien infer ngay
- Mo endpoint health check: `GET /health`
- Mo websocket endpoint: `ws://127.0.0.1:8765/ws`

### Tham so server thuong dung

```bash
python benchmark/ws_server.py \
  --host 127.0.0.1 \
  --port 8765 \
  --nfe-step 32 \
  --fuse-nfe 1 \
  --speed 1.0 \
  --chunk-size 32768
```

## 4) Protocol WebSocket (bytes)

Server dung binary protocol:

1. Client gui 1 binary frame chua `text.encode("utf-8")`
2. Server infer va stream audio WAV bytes bang nhieu binary frame
3. Server gui binary frame rong `b""` de danh dau ket thuc response

Neu loi:

- Server gui binary frame bat dau bang `b"ERR:" + message`
- Sau do van gui `b""` de dong response

## 5) Chay benchmark client

Benchmark chi bat dau khi websocket connect thanh cong.

```bash
python benchmark/ws_benchmark.py \
  --url ws://127.0.0.1:8765/ws \
  --min-words 3 \
  --max-words 100 \
  --samples-per-length 2 \
  --output-csv benchmark/ws_benchmark_results.csv
```

Mac dinh benchmark se:

- Tao danh sach text tu 3 den 100 tu
- Moi do dai co `samples-per-length` cau
- Gui lan luot tung query tren cung websocket connection

## 6) Metric benchmark

Moi query co cac metric:

- `ttfb_ms`: time to first byte
- `total_ms`: tong thoi gian request-response
- `response_bytes`: tong byte WAV nhan duoc
- `throughput_kib_s`: toc do nhan du lieu
- `audio_duration_s`: thoi luong audio WAV
- `rtf`: real-time factor = `total_time / audio_duration` (cang thap cang tot)

Tong hop cuoi dot benchmark:

- p50/p95/mean cho TTFB
- p50/p95/mean cho Total latency
- mean throughput
- p50/p95/mean cho RTF

## 7) Kiem tra nhanh server

Sau khi chay server, test health:

```bash
curl http://127.0.0.1:8765/health
```

Ky vong:

```json
{ "status": "ok" }
```

## 8) Thu muc quan trong

- `vietvoicetts/`: model config, preprocessing, infer engine
- `benchmark/ws_server.py`: websocket inference server
- `benchmark/ws_benchmark.py`: websocket benchmark client
- `pyproject.toml`: dependencies va extras

## 9) Ghi chu hieu nang

- `nfe-step` la knob quan trong nhat cho trade-off chat luong/latency
- Neu benchmark CPU-only, thu tune `--inter-op-threads` va `--intra-op-threads`
- Server da preload + warm-up startup de request dau tien khong bi cold start lon
