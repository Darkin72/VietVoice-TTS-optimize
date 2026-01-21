# VietVoice-TTS Optimized (H100 & High-Performance Server)

Phiên bản tối ưu hóa đặc biệt của VietVoice-TTS dành cho các hạ tầng phần cứng mạnh mẽ (NVIDIA A100/H100), tập trung vào tốc độ inference, độ trễ tối thiểu (TTFB) và khả năng scale vô hạn.

## Các cải tiến chính

- **TensorRT & FP16 Acceleration**: Tự động biên dịch và tối ưu hóa model cho GPU NVIDIA Ampere/Hopper, tăng tốc độ inference lên gấp nhiều lần.
- **RAM-only Inference**: Reference voice sample được nạp trực tiếp vào RAM lúc khởi chạy, loại bỏ hoàn toàn I/O disk khi suy luận.
- **I/O Binding**: Giữ các tensor trên VRAM suốt quá trình lặp (Flow-Matching), loại bỏ chi phí sao chép dữ liệu giữa CPU và GPU.
- **Micro-chunking & Streaming**: Chia nhỏ đoạn hội thoại đầu tiên và stream audio binary ngay lập tức qua WebSocket, đạt TTFB (Time To First Byte) cực thấp.
- **FastAPI Lifespan**: Quản lý tài nguyên hiện đại, khởi tạo model một lần duy nhất.

## Cài đặt

Yêu cầu Python 3.8+ và NVIDIA GPU với TensorRT support.

```bash
# Cài đặt các dependency cần thiết
pip install fastapi uvicorn[standard] websockets sounddevice numpy torch
```

## Chạy Server (H100 Optimized)

Server được cấu hình mặc định để tận dụng tối đa sức mạnh phần cứng:

```bash
nohup bash -c "exec -a my_app python -m vietvoicetts" > se.log 2>&1 &
```

kilL: pkill my_app
_Server sẽ lắng nghe tại cổng 8000._

## Sử dụng WebSocket API

Kết nối tới: `ws://<server-ip>:8000/ws/tts`

**Giao thức**:

- **Input**: Gửi trực tiếp văn bản thô (Plain Text).
- **Output**:
  1. Stream các khối dữ liệu binary (Raw PCM int16).
  2. Một chuỗi JSON kết thúc: {"status": "completed", "chunks": X, "time": "Y.Zs"}.

## Client Example (Real-time Playback)

Sử dụng script `client.py` để kết nối và phát âm thanh ngay lập tức:

```bash
# Chạy trên máy cá nhân để nghe trực tiếp từ server
python client.py
```

## Cấu hình Tối ưu

Các tham số trong `ModelConfig` đã được tinh chỉnh cho hiệu suất:

- `use_tensorrt`: True
- `use_io_binding`: True
- `micro_chunking_words`: 5 (Tạo đoạn mồi nhanh)
- `first_chunk_nfe_step`: 16 (Giảm bước tính toán cho TTFB nhanh)

---

_Dự án gốc bởi: [nguyenvulebinh/VietVoice-TTS](https://github.com/nguyenvulebinh/VietVoice-TTS)_
