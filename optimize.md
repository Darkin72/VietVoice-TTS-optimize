# Optimize – To-Do List giảm thời gian Inference tối đa

Dưới đây là danh sách đầy đủ các hướng tối ưu hóa, được sắp xếp từ tác động cao → thấp và từ dễ triển khai → phức tạp. Mỗi mục đều có mô tả rõ **vấn đề** cần giải quyết và **giải pháp** cụ thể trong context của codebase.

---

## 1. Tối ưu hóa GPU / Tensor Execution

### 1.1 [ ] Triển khai đầy đủ CUDA Graph cho Transformer loop
- **Vấn đề**: `use_cuda_graph=True` đã có trong `ModelConfig` nhưng chưa được implement trong `_run_transformer_steps`. Mỗi bước lặp vẫn có CPU→GPU dispatch overhead.
- **Giải pháp**: Dùng `onnxruntime.CUDAGraph` (ORT ≥ 1.17) hoặc `torch.cuda.graph` để "record" toàn bộ 32 bước lặp transformer thành một đồ thị thực thi. Sau đó chỉ cần `graph.replay()` thay vì gọi `session.run_with_iobinding()` 32 lần.
- **File liên quan**: `vietvoicetts/core/tts_engine.py` → `_run_transformer_steps`

### 1.2 [ ] Chuyển đổi model sang FP16 (Half-Precision)
- **Vấn đề**: Flag `use_fp16=False` đã có trong `ModelConfig` nhưng chưa được dùng. Các model hiện chạy ở FP32. H100 có Tensor Core tốc độ FP16 gấp 2–4× FP32.
- **Giải pháp**: Dùng `onnxconverter_common.convert_float_to_float16()` hoặc `onnxmltools` để convert `transformer.onnx` và `decode.onnx` sang FP16 offline. Khi load, cast các numpy array đầu vào sang `np.float16`.
- **File liên quan**: `vietvoicetts/core/tts_engine.py`, `vietvoicetts/core/model.py`

### 1.3 [ ] Chuyển đổi model sang TensorRT Engine
- **Vấn đề**: ONNX Runtime với CUDA provider vẫn có overhead biên dịch kernel. TensorRT (TRT) tối ưu hóa tốt hơn nhiều cho phần cứng cụ thể.
- **Giải pháp**: Dùng `torch.onnx` → `tensorrt` hoặc `TensorrtExecutionProvider` trong ORT để build engine từ `.onnx`. Cache engine file (`.trt`) vào `model_cache_dir` để không phải compile lại mỗi lần khởi động.
- **File liên quan**: `vietvoicetts/core/model.py` → `_get_optimal_providers`, `_load_models_from_file`

### 1.4 [ ] Sử dụng Pinned Memory (Page-locked) cho CPU↔GPU transfer
- **Vấn đề**: `torch.from_numpy()` trong `_run_transformer_steps` allocate pageable memory, làm chậm DMA transfer.
- **Giải pháp**: Thay bằng `torch.empty(...).pin_memory()` rồi `copy_()` để dùng pinned buffer, cho phép async DMA transfer song song với tính toán GPU.
- **File liên quan**: `vietvoicetts/core/tts_engine.py` → `_run_transformer_steps`

### 1.5 [ ] Giảm số bước `nfe_step` với Consistency/Distillation model
- **Vấn đề**: 32 bước lặp là bottleneck chính. Mỗi bước = 1 lần chạy Transformer.
- **Giải pháp**: Thử nghiệm `nfe_step=8` hoặc `nfe_step=16` cho toàn bộ, không chỉ chunk đầu. Nếu chất lượng chưa đủ, xem xét distill model Flow-Matching về dạng Consistency Model (1–4 bước).

---

## 2. Tối ưu hóa Pipeline & Concurrency

### 2.1 [ ] Cache kết quả Preprocess cho giọng mặc định (RAM)
- **Vấn đề**: `_run_preprocess(audio, text_ids, max_duration)` được gọi lại mỗi request kể cả khi dùng giọng mặc định từ RAM (`cached_ref_audio`). Một phần output (`rope_cos_q`, `rope_sin_q`, `rope_cos_k`, `rope_sin_k`) phụ thuộc vào `max_duration` (thay đổi theo độ dài chunk), nhưng `cat_mel_text_drop` (reference audio embedding) hoàn toàn cố định.
- **Giải pháp**: Tách phần reference-only embedding ra khỏi preprocess và cache riêng. Cụ thể, nếu ONNX model cho phép tách, pre-compute và lưu `ref_embedding` tại `load_models()`. Nếu không tách được, cache một bảng lookup `{max_duration_bucket: preprocess_outputs}` cho các giá trị `max_duration` phổ biến (ví dụ: 5 buckets từ 3–20s) để tái sử dụng ở các request tương tự.
- **File liên quan**: `vietvoicetts/core/model.py`, `vietvoicetts/core/tts_engine.py`

### 2.2 [ ] Pipeline gối đầu (Async Prefetch): Preprocess chunk N+1 trong khi GPU chạy Transformer cho chunk N
- **Vấn đề**: CPU và GPU hiện chạy tuần tự: Preprocess → Transformer → Decode → Preprocess → ...
- **Giải pháp**: Dùng `concurrent.futures.ThreadPoolExecutor` để preprocess chunk tiếp theo trên luồng CPU riêng, song song với lúc GPU đang chạy transformer cho chunk hiện tại.
- **File liên quan**: `vietvoicetts/core/tts_engine.py` → `synthesize_stream`

### 2.3 [ ] Warm-up Inference khi khởi động server
- **Vấn đề**: Lần inference đầu tiên sau khi load model thường chậm hơn do CUDA kernel compilation (JIT cuDNN) và GPU cache miss.
- **Giải pháp**: Sau khi load model xong, chạy một dummy inference với tensor shape ngắn để warm-up tất cả CUDA kernels trước khi nhận request thực.
- **File liên quan**: `vietvoicetts/server.py` → lifespan context manager

### 2.4 [ ] Persistent CUDA Streams qua ONNX Runtime IOBinding
- **Vấn đề**: Các lần `session.run()` dùng default CUDA stream nội bộ của ORT, không thể override bằng `torch.cuda.Stream()` trực tiếp vì ORT quản lý stream riêng.
- **Giải pháp**: Dùng `IOBinding` của ORT cùng với `bind_input(..., device_type="cuda")` và `bind_output(...)` để giữ data trên VRAM liên tục. Sau đó gọi `session.run_with_iobinding(io_binding)` và truyền cùng một `io_binding` object qua các bước lặp (đã implemented một phần). Để overlap I/O và compute, set `do_copy_in_default_stream=False` trong `CUDAExecutionProvider` options và allocate output buffer trước khi gọi run.
- **File liên quan**: `vietvoicetts/core/tts_engine.py`, `vietvoicetts/core/model.py` → `_get_optimal_providers`

---

## 3. Tối ưu hóa Memory

### 3.1 [ ] Pre-allocate numpy/torch buffers cố định (per-request)
- **Vấn đề**: Mỗi bước lặp Transformer `session.run()` allocate numpy array mới cho `noise` và `time_step`, tạo GC pressure. Nếu dùng shared buffer toàn cục, sẽ xảy ra race condition khi có nhiều concurrent requests.
- **Giải pháp**: Pre-allocate buffer pool theo `max_chunk_audio_len` tại thời điểm bắt đầu mỗi request (không phải toàn cục). Dùng context manager để checkout/return buffer từ pool với lock. Với I/O Binding hiện tại, giữ `noise_tensor_a`/`noise_tensor_b` là CUDA tensor được tạo một lần mỗi request (như đang làm) thay vì mỗi bước lặp.
- **File liên quan**: `vietvoicetts/core/tts_engine.py` → `_run_transformer_steps`

### 3.2 [ ] Tải tất cả voice samples vào RAM (không chỉ sample mặc định)
- **Vấn đề**: Khi request dùng `gender`/`area`/`emotion` cụ thể, `select_sample()` mở lại file `.tar` từ disk để đọc audio → I/O disk latency.
- **Giải pháp**: Tại `load_models()`, đọc toàn bộ audio bytes của tất cả samples vào dictionary `{file_name: bytes}` trong RAM. Thay thế `tarfile.open()` trong `select_sample()` bằng dict lookup.
- **File liên quan**: `vietvoicetts/core/model.py` → `_load_models_from_file`, `select_sample`

### 3.3 [ ] Cache ONNX model bytes để tránh re-read từ tar
- **Vấn đề**: Nếu server restart, `_load_models_from_file` phải đọc lại bytes từ `.tar` (slow tar decompression).
- **Giải pháp**: Sau lần extract đầu tiên, lưu `preprocess.onnx`, `transformer.onnx`, `decode.onnx` trực tiếp ra `model_cache_dir` thay vì chỉ load vào memory. Lần sau load thẳng từ file `.onnx` đã giải nén.
- **File liên quan**: `vietvoicetts/core/model.py` → `_load_models_from_file`

---

## 4. Tối ưu hóa Text & Audio Processing

### 4.1 [ ] Precompile regex patterns trong TextProcessor
- **Vấn đề**: `re.sub(...)` trong `clean_text()` và `chunk_text()` compile regex mới mỗi lần gọi.
- **Giải pháp**: Compile tất cả pattern một lần trong `__init__` và lưu vào instance variable: `self._re_invalid = re.compile(...)`, `self._re_pause = re.compile(...)`, v.v.
- **File liên quan**: `vietvoicetts/core/text_processor.py`

### 4.2 [ ] Thay thế pydub bằng soundfile/librosa cho audio loading
- **Vấn đề**: `pydub` dùng `ffmpeg` subprocess để load audio, có overhead process spawn.
- **Giải pháp**: Dùng `soundfile.read()` hoặc `librosa.load()` trực tiếp trong `AudioProcessor.load_audio()` (nhanh hơn vì pure Python/C binding, không spawn process).
- **File liên quan**: `vietvoicetts/core/audio_processor.py` → `load_audio`

### 4.3 [ ] Tối ưu hóa cross-fade computation với NumPy vectorization
- **Vấn đề**: Cross-fade tính `fade_out`/`fade_in` array và nhiều phép toán `astype()` cho mỗi chunk join.
- **Giải pháp**: Precompute fade curves với shape cố định tại khởi tạo. Dùng `np.multiply(..., out=buffer)` với pre-allocated buffer để tránh tạo array tạm.
- **File liên quan**: `vietvoicetts/core/tts_engine.py` → `synthesize_stream`, `vietvoicetts/core/audio_processor.py`

---

## 5. Tối ưu hóa Server & Network

### 5.1 [ ] Xử lý concurrent requests với async thread pool
- **Vấn đề**: Inference trong `synthesize_stream` là CPU/GPU-bound, blocking event loop của FastAPI nếu chạy trực tiếp trong coroutine.
- **Giải pháp**: Wrap inference trong `asyncio.get_event_loop().run_in_executor(executor, ...)` với `ThreadPoolExecutor` để không block event loop. Điều này cho phép nhiều WebSocket connection xử lý đồng thời.
- **File liên quan**: `vietvoicetts/server.py` → `websocket_endpoint`

### 5.2 [ ] Tăng kích thước WebSocket send buffer
- **Vấn đề**: `await websocket.send_bytes()` gọi mỗi chunk nhỏ → nhiều syscall network.
- **Giải pháp**: Gom các chunk nhỏ liên tiếp thành buffer đủ lớn (ví dụ 4096 samples ~170ms audio) trước khi send, giảm số lần syscall.
- **File liên quan**: `vietvoicetts/server.py` → `websocket_endpoint`

### 5.3 [ ] Thêm HTTP/2 hoặc gRPC streaming thay thế WebSocket
- **Vấn đề**: WebSocket có overhead handshake và frame parsing.
- **Giải pháp**: Cân nhắc dùng gRPC server-streaming (bidirectional) với `grpc.aio` cho latency thấp hơn, đặc biệt trong môi trường internal microservices.

---

## 6. Profiling & Đo lường

### 6.1 [ ] Thêm timer chi tiết cho từng bước inference
- **Giải pháp**: Thêm `time.perf_counter()` để đo riêng thời gian của: preprocess, mỗi bước transformer, decode, cross-fade, network send. Log kết quả để xác định bottleneck thực sự.
- **File liên quan**: `vietvoicetts/core/tts_engine.py`

### 6.2 [ ] Profile với NVIDIA Nsight Systems
- **Giải pháp**: Chạy `nsys profile python -m vietvoicetts` để xem timeline GPU/CPU, phát hiện idle time, memory transfer bottleneck.

### 6.3 [ ] Benchmark thay đổi `nfe_step` vs chất lượng audio (MOS/PESQ)
- **Giải pháp**: Chạy MOS evaluation với `nfe_step` = 4, 8, 16, 24, 32 để tìm điểm cân bằng tối ưu giữa tốc độ và chất lượng cho use case cụ thể.

---

## Tóm tắt ưu tiên triển khai

| # | Hạng mục | Độ khó | Tác động dự kiến |
|---|----------|--------|-----------------|
| 2.1 | Cache preprocess output cho giọng mặc định | Thấp | **Rất cao** (bỏ hoàn toàn bước Preprocess cho default voice) |
| 3.2 | Load toàn bộ samples vào RAM | Thấp | Cao (loại bỏ I/O disk) |
| 2.3 | Warm-up inference khi khởi động | Thấp | Trung bình (giảm latency request đầu tiên) |
| 1.2 | Chuyển model sang FP16 | Trung bình | **Rất cao** (2–4× faster Transformer) |
| 1.1 | Triển khai CUDA Graph đầy đủ | Trung bình | Cao (giảm CPU dispatch overhead 32 bước) |
| 2.2 | Pipeline gối đầu Preprocess/Transformer | Trung bình | Cao (GPU utilization ~100%) |
| 4.1 | Precompile regex | Thấp | Thấp (tiết kiệm vài ms) |
| 3.3 | Cache ONNX files ra disk | Thấp | Trung bình (giảm startup time) |
| 1.3 | Chuyển sang TensorRT | Cao | **Rất cao** (2–5× faster vs ORT+CUDA) |
| 5.1 | Async thread pool cho concurrent requests | Trung bình | Cao (scale throughput) |
