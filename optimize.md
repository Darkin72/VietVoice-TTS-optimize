# To-Do Tối Ưu Thời Gian Infer VietVoice TTS

Mục tiêu: giảm tối đa latency infer và tăng throughput, ưu tiên các thay đổi có tác động lớn trước.

## 1) Thiết lập baseline và đo đạc bắt buộc

- [ ] Thêm profiler theo stage trong pipeline: preprocess, transformer loop, decode, postprocess.
- [ ] Log p50/p95 latency theo 3 kịch bản: text ngắn, text trung bình, text dài có chunk.
- [ ] Ghi rõ môi trường benchmark: CPU/GPU, số core, RAM, phiên bản onnxruntime, batch cấu hình.
- [ ] Đặt ngưỡng thành công rõ ràng (ví dụ: giảm >= 35% tổng infer time trên cùng phần cứng).

Gợi ý vị trí đo:

- `TTSEngine._run_preprocess`
- `TTSEngine._run_transformer_steps`
- `TTSEngine._run_decode`
- `TTSEngine.synthesize` (tổng)

## 2) Quick wins có tác động lớn nhất

- [ ] Giảm số bước solver `nfe_step` từ 32 xuống 24, 20, 16 rồi A/B test chất lượng.
- [ ] Nếu chất lượng vẫn ổn, đặt profile mặc định `fast` dùng `nfe_step` thấp hơn.
- [ ] Dùng speaker cố định theo session khi không yêu cầu random voice để tránh chọn mẫu ngẫu nhiên và đọc lại metadata không cần thiết.
- [ ] Hạn chế log trong đường infer nóng (giảm print trong loop/chunking ở mode production).
- [ ] Giảm chi phí cross-fade khi chỉ có 1 chunk hoặc overlap quá nhỏ.

Lý do: transformer loop là phần tốn thời gian nhất, giảm số bước thường cho lợi ích lớn nhất.

## 3) Tối ưu model loading và I/O (load sẵn RAM)

- [ ] Giữ engine singleton ở mức process để model chỉ load 1 lần cho nhiều request.
- [ ] Warm-up ngay khi khởi động dịch vụ bằng 1 request giả để tạo cache kernel và allocator.
- [ ] Tránh mở lại tar model nhiều lần để đọc sample: preload `audio_metadata.json` và cache audio mẫu phổ biến vào RAM.
- [ ] Thêm cache cho audio tham chiếu ngoài (key theo hash bytes hoặc path+mtime) sau khi đã resample/normalize.
- [ ] Cân nhắc tách model tar ra thư mục local một lần, load ONNX trực tiếp từ file để giảm overhead tar extract theo request.

Kết quả mong đợi: giảm cold-start và giảm jitter giữa các request.

## 4) ONNX Runtime tuning theo thiết bị

- [ ] Bật ưu tiên GPU triệt để khi có CUDA; xác nhận model thực sự chạy trên CUDA provider.
- [ ] Thử `enable_cuda_graph` (nếu build ORT hỗ trợ) để giảm launch overhead.
- [ ] Tinh chỉnh `intra_op_num_threads` và `inter_op_num_threads` cho CPU-only (benchmark theo từng máy).
- [ ] Đảm bảo session options nhất quán giữa 3 model và không tạo lại session trong runtime.
- [ ] Thử đổi execution mode khi CPU-only để xác nhận `ORT_SEQUENTIAL` có còn tối ưu nhất.

Lưu ý: cấu hình thread tối ưu phụ thuộc số core thực và contention của hệ thống.

## 5) Tối ưu transformer loop

- [ ] Kiểm tra khả năng tăng `fuse_nfe` để giảm số lần gọi runtime (nếu model hỗ trợ nhiều step nội bộ/lần gọi).
- [ ] Tránh tạo lại dict input mỗi vòng lặp; tái sử dụng cấu trúc input và chỉ cập nhật tensor thay đổi.
- [ ] Đảm bảo `noise`, `time_step` không bị ép copy không cần thiết giữa các vòng.
- [ ] Profile chi tiết từng vòng để phát hiện spike (ví dụ ở bước đầu do lazy init).
- [ ] Với workload lặp nhiều request cùng shape, gom theo profile shape để tăng cache hit của runtime.

## 6) Precision và model optimization

- [ ] Tạo biến thể ONNX FP16 cho GPU, benchmark chất lượng và latency.
- [ ] Thử quantization (INT8 hoặc mixed precision) cho CPU path, đo MOS nội bộ và artifact.
- [ ] Chạy tối ưu graph offline (onnx simplifier/optimizer) và so sánh với `ORT_ENABLE_ALL`.
- [ ] Duy trì 2 profile model: `best_quality` và `best_latency` để chọn theo nhu cầu.

## 7) Tối ưu text/chunking để giảm compute

- [ ] Cải thiện thuật toán chunk để giảm số chunk tổng thể (ít chunk hơn => ít lần preprocess/transformer/decode).
- [ ] Ưu tiên câu dài vừa đủ thay vì chia quá nhỏ, vẫn giữ giới hạn `max_chunk_duration`.
- [ ] Tăng `max_chunk_duration` khi phần cứng đủ khỏe để giảm số chunk cho văn bản dài.
- [ ] Cache kết quả `text_to_indices` cho các đoạn text lặp lại trong hệ thống dịch vụ.

## 8) Tối ưu audio processing

- [ ] Cache kết quả load/resample/normalize cho reference audio dùng lặp lại.
- [ ] Tránh chuyển đổi dtype dư thừa trong cross-fade và volume matching.
- [ ] Chỉ chạy xử lý clipping khi thật sự cần (đã có kiểm tra max, giữ nguyên logic này).
- [ ] Nếu pipeline online không cần ghi file, tách đường xử lý trả mảng trực tiếp để bỏ I/O save.

## 9) Concurrency và kiến trúc runtime

- [ ] Chạy mô hình theo kiến trúc worker pool: mỗi worker giữ sẵn session trong RAM.
- [ ] Tách hàng đợi request để tránh tạo/destroy engine liên tục.
- [ ] Nếu nhiều request nhỏ, cân nhắc micro-batching ở mức preprocess/decode khi phù hợp.
- [ ] Khóa tài nguyên phù hợp để không tranh chấp CPU thread pool của ORT.

## 10) Kế hoạch triển khai theo sprint (khuyến nghị)

Sprint 1 (tác động cao, rủi ro thấp):

- [ ] Thêm profiler và benchmark chuẩn.
- [ ] Warm-up + singleton engine + cache sample phổ biến vào RAM.
- [ ] Thử `nfe_step`: 32 -> 24 -> 20.

Sprint 2 (tác động cao, rủi ro trung bình):

- [ ] FP16 GPU hoặc quantization CPU path.
- [ ] Tối ưu thread ORT theo máy đích.
- [ ] Tối ưu chunking để giảm số chunk.

Sprint 3 (nâng cao):

- [ ] Tối ưu sâu transformer loop và shape profile.
- [ ] Worker pool/micro-batching cho production traffic.

## 11) Bảng theo dõi kết quả

| Hạng mục              | Trạng thái | Latency trước | Latency sau | Chất lượng chủ quan | Ghi chú |
| --------------------- | ---------- | ------------: | ----------: | ------------------- | ------- |
| Baseline profiler     | TODO       |             - |           - | -                   |         |
| Warm-up + preload RAM | TODO       |             - |           - | -                   |         |
| Giảm nfe_step         | TODO       |             - |           - | -                   |         |
| ORT thread tuning     | TODO       |             - |           - | -                   |         |
| FP16/INT8             | TODO       |             - |           - | -                   |         |
| Chunking optimization | TODO       |             - |           - | -                   |         |

---

## Mức ưu tiên triển khai ngay

1. Giảm `nfe_step` có kiểm soát chất lượng.
2. Giữ model/session sống lâu trong RAM + warm-up khi start.
3. Cache reference audio và sample phổ biến sau resample/normalize.
4. Tinh chỉnh ORT threads theo phần cứng thực tế.
5. Tối ưu chunking để giảm số chunk tổng.
