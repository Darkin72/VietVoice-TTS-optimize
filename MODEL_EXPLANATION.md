# Giải Thích Chi Tiết Cơ Chế Hoạt Động của VietVoice TTS

## Mục lục

1. [Tổng quan kiến trúc](#1-tổng-quan-kiến-trúc)
2. [Pipeline 3 giai đoạn](#2-pipeline-3-giai-đoạn)
3. [Giai đoạn 1 – Preprocessing (Tiền xử lý)](#3-giai-đoạn-1--preprocessing-tiền-xử-lý)
4. [Giai đoạn 2 – Transformer Flow Matching (Sinh mel spectrogram)](#4-giai-đoạn-2--transformer-flow-matching-sinh-mel-spectrogram)
5. [Giai đoạn 3 – Decode (Giải mã thành sóng âm)](#5-giai-đoạn-3--decode-giải-mã-thành-sóng-âm)
6. [Xử lý văn bản (TextProcessor)](#6-xử-lý-văn-bản-textprocessor)
7. [Xử lý âm thanh (AudioProcessor)](#7-xử-lý-âm-thanh-audioprocessor)
8. [Chunking và xử lý văn bản dài](#8-chunking-và-xử-lý-văn-bản-dài)
9. [Cross-fade: Ghép các đoạn âm thanh](#9-cross-fade-ghép-các-đoạn-âm-thanh)
10. [Toàn bộ luồng dữ liệu từ đầu đến cuối](#10-toàn-bộ-luồng-dữ-liệu-từ-đầu-đến-cuối)
11. [Tổng hợp các công thức quan trọng](#11-tổng-hợp-các-công-thức-quan-trọng)

---

## 1. Tổng quan kiến trúc

VietVoice TTS là hệ thống **tổng hợp giọng nói tiếng Việt zero-shot** (không cần train lại) dựa trên kiến trúc **F5-TTS** kết hợp **Flow Matching** và **Diffusion Transformer (DiT)**. Hệ thống có khả năng **nhân bản giọng nói** (voice cloning) bằng cách chỉ cung cấp một đoạn audio tham chiếu ngắn (~5–15 giây) cùng transcript tương ứng.

### Các thành phần chính

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          VietVoice TTS System                            │
│                                                                           │
│  Input: Văn bản cần đọc + Audio tham chiếu + Transcript tham chiếu      │
│                                                                           │
│  ┌──────────────┐    ┌───────────────────────┐    ┌──────────────────┐  │
│  │  TextProces- │    │     TTSEngine          │    │  AudioProcessor  │  │
│  │  sor         │───▶│  (Điều phối pipeline)  │───▶│  (Hậu xử lý     │  │
│  │  (Tokenize)  │    │                        │    │   âm thanh)      │  │
│  └──────────────┘    └───────────────────────┘    └──────────────────┘  │
│                              │                                            │
│                    ┌─────────▼──────────┐                                │
│                    │  ModelSessionMgr   │                                 │
│                    │  (3 ONNX models)   │                                 │
│                    │  ┌─────────────┐   │                                 │
│                    │  │ preprocess  │   │                                 │
│                    │  │ .onnx       │   │                                 │
│                    │  ├─────────────┤   │                                 │
│                    │  │ transformer │   │                                 │
│                    │  │ .onnx       │   │                                 │
│                    │  ├─────────────┤   │                                 │
│                    │  │ decode      │   │                                 │
│                    │  │ .onnx       │   │                                 │
│                    │  └─────────────┘   │                                 │
│                    └────────────────────┘                                 │
│                                                                           │
│  Output: File WAV chất lượng cao ở 24kHz                                 │
└─────────────────────────────────────────────────────────────────────────┘
```

### Luồng tổng quát

```
Văn bản                  Audio ref
    │                        │
    ▼                        ▼
[Làm sạch]            [Load & Normalize]
[Tokenize]            [Reshape → (1,1,L)]
    │                        │
    └──────────┬─────────────┘
               ▼
        [PREPROCESS.ONNX]
        ─ Mel spectrogram ref
        ─ Noise ngẫu nhiên X₀
        ─ Embedding văn bản
        ─ RoPE cos/sin
               │
               ▼
    [TRANSFORMER.ONNX × 32 bước]   ← Flow Matching ODE
        ─ Classifier-Free Guidance
        ─ RoPE positional encoding
        ─ Multi-head self-attention
               │
               ▼
        [DECODE.ONNX]
        ─ Vocoder (mel → waveform)
               │
               ▼
          Audio WAV 24kHz
```

---

## 2. Pipeline 3 giai đoạn

Toàn bộ quá trình sinh giọng nói được chia làm **3 giai đoạn** tương ứng với 3 file model ONNX:

| Giai đoạn | File ONNX | Vai trò |
|---|---|---|
| **1. Preprocess** | `preprocess.onnx` | Trích xuất mel spectrogram, khởi tạo nhiễu, chuẩn bị conditioning |
| **2. Transformer** | `transformer.onnx` | Lặp lại 32 lần, dần dần "chưng cất" nhiễu thành mel spectrogram theo phương pháp Flow Matching |
| **3. Decode** | `decode.onnx` | Chuyển mel spectrogram thành sóng âm thực (Vocoder) |

---

## 3. Giai đoạn 1 – Preprocessing (Tiền xử lý)

### 3.1 Tải và chuẩn hóa audio tham chiếu

**Đầu vào:** File audio (.wav, .m4a, .mp3, ...)

**Bước 1 – Mono & Resample:**
```
AudioSegment.from_file(path)
    .set_channels(1)          ← chuyển về mono
    .set_frame_rate(24000)    ← resample về 24kHz
```

**Bước 2 – Chuẩn hóa về dải int16:**

Loại bỏ DC offset (offset trung bình):
```
audio = audio - mean(audio)
```

Tính hệ số tỉ lệ sao cho giá trị lớn nhất đạt 90% dải int16 (để tránh clipping):
```
scaling_factor = 29491.0 / max(|audio|)
```
> **Tại sao 29491?** Vì 32767 × 0.9 = 29490.3 ≈ 29491. Dải int16 là [−32768, 32767]. Dùng 90% tạo ra headroom, giảm nguy cơ clipping.

```
audio_normalized = audio × scaling_factor   (cast sang int16)
```

**Bước 3 – Reshape:** Chuyển từ vector 1D sang tensor 3D để phù hợp với đầu vào model:
```
audio.reshape(1, 1, L)    ← (batch=1, channel=1, samples=L)
```

---

### 3.2 Tính toán thời lượng (Duration Estimation)

Đây là cơ chế **ước lượng tốc độ đọc** từ audio tham chiếu và áp dụng cho văn bản cần tổng hợp.

**Bước 1 – Tính độ dài văn bản tham chiếu** (tính theo bytes UTF-8, dấu câu ngừng nghỉ được tính trọng số × 3):

```
ref_text_len = len(ref_text.encode('utf-8')) + 3 × count(pause_punctuation in ref_text)
```

> **Lý do dùng UTF-8 bytes thay vì số ký tự:** Tiếng Việt dùng ký tự đa byte (mỗi ký tự có dấu tốn 2–3 bytes), nên độ dài bytes tương quan tốt hơn với thời gian đọc thực tế.
> 
> **Dấu câu ngừng nghỉ** (., , ? ! :) được nhân trọng số × 3 vì chúng gây ra khoảng dừng trong tiếng nói, làm tăng thời gian thực tế.

**Bước 2 – Tính tốc độ đọc:**

```
ref_audio_duration = L / sample_rate    (giây)
speaking_rate = ref_text_len / ref_audio_duration    (bytes/giây)
```

**Bước 3 – Ước lượng thời lượng audio đích:**

```
target_text_len = calculate_text_length(target_text)
target_audio_duration = max(
    target_text_len / speaking_rate / speed,
    min_target_duration    ← 1.0 giây (tránh audio quá ngắn)
)
```

Tham số `speed` (mặc định = 1.0) điều chỉnh tốc độ nói: `speed=1.2` → nhanh hơn 20%.

**Bước 4 – Tính chiều dài frame mel:**

```
ref_audio_len  = L // hop_length + 1
target_audio_samples = int(target_audio_duration × sample_rate)
target_audio_len = target_audio_samples // hop_length + 1
chunk_audio_len = ref_audio_len + target_audio_len
```

> `hop_length = 256` là bước nhảy giữa các frame STFT. Công thức `L // hop_length + 1` cho biết số frame mel tương ứng với chuỗi âm thanh dài `L` mẫu.

---

### 3.3 Xây dựng tensor đầu vào cho Preprocess

**Input 1 – `audio`:** Tensor `(1, 1, L)` – waveform int16 đã chuẩn hóa  
**Input 2 – `text_ids`:** Tensor `(1, T)` – dãy chỉ số ký tự  
**Input 3 – `max_duration`:** Tensor `[chunk_audio_len]` – số frame mel tối đa cần sinh

**Output của Preprocess:**

| Output | Ký hiệu | Ý nghĩa |
|---|---|---|
| `noise` | $X_0$ | Nhiễu Gaussian ngẫu nhiên, shape `(1, N, d_mel)` |
| `rope_cos_q`, `rope_sin_q` | $\cos\theta_q$, $\sin\theta_q$ | RoPE embeddings cho Query |
| `rope_cos_k`, `rope_sin_k` | $\cos\theta_k$, $\sin\theta_k$ | RoPE embeddings cho Key |
| `cat_mel_text` | $c$ | Điều kiện đầy đủ (mel ref + embedding text) |
| `cat_mel_text_drop` | $\emptyset$ | Điều kiện bỏ trống (dùng cho CFG) |
| `ref_signal_len` | $L_{ref}$ | Độ dài (frames) của audio tham chiếu |

**Cụ thể preprocess thực hiện bên trong:**

1. **Mel Spectrogram từ audio tham chiếu:**
   ```
   mel_ref = mel_spectrogram(audio)    ← shape (1, n_mels, T_ref)
   ```

2. **Khởi tạo nhiễu Gaussian** (đây là điểm xuất phát của quá trình Flow Matching):
   ```
   X₀ ~ N(0, I)    ← shape (1, T_total, d_mel)
   ```
   Trong đó `T_total = T_ref + T_target` là tổng số frame mel.

3. **Embedding văn bản:** Chuyển `text_ids` thành biểu diễn véc-tơ thông qua lớp embedding, sau đó ghép với mel spectrogram tham chiếu tạo thành tensor điều kiện `cat_mel_text`.

4. **Tính RoPE embeddings:** Precompute cos/sin cho từng vị trí trong chuỗi.

---

## 4. Giai đoạn 2 – Transformer Flow Matching (Sinh mel spectrogram)

Đây là **trái tim** của toàn bộ hệ thống. Bước này dùng **Flow Matching** kết hợp với **Diffusion Transformer (DiT)** để "chưng cất" nhiễu ngẫu nhiên thành mel spectrogram.

---

### 4.1 Flow Matching là gì?

**Flow Matching** là phương pháp sinh dữ liệu bằng cách học một **trường véc-tơ** (vector field) $v_\theta(X_t, t)$ ánh xạ phân phối nhiễu $p_0 = \mathcal{N}(0, I)$ sang phân phối dữ liệu thực $p_1$ thông qua một ODE (phương trình vi phân thường).

**ODE cần giải:**
$$\frac{dX_t}{dt} = v_\theta(X_t, t, c)$$

Trong đó:
- $X_t$ – trạng thái tại thời điểm $t \in [0, 1]$
- $v_\theta$ – mạng neural (Transformer) dự đoán hướng di chuyển
- $c$ – điều kiện (conditioning): mel ref + embedding văn bản
- $t = 0$ → xuất phát từ nhiễu thuần túy
- $t = 1$ → đến mel spectrogram mục tiêu

**Đường thẳng (Linear Flow Path):**

Không giống DDPM (Denoising Diffusion Probabilistic Models) dùng lịch trình noise phức tạp, Flow Matching dùng **đường thẳng** nối điểm xuất phát $X_0 \sim \mathcal{N}(0,I)$ và dữ liệu thực $X_1$ (mel spectrogram):

$$X_t = (1 - t) \cdot X_0 + t \cdot X_1$$

Trường véc-tơ mục tiêu (ground truth) tương ứng:

$$u_t(X_t \mid X_1) = \frac{X_1 - X_t}{1 - t} = X_1 - X_0$$

> **Nhận xét quan trọng:** Với đường thẳng, hướng di chuyển là **hằng số** theo $t$ – không đổi dọc theo quỹ đạo. Điều này làm cho phương pháp này hiệu quả hơn về mặt tính toán so với DDPM.

**Hàm mục tiêu huấn luyện (Conditional Flow Matching Loss):**
$$\mathcal{L}_{CFM} = \mathbb{E}_{t, X_0, X_1} \left[ \| v_\theta(X_t, t, c) - u_t(X_t \mid X_1) \|^2 \right]$$

---

### 4.2 Numerical ODE Solver – Euler Method

Tại **thời điểm inference**, ta giải ODE bằng phương pháp **Euler** với `nfe_step = 32` bước:

$$X_{t+\Delta t} = X_t + \Delta t \cdot v_\theta(X_t, t, c)$$

Trong đó:
- $\Delta t = 1 / \text{nfe\_step} = 1/32$
- $t_i = i / \text{nfe\_step}$, với $i = 0, 1, \ldots, 30$ (bước cuối là bước 31)

**Vòng lặp trong code** (từ `tts_engine.py`):
```python
for i in tqdm(range(0, self.config.nfe_step - 1, self.config.fuse_nfe),
              total=self.config.nfe_step // self.config.fuse_nfe - 1):
    noise, time_step = session.run(output_names, inputs)
```

`fuse_nfe = 1` nghĩa là mỗi lần gọi model thực hiện **1 bước Euler**. Vòng lặp chạy `31` lần (từ bước 0 đến bước 30 với bước cuối là 31 được xử lý nội bộ).

Tại mỗi bước, model nhận:
- `noise` ($X_t$): trạng thái hiện tại
- `time_step` ($t$): thời điểm hiện tại
- `rope_cos_q/sin_q`, `rope_cos_k/sin_k`: positional encoding cố định
- `cat_mel_text` ($c$): điều kiện đầy đủ
- `cat_mel_text_drop` ($\emptyset$): điều kiện trống (cho CFG)

Và trả về:
- `noise` mới ($X_{t+\Delta t}$): trạng thái sau bước Euler
- `time_step` mới ($t + \Delta t$)

---

### 4.3 Classifier-Free Guidance (CFG)

**Classifier-Free Guidance** là kỹ thuật quan trọng giúp model bám sát điều kiện (văn bản + giọng nói tham chiếu) khi sinh âm thanh.

**Ý tưởng:** Chạy model song song với **2 điều kiện**:
1. **Có điều kiện** ($c$ = mel ref + text): `cat_mel_text` → cho ra $v_\theta(X_t, t, c)$
2. **Không có điều kiện** ($\emptyset$): `cat_mel_text_drop` → cho ra $v_\theta(X_t, t, \emptyset)$

**Kết hợp theo công thức CFG:**
$$v_{cfg}(X_t, t) = v_\theta(X_t, t, \emptyset) + \alpha \cdot \big(v_\theta(X_t, t, c) - v_\theta(X_t, t, \emptyset)\big)$$

Trong đó $\alpha$ là **CFG scale** (hệ số hướng dẫn). Giá trị $\alpha > 1$ làm tăng độ "trung thành" với điều kiện nhưng giảm đa dạng.

> **Tại sao cần CFG?** Nếu chỉ dùng $v_\theta(X_t, t, c)$, model có thể không bám đủ chặt vào văn bản đầu vào. CFG "kéo" kết quả theo hướng điều kiện mạnh hơn, cải thiện độ tự nhiên và độ chính xác.

**Trong code:** `cat_mel_text` và `cat_mel_text_drop` được truyền đồng thời vào transformer, và model nội bộ tự thực hiện tính toán CFG này.

---

### 4.4 Kiến trúc Diffusion Transformer (DiT)

**Transformer block cơ bản:**

Mỗi block trong transformer gồm:

**a) Multi-Head Self-Attention với RoPE:**

Attention tiêu chuẩn:
$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

Với $d_k$ là chiều của Key/Query, và $1/\sqrt{d_k}$ là hệ số chuẩn hóa tránh gradient vanishing khi $d_k$ lớn.

**b) Rotary Position Embedding (RoPE):**

RoPE mã hóa thông tin vị trí **trực tiếp** vào véc-tơ Query và Key thay vì cộng thêm positional embedding.

Cho véc-tơ tại vị trí $m$, chiều $2i$ và $2i+1$ (cặp đôi liên tiếp):

$$q_m^{(2i)} \leftarrow q_m^{(2i)} \cos(m\theta_i) - q_m^{(2i+1)} \sin(m\theta_i)$$
$$q_m^{(2i+1)} \leftarrow q_m^{(2i)} \sin(m\theta_i) + q_m^{(2i+1)} \cos(m\theta_i)$$

Tương tự cho $k_m$. Tần số $\theta_i$ được tính theo:
$$\theta_i = \frac{1}{10000^{2i/d}}$$

**Tính chất quan trọng của RoPE:** Tích vô hướng sau khi áp dụng RoPE chỉ phụ thuộc vào **hiệu vị trí** $(m - n)$, không phụ thuộc vị trí tuyệt đối:
$$\langle \text{RoPE}(q_m), \text{RoPE}(k_n) \rangle = f(q, k, m-n)$$

Điều này giúp model học được tính chất tương đối (vị trí này cách vị trí kia bao nhiêu frame) thay vì tuyệt đối.

**Trong code:** `rope_cos_q`, `rope_sin_q`, `rope_cos_k`, `rope_sin_k` được tính 1 lần trong preprocess rồi dùng lại ở tất cả 32 bước transformer.

**c) Feed-Forward Network (FFN):**
$$\text{FFN}(x) = \text{GELU}(xW_1 + b_1) W_2 + b_2$$

**d) AdaLN (Adaptive Layer Normalization):**

Thay vì LayerNorm thông thường, DiT dùng AdaLN để tích hợp thông tin thời gian $t$:
$$\text{AdaLN}(x, t) = \gamma(t) \cdot \frac{x - \mu}{\sigma} + \beta(t)$$

Trong đó $\gamma(t)$ và $\beta(t)$ là các tham số được học từ embedding thời gian $t$.

---

### 4.5 Time Embedding

Thời gian $t \in [0,1]$ được mã hóa thành véc-tơ thông qua **sinusoidal embedding**:

$$\text{emb}(t)_j = \begin{cases} \sin\!\left(t \cdot 10000^{-j/d}\right) & \text{nếu } j \text{ chẵn} \\ \cos\!\left(t \cdot 10000^{-(j-1)/d}\right) & \text{nếu } j \text{ lẻ} \end{cases}$$

Sau đó đi qua MLP để tạo ra các tham số cho AdaLN.

---

### 4.6 Toàn bộ luồng Transformer – sơ đồ chi tiết

```
X_t (noisy mel, shape: (1, T_total, d_mel))
        │
        ▼
 ┌──────────────┐
 │  Linear proj │  ← project vào d_model chiều
 └──────┬───────┘
        │
        ▼
 ┌──────────────────────────────────────────────────────┐
 │              DiT Block × N_layers                     │
 │                                                        │
 │  ┌────────────────────────────────────────────────┐   │
 │  │  AdaLN(x, t_emb)                               │   │
 │  │  Multi-Head Self-Attention + RoPE               │   │
 │  │  Cross-Attention với conditioning c             │   │
 │  │  FFN (GELU activation)                          │   │
 │  │  Residual connection                            │   │
 │  └────────────────────────────────────────────────┘   │
 └──────────────────────────────────────────────────────┘
        │
        ▼
 ┌──────────────┐
 │  Linear proj │  ← project về d_mel chiều
 └──────┬───────┘
        │
        ▼
  v_θ(X_t, t, c)   ← vector field prediction
        │
        ▼
  X_{t+Δt} = X_t + Δt × v_θ    ← Euler step
```

---

## 5. Giai đoạn 3 – Decode (Giải mã thành sóng âm)

### 5.1 Từ mel spectrogram đến waveform

Sau 32 bước Flow Matching, ta có:
$$X_1 \approx \text{mel spectrogram đích}$$
Shape: `(1, T_total, d_mel)`

Model `decode.onnx` thực hiện vai trò **Vocoder** – chuyển mel spectrogram thành sóng âm thực.

**Đầu vào:**
- `noise` ($X_1$): mel spectrogram đã sinh, shape `(1, T_total, d_mel)`
- `ref_signal_len` ($L_{ref}$): số frame mel của audio tham chiếu

**Xử lý nội bộ của decode:**

Vocoder chỉ lấy **phần target** (loại bỏ phần tham chiếu):
```
mel_target = X₁[:, L_ref:, :]    ← chỉ lấy phần mới sinh ra
```

Sau đó áp dụng **Vocoder** (thường là HiFi-GAN hoặc BigVGAN) để chuyển từng frame mel thành đoạn âm thanh:

**Mel Spectrogram được tính từ STFT:**
$$S(n, k) = \left| \sum_{m=0}^{N-1} x(m + nH) \cdot w(m) \cdot e^{-j2\pi km/N} \right|^2$$

Trong đó:
- $N$ = FFT size (thường 1024)
- $H$ = hop length = 256
- $w(m)$ = cửa sổ Hann/Hamming

**Mel Filterbank:** Nhân ma trận filterbank mel $M \in \mathbb{R}^{n\_mels \times (N/2+1)}$ với phổ năng lượng:
$$\text{mel}(n) = M \cdot S(n, :)$$

Áp dụng log để tuyến tính hóa thang độ:
$$\text{log\_mel}(n) = \log(\text{mel}(n) + \epsilon)$$

Vocoder (HiFi-GAN/BigVGAN) **đảo ngược** quá trình này – từ log mel spectrogram sinh ra waveform chất lượng cao.

**Output:** Tensor 1D waveform ở 24kHz.

---

## 6. Xử lý văn bản (TextProcessor)

### 6.1 Làm sạch văn bản

Chỉ giữ lại các ký tự hợp lệ:
- Bảng chữ cái tiếng Anh (a-z, A-Z, 0-9)
- Ký tự tiếng Việt có dấu
- Dấu câu: `. , ! ? ' @ $ % & / : ; ( ) ` và dấu cách

**Quy trình làm sạch** (theo thứ tự):
1. Nếu có ký tự xuống dòng `\n`: chia thành đoạn, mỗi đoạn thêm `.` ở cuối nếu chưa có
2. Thay tất cả ký tự không hợp lệ bằng dấu cách
3. Thay `;:()` bằng `,`
4. Loại bỏ dấu câu lặp: `...` → `.`, `,,` → `,`
5. Chuẩn hóa khoảng trắng: `  ` → ` `
6. Nếu không kết thúc bằng `.?!,` thì thêm `.`

### 6.2 Tokenization (character-level)

VietVoice dùng **tokenization cấp ký tự** (character-level), không dùng BPE hay word-piece.

**Vocabulary file:** `vocab.txt` – mỗi dòng là một ký tự, chỉ số là số thứ tự dòng.

**Chuyển văn bản thành chỉ số:**
```python
text_ids = [vocab_char_map.get(c, 0) for c in text]
```
Ký tự không tìm thấy trong vocab → index 0 (padding/unknown).

**Ghép văn bản tham chiếu và đích:**
```
combined_text = reference_text + target_text
text_ids = text_to_indices([list(combined_text)])
```
→ shape `(1, T_text)` với `T_text = len(reference_text) + len(target_text)`

### 6.3 Tính độ dài văn bản

```python
text_length = len(text.encode('utf-8')) + 3 * len(re.findall(pause_punctuation, text))
```

**Bảng trọng số:**
| Loại ký tự | Trọng số |
|---|---|
| Ký tự ASCII (a-z, 0-9) | 1 byte |
| Ký tự tiếng Việt có dấu | 2–3 bytes |
| Dấu câu ngừng nghỉ (.,?!:) | +3 bytes bổ sung |

---

## 7. Xử lý âm thanh (AudioProcessor)

### 7.1 Chuẩn hóa int16

Xem [Mục 3.1](#31-tải-và-chuẩn-hóa-audio-tham-chiếu).

Công thức tổng hợp:

$$\text{audio\_norm} = \text{round}\!\left( (x - \bar{x}) \cdot \frac{29491}{\max|x - \bar{x}|} \right)$$

### 7.2 Sửa clipping

```python
if max(|audio|) >= 32767:
    scale = 26214.0 / max(|audio|)    # 80% × 32767
    audio = round(audio × scale)
```

### 7.3 Lưu file WAV

Dùng `soundfile.write` với format `WAVEX` (WAV Extensible, hỗ trợ metadata đầy đủ).

---

## 8. Chunking và xử lý văn bản dài

### 8.1 Lý do cần chunking

Model có giới hạn về **tổng thời lượng** xử lý mỗi lần (`max_chunk_duration = 15.0` giây mặc định). Nếu:
```
ref_audio_duration + target_audio_duration > max_chunk_duration
```
thì cần chia văn bản đích thành nhiều chunk.

### 8.2 Công thức chia chunk

**Bước 1 – Tính thời lượng khả dụng cho target mỗi chunk:**
```
available_target_duration = max_chunk_duration - ref_audio_duration - safety_margin
```
`safety_margin = 1.0` giây để tránh vượt giới hạn.

**Bước 2 – Tính số ký tự tối đa mỗi chunk:**
```
max_chars_per_chunk = int(speaking_rate × available_target_duration × speed)
```

**Bước 3 – Thuật toán chia câu** (`chunk_text`):

```
1. Chia thành câu theo dấu .?!
2. Câu dài → chia tiếp theo dấu ,
3. Phần vẫn dài → chia đều
4. Gom các câu vào chunk sao cho tổng ≤ max_chars
5. Ghép chunk ngắn (<4 từ) với chunk liền kề
```

---

## 9. Cross-fade: Ghép các đoạn âm thanh

Khi có nhiều chunk, các waveform được ghép lại với **cross-fading** để tránh "bụp" giữa các đoạn.

### 9.1 Cross-fade tuyến tính (phương pháp cơ bản)

```
N_cf = cross_fade_duration × sample_rate    (số mẫu overlap)

fade_out = linspace(1, 0, N_cf)   ← giảm dần từ 1 về 0
fade_in  = linspace(0, 1, N_cf)   ← tăng dần từ 0 lên 1

overlap = prev[-N_cf:] × fade_out + next[:N_cf] × fade_in
result = concat(prev[:-N_cf], overlap, next[N_cf:])
```

### 9.2 Cross-fade cosine (phương pháp cải tiến)

Thay vì tuyến tính, dùng **đường cong cosine** để chuyển tiếp mượt hơn:

$$\text{fade\_out}(i) = \cos^2\!\left(\frac{i \cdot \pi/2}{N_{cf}}\right)$$
$$\text{fade\_in}(i) = \sin^2\!\left(\frac{i \cdot \pi/2}{N_{cf}}\right)$$

**Tính chất:** $\text{fade\_out}(i) + \text{fade\_in}(i) = \cos^2\theta + \sin^2\theta = 1$ → tổng luôn bằng 1, đảm bảo năng lượng không thay đổi.

### 9.3 Volume matching

Trước khi cross-fade, điều chỉnh âm lượng chunk mới để khớp với chunk trước:

```
prev_rms = sqrt(mean(prev_overlap²))
next_rms = sqrt(mean(next_overlap²))

volume_ratio = clip(prev_rms / next_rms, 0.7, 1.5)
next_wave = (next_wave × volume_ratio).astype(int16)
```

**RMS (Root Mean Square)** là thước đo âm lượng trung bình:
$$\text{RMS} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2}$$

Hệ số điều chỉnh bị giới hạn trong `[0.7, 1.5]` để tránh biến dạng âm thanh khi hai chunk quá chênh lệch.

---

## 10. Toàn bộ luồng dữ liệu từ đầu đến cuối

```
INPUT
  text = "Xin chào các bạn!"
  ref_audio = bytes (audio tham chiếu)
  ref_text  = "Đây là đoạn văn tham chiếu."

BƯỚC 1: TextProcessor.clean_text()
  text     = "Xin chào các bạn!"   → "Xin chào các bạn!"
  ref_text = "Đây là đoạn văn tham chiếu."

BƯỚC 2: AudioProcessor.load_audio()
  audio_bytes → AudioSegment (mono, 24kHz) → np.float32
             → normalize_to_int16()
             → audio: np.int16, shape (L,)
             → audio.reshape(1, 1, L)

BƯỚC 3: Duration estimation
  ref_text_len  = calculate_text_length(ref_text)
  ref_audio_dur = L / 24000
  speaking_rate = ref_text_len / ref_audio_dur
  target_dur    = max(target_text_len / speaking_rate / speed, 1.0)

BƯỚC 4: TextProcessor.text_to_indices()
  combined = ref_text + text
  text_ids = [vocab[c] for c in combined]   → np.int32, shape (1, T)

BƯỚC 5: preprocess.onnx
  Inputs:
    audio       : (1, 1, L)
    text_ids    : (1, T)
    max_duration: [chunk_audio_len]
  Outputs:
    noise           : (1, N, d_mel)    ← Gaussian noise
    rope_cos_q/sin_q: (1, N, d_head/2)
    rope_cos_k/sin_k: (1, N, d_head/2)
    cat_mel_text    : (1, N, d_model)  ← conditioning với text+mel
    cat_mel_text_drop:(1, N, d_model)  ← conditioning không có text
    ref_signal_len  : [T_ref]

BƯỚC 6: transformer.onnx × 31 lần (Euler steps)
  t = 0/32
  FOR i IN range(31):
    v = transformer(noise, rope_cos_q, rope_sin_q,
                    rope_cos_k, rope_sin_k,
                    cat_mel_text, cat_mel_text_drop, t)
    noise = noise + (1/32) × v_cfg   ← Euler update
    t = t + 1/32
  ENDFOR
  → noise ≈ mel spectrogram đích, shape (1, N, d_mel)

BƯỚC 7: decode.onnx
  Inputs:
    noise          : (1, N, d_mel)
    ref_signal_len : [T_ref]
  → cắt lấy phần target: mel[:, T_ref:, :]
  → Vocoder (HiFi-GAN/BigVGAN): mel → waveform
  Output: waveform, shape (1, L_out)

BƯỚC 8: AudioProcessor.concatenate_with_crossfade_improved()
  (nếu nhiều chunk)
  fix_clipped_audio → cosine cross-fade → volume matching
  → final_wave: np.int16, shape (L_total,)

BƯỚC 9: AudioProcessor.save_audio()
  soundfile.write(output_path, final_wave, 24000, format='WAVEX')

OUTPUT: File WAV ở 24kHz
```

---

## 11. Tổng hợp các công thức quan trọng

### Nhóm 1: Xử lý âm thanh

| Công thức | Ý nghĩa |
|---|---|
| $x_{norm} = (x - \bar{x}) \cdot \frac{29491}{\max\|x - \bar{x}\|}$ | Chuẩn hóa audio về int16 (90% headroom) |
| $n_{frames} = \lfloor L / H \rfloor + 1$ | Số frame mel từ $L$ mẫu, hop $H=256$ |
| $\text{RMS} = \sqrt{\frac{1}{N}\sum x_i^2}$ | Âm lượng trung bình (Root Mean Square) |

### Nhóm 2: Ước lượng thời gian

| Công thức | Ý nghĩa |
|---|---|
| $r = L_{text\_ref} / d_{ref}$ | Tốc độ đọc (bytes/giây) |
| $d_{target} = \max\!\left(\frac{L_{text\_target}}{r \cdot s}, 1.0\right)$ | Thời lượng audio đích |
| $L_{max} = d_{max} - d_{ref} - 1.0$ | Thời lượng target tối đa mỗi chunk |

### Nhóm 3: Flow Matching

| Công thức | Ý nghĩa |
|---|---|
| $X_t = (1-t)X_0 + tX_1$ | Linear interpolation path |
| $u_t = X_1 - X_0$ | Target vector field (hằng số) |
| $X_{t+\Delta t} = X_t + \Delta t \cdot v_\theta(X_t, t, c)$ | Euler update |
| $v_{cfg} = v_\emptyset + \alpha(v_c - v_\emptyset)$ | Classifier-Free Guidance |

### Nhóm 4: Transformer & Attention

| Công thức | Ý nghĩa |
|---|---|
| $\text{Attn}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$ | Multi-Head Attention |
| $\theta_i = 10000^{-2i/d}$ | RoPE base frequencies |
| $q_m^{(2i)} \leftarrow q_m^{(2i)}\cos(m\theta_i) - q_m^{(2i+1)}\sin(m\theta_i)$ | RoPE rotation (phần thực) |
| $q_m^{(2i+1)} \leftarrow q_m^{(2i)}\sin(m\theta_i) + q_m^{(2i+1)}\cos(m\theta_i)$ | RoPE rotation (phần ảo) |

### Nhóm 5: Cross-fade

| Công thức | Ý nghĩa |
|---|---|
| $f_{out}(i) = \cos^2\!\left(\frac{i\pi}{2N_{cf}}\right)$ | Cosine fade-out |
| $f_{in}(i) = \sin^2\!\left(\frac{i\pi}{2N_{cf}}\right)$ | Cosine fade-in |
| $f_{out}(i) + f_{in}(i) = 1$ | Bảo toàn năng lượng |
| $r_{vol} = \text{clip}(\text{RMS}_{prev}/\text{RMS}_{next},\ 0.7,\ 1.5)$ | Điều chỉnh âm lượng |

---

## Phụ lục A: Thông số cấu hình mặc định

| Tham số | Giá trị | Giải thích |
|---|---|---|
| `sample_rate` | 24000 | Tần số lấy mẫu (Hz) |
| `hop_length` | 256 | Bước nhảy STFT (mẫu) |
| `nfe_step` | 32 | Số bước Euler giải ODE |
| `fuse_nfe` | 1 | Số bước mỗi lần gọi transformer |
| `speed` | 1.0 | Tốc độ nói (1.0 = bình thường) |
| `random_seed` | 9527 | Seed ngẫu nhiên (tái lặp kết quả) |
| `max_chunk_duration` | 15.0s | Giới hạn tổng thời lượng mỗi chunk |
| `min_target_duration` | 1.0s | Thời lượng target tối thiểu |
| `cross_fade_duration` | 0.1s | Thời lượng overlap khi ghép chunk |

## Phụ lục B: Lựa chọn giọng nói

| Chiều | Giá trị hợp lệ |
|---|---|
| `gender` | `male`, `female` |
| `area` | `northern`, `southern`, `central` |
| `group` | `story`, `news`, `audiobook`, `interview`, `review` |
| `emotion` | `neutral`, `serious`, `monotone`, `sad`, `surprised`, `happy`, `angry` |

Mẫu được lọc từ `audio_metadata.json` (trong file model) theo các tiêu chí này, rồi chọn ngẫu nhiên trong tập hợp phù hợp.

## Phụ lục C: Tăng tốc ONNX Runtime

Model được tối ưu qua các tùy chọn ONNX Runtime:

| Tùy chọn | Giá trị | Lợi ích |
|---|---|---|
| `graph_optimization_level` | `ORT_ENABLE_ALL` | Tự động tối ưu graph: fusion ops, constant folding |
| `execution_mode` | `ORT_SEQUENTIAL` | Phù hợp model dạng chuỗi (pipeline) |
| `allow_spinning` | `1` | Busy-wait thay vì sleep → giảm latency |
| `set_denormal_as_zero` | `1` | Tránh xử lý số denormal (gần 0) tốn kém |
| `CUDAExecutionProvider` | ưu tiên đầu | Tự động dùng GPU nếu có CUDA |

Thứ tự ưu tiên provider: `CUDA` → `CPU`. Nếu không có GPU, tự động fallback về CPU.
