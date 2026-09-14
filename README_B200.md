# ShadowKV + Quest + Query-Robust / Qwen3 + DeepSeek-R1-Distill-Llama-8B / RULER 128K

Bundle này gồm model Qwen3-4B-Instruct-2507, DeepSeek-R1-Distill-Llama-8B,
source ShadowKV có Qwen3 + Quest, CUTLASS v3.5.1, và hai bộ RULER
tokenizer-specific: Qwen `13 task × 100` và Llama `13 task × 100` ở 131072
token. Dữ liệu và model đều được đọc local; các script chạy với chế độ
HuggingFace offline.

Repo này cũng có một đường Query-Robust cho Qwen3. Nó lấy nguyên
logic QR từ `ihbkaiser/ihb-sparse`: mỗi page có `landmark`, `bias`, và chứng
nhận `epsilon`; lúc decode page được xếp hạng theo
`scale * q·landmark + bias + alpha * epsilon`. Vertex asset Qwen3 M32 đã được
đóng gói ở `source/ShadowKV/artifacts/query_robust/qwen3_4b_128k/` và được
kiểm tra shape, BF16, fingerprint, metadata, padding, và SHA-256 trước khi
dùng. Decode scorer có backend Triton fused bảo toàn top-k/invalid-page
semantics; summary solver vẫn giữ FP32 canonical và có backend `compile` được
parity-test với eager.

Wheelhouse hiện dành cho Linux `x86_64` + Python 3.12 và có PyTorch `2.11.0`
CUDA 12.8; `flash-attn` được giữ dạng source để compile đúng trên B200.

Do phân vùng chuẩn bị chỉ còn khoảng 13G, hai shard DeepSeek được lưu nén
`.safetensors.zst` (tổng raw khoảng 16.06GB). Khôi phục raw weights trên máy
B200 bằng script ở dưới.

## 1. Khôi phục DeepSeek weights

```bash
cd b200_sparse_attention_128k
PY=python3 ./restore_deepseek.sh
```

Mặc định script giữ lại file `.zst` để có thể kiểm tra lại. Nếu B200 thiếu chỗ,
chỉ xóa archive sau khi giải nén và kiểm tra thành công:

```bash
KEEP_COMPRESSED=0 PY=python3 ./restore_deepseek.sh
```

## 2. Cài trên B200

B200/Blackwell cần PyTorch wheel có CUDA/kiến trúc phù hợp. Cài hoặc xác nhận
PyTorch trước, rồi mới chạy `prepare_b200.sh`; không dùng binary CUDA extension
được build trên RTX 3090.

```bash
cd b200_sparse_attention_128k
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'

# Nếu máy chưa có torch, dùng wheel tương thích với Python/CUDA của máy.
# Ví dụ CUDA 13.x: pip install torch --index-url https://download.pytorch.org/whl/cu130

PY=python3 MAX_JOBS=16 ./prepare_b200.sh
```

Nếu máy chỉ cho cài từ wheelhouse offline:

```bash
OFFLINE_INSTALL=1 PY=python3 ./prepare_b200.sh
```

Lệnh trên giả định Python của máy là 3.12. Nếu khác phiên bản Python, dùng
`OFFLINE_INSTALL=0` và cài dependencies từ package index nội bộ.

`prepare_b200.sh` sẽ compile `source/ShadowKV/kernels/*.cu` với
`TORCH_CUDA_ARCH_LIST=10.0`, sau đó import-test cả extension và flash-attn.

## 3. Smoke test trước

```bash
./smoke_b200.sh
```

Kiểm tra QR vertex asset mà không cần model weights hoặc mạng:

```bash
cd source/ShadowKV
PYTHONPATH="$PWD" python tools/validate_query_robust.py
python -m pytest tests/test_query_robust.py -q
```

## 4. Preflight benchmark QR

Sau khi `prepare_b200.sh` hoàn tất, chạy một prompt Qwen3 RULER để lấy latency
decode, latency summary update, HBM peak và throughput:

```bash
QUERY_ROBUST_ROUTER_BACKEND=triton \
QUERY_ROBUST_SUMMARY_BACKEND=compile \
./benchmark_qr_b200.sh
```

Script này dùng `--runtime_out` của harness và bật CUDA-event instrumentation.
Đây là preflight trên máy B200; các số đo RTX 3090/4090 không được dùng làm
claim B200.

Muốn so sánh canonical eager/reference trên cùng máy:

```bash
QUERY_ROBUST_ROUTER_BACKEND=torch \
QUERY_ROBUST_SUMMARY_BACKEND=eager \
./benchmark_qr_b200.sh
```

QR mặc định giữ exact K/V trên HBM B200. Có thể chạy thêm protocol CPU/UVA:

```bash
QUERY_ROBUST_OFFLOAD=1 ./benchmark_qr_b200.sh
```

## 5. Chạy campaign Qwen3

```bash
METHODS=quest_streaming,shadowkv_cpu \
  RESULTS_ROOT="$PWD/results/qwen3_128k_100" \
  ./run_b200_ruler.sh
```

Chạy riêng Query-Robust trên Qwen3:

```bash
MODEL_NAME=qwen3 METHODS=query_robust \
  RESULTS_ROOT="$PWD/results/qwen3_qr_128k_100" \
  ./run_b200_ruler.sh
```

Mặc định QR giữ K/V trên GPU B200, router `auto` và summary `compile`. Nếu muốn
dùng exact K/V pinned CPU và UVA:

```bash
QUERY_ROBUST_OFFLOAD=1 MODEL_NAME=qwen3 METHODS=query_robust \
  ./run_b200_ruler.sh
```

Muốn chạy ShadowKV GPU-resident nguyên bản trên bộ nhớ lớn của B200, dùng
`METHODS=quest_streaming,shadowkv`; `shadowkv_cpu` ở lệnh mặc định giữ đúng
protocol offload đã dùng cho các máy 24 GiB.

Quest được GPU-resident mặc định trên B200. Nếu đặt `QUEST_OFFLOAD=1`, runner
tự động đặt `QUEST_DENSE_LAYERS=0` vì dense leading layers không tương thích
với CPU-offloaded KV.

Chạy DeepSeek-R1-Distill-Llama-8B với bộ RULER Llama tương ứng:

```bash
MODEL_NAME=deepseek \
METHODS=quest_streaming,shadowkv \
RESULTS_ROOT="$PWD/results/deepseek_128k_100" \
./run_b200_ruler.sh
```

Mỗi method được load model một lần và chạy đủ 13 task × 100 mẫu. Cấu hình đã
đóng gói:

- context: `131072`;
- budget: `4096` (`L/32`);
- Quest: page `8`, dense leading layers `2`, prefix `32`, recent `256`, flush `256`;
- ShadowKV: `shadowkv_cpu`, rank `160`, chunk `4`, outlier chunks `96`;
- Quest dùng exact streaming K/V pinned CPU để giữ protocol ổn định.

## 6. Kiểm tra dữ liệu

```bash
./verify_bundle.sh
PY=python3 source/repro/shadowkv/status_ruler_build.py \
  --root source/ShadowKV/data/ruler/data --model qwen --lengths 131072 --samples 100
```

Không cần token HuggingFace, NeMo, hay tải dataset từ internet để chạy bundle
này. Chỉ riêng PyTorch và `flash-attn` phải là bản phù hợp với B200; nếu chúng
chưa có trong wheelhouse thì cài chúng trước trên máy đích. DeepSeek cần raw
shards đã được khôi phục trước khi gọi `run_b200_ruler.sh`.
