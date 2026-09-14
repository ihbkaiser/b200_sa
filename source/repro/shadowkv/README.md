# ShadowKV research — reproducible setup and campaigns

The stable project contract is [`ShadowKV/PROJECT.md`](../../ShadowKV/PROJECT.md).
This README is the executable setup guide. **Code always moves through Git.**
Models, datasets, environments and other artifacts are never copied through an
SSH control tunnel: each machine installs, downloads or generates them directly.

## Fresh-machine setup (canonical for m2, then m3)

The order below is deliberate. Machine 2 is the validation machine for this
recipe; machine 3 must follow the same recipe with its own `$BASE` and GPU list.
Do not copy an environment archive or the generated RULER tree from another
machine.

### 1. Obtain code through Git

```bash
# m2
export BASE=/home/nbnguyen
cd "$BASE"
GIT_LFS_SKIP_SMUDGE=1 git clone --branch shadowkv-paris-retro-streaming --single-branch \
  https://github.com/nguyenngocbaocmt02/s2-ttt.git shadowkv-research
export CODE=$BASE/shadowkv-research

# m3 uses its permitted workspace instead:
# BASE=/home/zhufangzhou/workspace/sheruifeng/baonn/baonn
# clone into "$BASE/shadowkv-research"
```

For an existing checkout, use `git fetch origin shadowkv-paris-retro-streaming`
and a clean fast-forward/switch. Never create or repair source files with `scp`, `rsync`, a
tar stream, or a heredoc over SSH.

### 2. Install the environment directly from package channels

This project intentionally does not install upstream `requirements.txt`: it
pins torch 2.3/Transformers 4.43 and vLLM 0.5, which cannot load Qwen3. The
verified research stack is Python 3.10, torch 2.6.0+cu124, Transformers 4.55.4
and FlashAttention 2.7.4.post1.

```bash
export ENV=$BASE/envs/shadowkv
conda create -y -p "$ENV" python=3.10 pip

# Download CUDA-enabled torch normally from the official wheel index.
"$ENV/bin/python" -m pip install \
  --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0

"$ENV/bin/python" -m pip install -r "$CODE/ShadowKV/requirements-research.txt"

# Install only the CUDA 12.4 compiler/runtime headers into this env. The label
# is essential: the unlabelled future NVIDIA channel can resolve the 12.4
# meta-package to CUDA 13.x dependencies.
conda install -y -p "$ENV" --override-channels \
  -c nvidia/label/cuda-12.4.1 -c defaults \
  cuda-nvcc=12.4.131 cuda-cudart-dev=12.4.127 \
  libcublas-dev=12.4.5.8 libcusparse-dev=12.3.1.170 \
  libcusolver-dev=11.6.1.9 libcurand-dev=10.3.5.147
# Install the matching published wheel directly. Select by the complete tuple
# (flash-attn, Python, PyTorch, CUDA, platform) from
# https://github.com/mjun0812/flash-attention-prebuild-wheels/blob/main/doc/packages.md
# Going through the sdist can trigger a costly build or fail with EXDEV when
# pip's cache crosses filesystems.
"$ENV/bin/python" -m pip install --no-cache-dir \
  "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.0.8/flash_attn-2.7.4.post1%2Bcu124torch2.6-cp310-cp310-linux_x86_64.whl"
# Verified asset SHA256 (downloaded on m2, 2026-09-04):
# 79ee00cc709d05e0c29e3ddafbdbfce4a482df7c183164ef3742736e9d5394ef
```

Do not shorten the match to just `cu12`: verify the wheel filename against
`python --version`, `torch.__version__`, `torch.version.cuda`, the platform and
`torch._C._GLIBCXX_USE_CXX11_ABI` when the publisher encodes the ABI. After
installation, importing `flash_attn` is only the first gate; the CUDA extension
build, the tests and the one-cell GPU smoke below are still mandatory.

On m2 this fingerprint was verified on 2026-09-04: Python 3.10, torch
`2.6.0+cu124`, FlashAttention `2.7.4.post1`, Transformers `4.55.4`, nvcc
`12.4.131`, and Ada compute capability 8.9. Do not reuse that wheel if any
member of the tuple changes.

Corporate mirrors may be configured with `pip config --user`; keep the package
versions above unchanged. Installation is allowed to use the machine's normal
Internet connection. Do not substitute an env from KVPress/TrimKV.

### 3. Download models/datasets normally, or use verified local copies

Launchers contain paths, never model blobs. If a path already exists, verify
its config, tokenizer and every indexed weight shard before downloading again.
If it does not exist, use `huggingface-cli download ... --local-dir <path>` on
that machine. LongBench, AIME and GPQA obey the environment variables recorded
in `PROJECT.md`.

RULER is tokenizer-specific and is generated locally from Git-tracked source
corpora:

```bash
cd "$CODE"
source repro/shadowkv/env_m2.sh       # on m3: source repro/shadowkv/env_m3.sh

$PY -c "import nltk; nltk.download('punkt')"

repro/shadowkv/build_ruler.sh "$SHADOWKV_LLAMA32_PATH" llama-3 100 \
  4096 8192 16384 32768 65536 131072
repro/shadowkv/build_ruler.sh "$SHADOWKV_QWEN3_PATH" qwen 100 \
  4096 8192 16384 32768 65536 131072

$PY repro/shadowkv/status_ruler_build.py
```

This creates exactly 156 `validation.jsonl` files (2 tokenizers × 6 lengths ×
13 tasks), 100 rows each. These generated files are not committed to Git.

### 4. Build and validate the ShadowKV extension

```bash
cd "$CODE/ShadowKV"
test -d 3rdparty/cutlass || \
  git clone --branch v3.5.1 --depth 1 https://github.com/NVIDIA/cutlass.git 3rdparty/cutlass
CUDA_HOME="$ENV" TORCH_CUDA_ARCH_LIST="8.9" MAX_JOBS=8 \
  "$ENV/bin/python" setup.py build_ext --inplace

cd "$CODE"
source repro/shadowkv/env_m2.sh
$PY -c 'import torch, transformers, flash_attn; from ShadowKV.kernels import shadowkv; print(torch.__version__, transformers.__version__, flash_attn.__version__)'
cd "$SHADOWKV_DIR"
$PY -m pytest -q tests ../repro/shadowkv/test_cell_key.py
cd "$CODE"
$PY repro/shadowkv/preflight_benchmark_data.py --samples 100 --skip-ruler
```

On m3 change `TORCH_CUDA_ARCH_LIST` to `8.9` as well (L40). The `--skip-ruler`
flag avoids rereading 2.4 GB after the 156-file manifest has already passed.

### 5. One real GPU smoke, then stop

Setup is not complete until this uses the production `run_cell.sh` entry point
and writes one prediction plus a stamp. It is not authorization to launch a
campaign.

```bash
cd "$CODE"
source repro/shadowkv/env_m2.sh
$PY repro/shadowkv/preflight.py --model "$SHADOWKV_QWEN3_PATH" \
  --template qwen --datalen 4096 --method quest_streaming --sparse_budget 512

SHADOWKV_RESULTS_ROOT=$BASE/_smoke_shadowkv_setup \
  repro/shadowkv/run_cell.sh qwen3 4096 niah_single_1 \
  adaptive_centroid_lse_prefix4 512 160 8 16 2 1 1
```

Machine 2 uses only GPU1--4; GPU0 is the 4 GB T400 and is forbidden. Always
source the machine launcher so `CUDA_DEVICE_ORDER=PCI_BUS_ID` is set. Machine 3
must use a GPU explicitly approved in its launcher and run the same preflight
and smoke before a queue is opened.

## Machine 4 (`sashimi`, 2 x RTX 4090)

Machine 4 has a clean project checkout under `/storage/nbao/shadowkv-research` and an
isolated CUDA 12.4 environment.  Do not activate its older `kvpress` env for
this project.

```bash
ssh baonn@192.168.5.185
cd /storage/nbao/shadowkv-research
source repro/shadowkv/env_m4.sh

# Optional CPU-only readiness check. The locally generated RULER tree is the
# canonical 156-file/100-sample grid; --skip-ruler avoids rereading it.
$PY repro/shadowkv/preflight_benchmark_data.py --samples 100 --skip-ruler

# One-cell smoke through the exact pool entry point.
SHADOWKV_RESULTS_ROOT=/storage/nbao/_smoke_shadowkv \
  repro/shadowkv/run_cell.sh qwen3 4096 niah_single_1 \
  adaptive_centroid_lse_prefix4 512 160 8 16 2 1 0

# Example two-GPU campaign.
$PY repro/shadowkv/gen_queue.py \
  --models llama32,qwen3 --datalens 32768 \
  --methods quest_streaming,shadowkv --budgets 512,1024,2048 \
  > "$SHADOWKV_RESULTS_ROOT/queue.txt"
repro/shadowkv/run_pool.sh "$SHADOWKV_RESULTS_ROOT/queue.txt"
```

`env_m4.sh` fixes all model/dataset paths, forces offline Hugging Face access,
uses GPUs `0,1`, and writes outputs under `/storage/nbao/shadowkv_results` by
default.  To isolate a campaign, override `SHADOWKV_RESULTS_ROOT` before
generating its queue.

The machine-specific launcher resolves `CODE` from its own checkout; it must
never point a clean clone back to `/storage/nbao/s2-ttt`.

### 128K memory gate on 24 GiB GPUs

GPU-only Qwen3 fails at 128K because exact K/V alone is 18 GiB before adding
7.5 GiB of weights. ParisKV quality and speed therefore run through the
authors' native CPU-offload/UVA implementation; our adaptive router uses its
own pinned-CPU backing store and fused UVA gather.

The real 128K gate passed on an RTX A5000 (24 GiB), budget 512, prefix 32 and
recent 32 on 2026-09-05.  The current efficiency backend adds three exact
runtime optimizations: CUDA-graph replay of the unchanged block-8 optimizer,
one fused decode append, and a ping-pong temporal gather that reuses selected
blocks already on GPU while fetching only replacements from pinned CPU.  It
also prevents Triton from recompiling the router for every newly active block.

| runtime | prefill | decode | measured GPU peak |
|---|---:|---:|---:|
| ours, Qwen3-4B, current runtime (16 measured tokens) | 133.38 s | 35.22 ms/token | 16.07 GiB |
| ours, Llama-3.2-3B, current runtime (16 measured tokens) | 76.84 s | 26.87 ms/token | 12.79 GiB |
| ParisKV authors' native CUDA runtime + CPU offload | 135.83 s | 66.60 ms/token | 17.62 GiB |

The decode figures are engineering gates, not paper throughput claims; the
ParisKV row is the earlier seven-token author-runtime gate and is not an
identical loop.  At 32K, longer isolated runs (96 measured tokens) give Qwen3
34.53 ms/token median, 35.64 mean, and Llama-3.2 26.19 median, 27.01 mean.
The important result here is that both models complete a real 128K prefill and
decode on 24 GiB without OOM.

The validation exposed and fixed two dormant >64K/CPU-only bugs: integer
parsing of `--sparse_budget` and Qwen3's tiled post-attention output width.
The fixes are in commits `6a45a76f` and `6d00ca82` respectively.

## Streaming retrieval integration branch

The integration contract and provenance are documented in
[`ShadowKV/docs/streaming_retrieval_port.md`](../../ShadowKV/docs/streaming_retrieval_port.md).
Released `shadowkv` remains the original static-prefill baseline.  The methods
below own an append-only prompt/decode lifecycle; a partial prompt tail is
completed by later decode tokens instead of resetting the block grid:

- `quest_streaming`;
- `adaptive_centroid_lse_streaming_prefix4_querymean` (ours, canonical:
  mean self-key LSE placement and tail-CVaR allocation at 0.25, with the
  independent fixed center price `lambda=1.5` and no gap correction);
- `adaptive_centroid_lse_streaming_prefix4` (max-GQA ablation);
- `exact_block_lse_streaming` and `exact_block_max_streaming` (offline
  full-key routing oracles);
- `pariskv_author_common` (Qwen3 and Llama-3; authors' retrieval components inside
  the shared model-forward/evaluation path);
- `retroinfer_reference_streaming`.

RetroInfer retains `reference` in its name.  The canonical ParisKV quality
baseline is `pariskv_author_common`: it reuses the authors' SRHT, collision
voting, radix selection and RaBitQ components, but deliberately shares the
same model-forward path as the other methods.  Results from the older
`pariskv_official` end-to-end path are historical diagnostics and must not be
included in comparison tables.

Run the complete CPU gate before a GPU smoke:

```bash
cd "$CODE"
$PY -m pytest -q \
  ShadowKV/tests/test_streaming_blocks.py \
  ShadowKV/tests/test_streaming_caches.py \
  ShadowKV/tests/test_retroinfer_reference.py \
  repro/shadowkv/test_cell_key.py

$PY repro/shadowkv/compare_retroinfer_author_reference.py \
  --author-root /path/to/RetrievalAttention
```

The external checkouts are validation inputs only.  ParisKV code is not copied
because the inspected repository has no license; the RetroInfer checkout is
MIT-licensed.  A one-cell production-entry smoke uses the ordinary launcher:

```bash
source repro/shadowkv/env_m1.sh       # use env_m2.sh or env_m4.sh on that host
export STREAMING_RECENT_TOKENS=32
# Optional for a separate Git worktree sharing an already generated dataset:
# export SHADOWKV_RULER_DATA_ROOT=/absolute/path/to/ShadowKV/data/ruler/data
repro/shadowkv/run_cell.sh qwen3 4096 niah_single_1 \
  quest_streaming 512 160 8 16 0 1 1
repro/shadowkv/run_cell.sh qwen3 4096 niah_single_1 \
  adaptive_centroid_lse_streaming_prefix4 512 160 8 16 0 1 1
```

For 24-GiB runs, opt into offload explicitly.  Ours can additionally use its
two-slot Triton router; the quality/reference default remains `torch`:

```bash
export STREAMING_OFFLOAD=1
export STREAMING_GATHER_BACKEND=uva
export STREAMING_ROUTER_BACKEND=triton
export SHADOWKV_RMSNORM_BACKEND=flashinfer  # efficiency only; torch is default
# Query-mean default: mean-gap path + per-block fixed-price RD (lambda=1.5).
# No ADAPTIVE_LSE_EXTRA_FRACTION quota is used.
export QUEST_PREFIX_TOKENS=32 STREAMING_RECENT_TOKENS=32
repro/shadowkv/run_cell.sh qwen3 131072 niah_single_1 \
  adaptive_centroid_lse_streaming_prefix4_querymean 512 160 8 16 0 1 1
```

### Authors' native ParisKV runtime

The production comparison uses the authors' GPU-native collision voting,
radix top-k, fused 4-bit reranker and UVA fetch, pinned to commit
`db7ad7f59ebe5670f9f2e1092b1b689802dda7c5`.  Prepare it as an external
dependency (do not copy its source into this repository):

```bash
export PARISKV_AUTHOR_ROOT=$BASE/ParisKV-official
export SHADOWKV_THIRD_PARTY_ROOT=$BASE/third_party
PY=$PY repro/shadowkv/setup_pariskv_official.sh "$PARISKV_AUTHOR_ROOT"
```

The setup installs FlashInfer 0.2.4 and fast-hadamard-transform 1.1.0 and
applies only compatibility edits for a local model path/Transformers 4.55.
Exercise the native end-to-end path with CPU offload:

```bash
CUDA_VISIBLE_DEVICES=0 $PY repro/shadowkv/smoke_pariskv_official.py \
  --author-root "$PARISKV_AUTHOR_ROOT" --model "$SHADOWKV_QWEN3_PATH" \
  --input-tokens 131072 --new-tokens 34 --final-topk 512 \
  --sink-size 32 --local-size 32 --device cuda:0
```

The native path is retained only as an implementation diagnostic.  The
canonical quality path uses the same model forward as the other methods:

```bash
repro/shadowkv/run_cell.sh qwen3 32768 cwe \
  pariskv_author_common 1024 160 8 16 0 30 1
```

For the authors' published operating point, omit the last two overrides (the
script defaults to sink 4, local 512). Keep the fair 32/32 variant separate in
tables: it changes exact-region accounting, not the ParisKV retrieval kernel.

The corresponding engineering gates at 32K/512 on A5000 use identical 32/32
exact regions: current ours is 34.53 ms/token median over 96 measured tokens;
the earlier native ParisKV smoke was 45.61 ms/token. These loops are not
implementation-identical, so use them to detect regressions and capacity
issues; use matched wall-clock/component timings before making a paper claim.

Only after those cells pass should a queue be generated, for example:

```bash
$PY repro/shadowkv/gen_queue.py \
  --models qwen3 --datalens 32768 \
  --methods quest_streaming,adaptive_centroid_lse_streaming_prefix4,pariskv_author_common,retroinfer_reference_streaming \
  --budgets 512,1024,2048 > "$SHADOWKV_RESULTS_ROOT/streaming_queue.txt"
```

## 1. Cái gì đang có

| | |
|---|---|
| repo | `ShadowKV/` (clone `bytedance/ShadowKV`, ICML 2025 spotlight) |
| env | `~/miniconda3/envs/shadowkv` — py3.10, torch 2.6.0+cu124, transformers 4.55.4, flash_attn 2.7.4.post1 |
| kernel | `ShadowKV/kernels/shadowkv.cpython-310-x86_64-linux-gnu.so` (CUTLASS v3.5.1 + CUDA 12.4, sm_86) |
| model | Llama-3.2-3B-Instruct, Qwen3-4B-Instruct-2507 — cả hai **local**, không cần hub |
| method | `full`, `quest_streaming`, `shadowkv`, `shadowkv_cpu` |
| bench | RULER, build offline bằng chính generator của ShadowKV (13 task) |
| test | `ShadowKV/tests/` — 22 test, chạy trước mỗi commit đụng harness |

## 2. Chạy

```bash
cd /home/baonn/s2-ttt
source repro/shadowkv/env_m1.sh          # mọi path đều đã thấy trong ls

# canonical data grid: 100 examples for every one of 13 tasks, 4K--128K
repro/shadowkv/build_ruler.sh "$SHADOWKV_LLAMA32_PATH" llama-3 100 \
    4096 8192 16384 32768 65536 131072
repro/shadowkv/build_ruler.sh "$SHADOWKV_QWEN3_PATH" qwen 100 \
    4096 8192 16384 32768 65536 131072

# CPU-only progress/readiness checks (no CUDA call)
$PY repro/shadowkv/status_ruler_build.py
$PY repro/shadowkv/preflight_benchmark_data.py

# preflight — in [OK]/[MISSING], exit 1 nếu thiếu
$PY repro/shadowkv/preflight.py --model "$SHADOWKV_LLAMA32_PATH" --template llama-3 \
    --datalen 8192 --method quest_streaming --sparse_budget 1024

# một ô
repro/shadowkv/run_cell.sh llama32 8192 niah_single_1 quest_streaming 1024 160 8 16 2 96 6
#                          model    len  task          method b   rank ch pg dn N  gpu

# campaign
$PY repro/shadowkv/gen_queue.py --models llama32,qwen3 --datalens 8192,16384 \
    --methods full,quest_streaming,shadowkv --budgets 1024 > $SHADOWKV_RESULTS_ROOT/queue.txt
repro/shadowkv/run_pool.sh $SHADOWKV_RESULTS_ROOT/queue.txt

# trạng thái — một lệnh in hết
$PY repro/shadowkv/status.py
```

The evaluation adapter also accepts `longbench/<task>`, `aime25`, and
`gpqa/diamond`. It reads the same local parquet snapshots and imports the same
scorers as the established KVPress campaigns. LongBench has 16 tasks; AIME-25
has 30 questions; GPQA Diamond has 198. `--num_samples 100` therefore means
100 per LongBench/RULER task, all 30 AIME questions, and the first 100 GPQA
questions. For reasoning runs, `--max_new_tokens` can override the local
dataset default while the model allocation automatically reserves 32K tokens
for AIME and 16K for GPQA.

Examples (one process/GPU; use the pool only after choosing the final method
grid):

```bash
cd "$SHADOWKV_DIR"

# LongBench: pass any comma-separated subset, 100 rows from each task.
CUDA_VISIBLE_DEVICES=0 "$PY" test/eval_acc.py \
  --model_name "$SHADOWKV_QWEN3_PATH" --datalen 32768 --method full \
  --dataset_name longbench/narrativeqa,longbench/qasper,longbench/hotpotqa \
  --num_samples 100 --out_root /storage/baonn/shadowkv_longbench

# Reasoning: keep the answer budget explicit in every result-producing run.
CUDA_VISIBLE_DEVICES=0 "$PY" test/eval_acc.py \
  --model_name "$SHADOWKV_QWEN3_PATH" --datalen 4096 --method full \
  --dataset_name aime25 --num_samples -1 --max_new_tokens 4096 \
  --out_root /storage/baonn/shadowkv_aime25
CUDA_VISIBLE_DEVICES=0 "$PY" test/eval_acc.py \
  --model_name "$SHADOWKV_QWEN3_PATH" --datalen 4096 --method full \
  --dataset_name gpqa/diamond --num_samples 100 --max_new_tokens 4096 \
  --out_root /storage/baonn/shadowkv_gpqa
```

Dừng pool: lấy PID pool trước, kill **pool trước, cell sau**. Muốn chạy lại
một ô: xoá **marker** trong `.state/{done,failed}`, không xoá thư mục output.
Rút một GPU khỏi pool: xoá dòng của nó trong `.state/gpus.txt` — worker chạy
nốt ô hiện tại rồi dừng.

## 3. Bẫy đã trả giá trong phiên này

### Method của ta là entry riêng, không có ShadowKV outlier

`adaptive_centroid_lse` dùng `AdaptiveCentroidLSECache` trong
`ShadowKV/models/adaptive_centroid_cache.py`. Entry này tái dùng implementation
centroid-LSE đã test nhưng ép `outlier_chunk=0`, từ chối mọi override khác 0,
assert vùng exact-outlier rỗng sau prefill, và giữ exact post-RoPE K giống
entry streaming (không kế thừa phép tái tạo SVD rank-160 của ShadowKV).
Cấu hình canonical hiện tại là
block 8, `self_lse_adaptive_iso`, trung bình 1.25 tâm/block
(`extra_fraction=0.25`), proxy temperature 1. Cell mới có hậu tố
`_o0_exactkv`.

Đối chứng 16K/24 mẫu/task chạy ở
`/storage/baonn/adaptive_centroid_no_outlier_20260904_ruler16k`; dùng
`repro/shadowkv/compare_adaptive_outlier_ablation.py` để so equal-prefix với
bản lịch sử có 48 outlier block.

Biến thể ablation `adaptive_centroid_lse_prefix4` luôn giữ exact 4 block đầu
(32 token) **ngoài** b512, mask chúng khỏi dynamic top-k, rồi vẫn retrieve đủ
512 token. Nó vẫn có `outlier_chunk=0` và giữ recent local 32 token như bản
gốc. Cell ghi `adaptive_lse_prefix4_..._o0`; kết quả RULER-16K first-24 nằm ở
`/storage/baonn/adaptive_centroid_prefix4_20260904_ruler16k`. Không gọi đây là
matched-budget: tổng attended tăng 32 token so với `adaptive_centroid_lse`.

Từ bản exact-backing, cell static có hậu tố `_exactkv` và không còn trường
`_r160`, vì rank SVD không thuộc method này. Các cell lịch sử chỉ có `_o0`
nhưng không có `_exactkv` dùng key rank-160 kế thừa từ ShadowKV; không trộn
chúng vào phép so static/streaming. Static và streaming mới dùng cùng
allocation pool và cùng router-normalization pool (đều loại prefix/recent
exact); khác biệt còn lại chỉ là streaming đưa block decode mới vào index.

### 3.0 Outlier phải scale theo context, không giữ literal 48

Paper dùng 48 block × 8 token ở 128K, tương ứng 384/131072 = 0.293% token.
Harness hiện scale đúng tỷ lệ đó bằng `outlier_policy.py`: c8 cho
4K/8K/16K/32K/64K/128K lần lượt là 2/3/6/12/24/48 block. `run_cell.sh`
truyền giá trị explicit vào model; tên cell và stamp đều có `_o<N>` để kết quả
mới không bao giờ bị lẫn với campaign lịch sử dùng `_o48` ngầm định.

Có thể override cho ablation bằng `SHADOWKV_OUTLIER_CHUNKS=N`; override cũng đi
vào tên cell. Campaign trước ngày 04/09/2026 bên dưới là số lịch sử với 48
block cố định ở mọi context, không được dùng như bản ratio-matched.

### 3.1 ⚠ Cùng `--sparse_budget` KHÔNG phải cùng ngân sách

Đo bằng `budget_audit.py`, Llama-3.2-3B @ prefill 8192:

| flag budget | streaming Quest attended | shadowkv attended |
|---|---|---|
| 128 | 160 (=128 + recent 32) | **544** (=128 + outlier 384 + local 32) |
| 512 | 544 | **928** |
| 1024 | 1056 | **1440** |
| 2048 | 2080 | **2464** |

Campaign lịch sử này giữ thêm `outlier_chunk × chunk_size = 48 × 8 = 384`
token và một local window. Đặt hai method cạnh nhau ở cùng flag là **cho
ShadowKV thêm ~41% token ở budget 1024**. Bảng công bằng phải khớp theo
*attended tokens*:

```bash
$PY repro/shadowkv/budget_audit.py --datalen 8192 --budgets 128,512,1024,2048
# shadowkv b1024 attends 1440 -> pair it with quest_streaming b1408
```

**✅ ĐÃ CHỐT 27/08/2026 — chạy CẢ HAI lưới.** ShadowKV ở budget danh nghĩa
`{512, 1024, 2048}`; streaming Quest ở **cả** danh nghĩa `{512, 1024, 2048}` (đối chiếu
được với con số hai paper tự báo) **lẫn** khớp-attended `{896, 1408, 2432}`
(so sánh công bằng). Quest rẻ hơn ShadowKV nên phần thêm chỉ ~1.5× chi phí
Quest, không nhân đôi campaign.

Trong campaign lịch sử, offset là **hằng số +416** ở mọi độ dài và cả hai model (đo bằng
`budget_audit.py`, đã kiểm 8K/16K × llama32/qwen3): `outlier 384 + local 32`.
Lưu ý nhỏ: `prefill_local = incoming % 8 + 32`, mà `incoming` là độ dài prompt
thật sau tokenize chứ không đúng bằng `datalen`, nên offset thật dao động
416–423. Không đáng kể ở mức budget này, nhưng **đừng viết 416 vào paper như
một hằng số chính xác.**

### 3.2 head_dim: `hidden_size // num_heads` sai với Qwen3

Qwen3-4B: hidden 2560, 32 head, nhưng head_dim **128** (không phải 80), và
`num_heads × head_dim = 4096 ≠ hidden_size`. Bản released dùng công thức tắt ở
6 chỗ trong `kv_cache.py` + mỗi model file; đã thay hết bằng
`models/compat.py:head_dim_of()`. Lỗi không nổ lúc khởi tạo mà nổ trong một
`copy_` giữa decode → `tests/test_cache_head_dim.py` ghim shape trực tiếp.

### 3.3 Điểm cao chưa chứng minh method chạy đúng

Cả 3 method đều 1.0 trên `niah_single_1` 2 mẫu. Vô nghĩa. Hai kiểm tra riêng:

* `budget_audit.py` — câu hỏi **cấu trúc**: thật sự attend bao nhiêu token.
* `bite_check.sh` + `diff_predictions.py` — câu hỏi **hành vi**: predictions có
  đổi khi siết budget không.

Lần đầu chạy, `diff_predictions.py` báo "FLAG SWALLOWED" cho ShadowKV; probe
cấu trúc cho thấy cờ **có** cắn (544 → 2464 token). Detector đã sửa lời lẽ:
trên 4 mẫu của một task model luôn trả lời đúng, trùng khớp là **vô kết luận**,
không phải bằng chứng.

## 4. Lệch so với bản released, và vì sao

Mọi lệch nằm trong `ShadowKV/models/compat.py` kèm test tương đương bit-exact
(`tests/test_compat_equivalence.py`). Không có cái nào đổi phép tính.

| bỏ | thay bằng | lý do |
|---|---|---|
| `vllm._custom_ops.rotary_embedding` | NeoX rope thuần torch | vllm 0.5.3.post1 ghim torch 2.3.1 → không nạp được Qwen3 |
| `vllm._custom_ops.silu_and_mul` | `silu(gate) * up` | như trên |
| `flashinfer.norm.rmsnorm` | RMSNorm thuần torch mặc định; FlashInfer chỉ bật tường minh bằng `SHADOWKV_RMSNORM_BACKEND=flashinfer` | reference không phụ thuộc package tùy chọn; backend efficiency sai khác tối đa một ULP BF16 trong gate |
| `nemo_toolkit[all]==1.23` | `data/ruler/manifest_utils.py` | RULER chỉ dùng 2 hàm đọc/ghi jsonl từ nemo |
| `minference` | import lazy | chỉ dùng sau cờ `--minference`, ta không dùng |

`transformers` 4.55.4 thay vì 4.43.1 pin của họ: **4.43.1 không biết Qwen3**.
Với transformers ≥ 4.48 `rotary_emb` chuyển từ attention lên model → xử lý
trong `compat.get_inv_freq()`.

GLM vẫn cần vllm; import đã thành optional, GLM báo lỗi rõ ràng lúc khởi tạo
thay vì âm thầm đi đường chưa kiểm chứng.

## 5. Streaming Quest — cài mới, không có trong repo ShadowKV

`ShadowKV/models/quest_streaming_cache.py`. Repo gốc `mit-han-lab/Quest` gắn với
transformers fork thời Llama-2 và kernel CUDA riêng → **không nạp được
Llama-3.2 lẫn Qwen3**. Cài trong khung ShadowKV cho ta thứ repo riêng không cho
được: **cùng prefill, cùng vòng generate, cùng tokenize, cùng data** — chênh
lệch Quest/ShadowKV quy về đúng quy tắc chọn, không lẫn thứ khác.

Score một page: `sum_d max(q_d·min_d, q_d·max_d)`, chọn top `budget/page_size`
page. Hai tính chất được ghim bằng test:

* bound ≥ mọi logit thật trong page (brute-force) — `test_page_score_upper_bounds…`
* **tương đương với code gốc** — `test_score_matches_the_reference_sign_trick`.

Đối chiếu `mit-han-lab/Quest` `evaluation/quest_attention.py` (27/08/2026):
họ đi đường khác nhưng ra đúng cùng một số —
`sign = ±1 theo dấu q`; `chunk_max_key = (k·sign).amax(chunk)`;
`score = (q·sign) @ chunk_max_keyᵀ`. Theo từng chiều d, đó là
`|q_d|·(sign_d·extreme_d)` = `q_d·max_d` khi `q_d>0` và `q_d·min_d` khi `q_d<0`,
tức `max(q_d·min_d, q_d·max_d)`. Test so cả giá trị lẫn **thứ hạng** (thứ hạng
mới là thứ selection dùng).

Ba chỗ paper không nói rõ, **đã phơi ra thành tham số, không phải hằng số ngầm**:

| | mặc định | trạng thái |
|---|---|---|
| `--dense_layers` | 2 | ✅ **đã verify 27/08/2026**: code gốc có `if q_len > 1 or self.layer_id < 2: return self.flash_forward(...)` — hai layer đầu dense, và prefill luôn dense. Khớp cài đặt của ta. |
| GQA reduce | `max` | ⚠ paper chỉ chạy model MHA (Llama-2-7B, LongChat-7B) nên **không có luật gốc**; lấy đúng cách ShadowKV gộp để hai method chỉ khác quy tắc chấm |
| local window | trailing partial page + mọi token sinh ra | ⚠ **lệch nhỏ có chủ ý**: code gốc cho page cuối (chưa đầy) *dự tuyển* — pad `-inf` để nó không thắng nhờ padding. Ta luôn giữ nó. Chênh tối đa `page_size-1` token, và bằng 0 khi độ dài prefill chia hết cho 16. |

Streaming Quest giữ **toàn bộ** KV cache (nó tiết kiệm FLOP, không tiết kiệm bộ nhớ) —
nên bộ nhớ ngang full attention, khác hẳn ShadowKV.

## 6. Campaign đầu tiên (chốt 27/08/2026)

RULER 13 task, **96 mẫu/task** (đúng mặc định của ShadowKV), độ dài **8K + 16K**,
hai model, ba method → **520 ô**.

```bash
source repro/shadowkv/env_m1.sh
$PY repro/shadowkv/gen_queue.py \
    --models llama32,qwen3 --datalens 8192,16384 --methods full,quest_streaming,shadowkv \
    --shadowkv_budgets 512,1024,2048 \
    --quest_budgets 512,1024,2048,896,1408,2432 \
  > $SHADOWKV_RESULTS_ROOT/queue.txt
repro/shadowkv/run_pool.sh $SHADOWKV_RESULTS_ROOT/queue.txt
```

Ước ban đầu 40–120 GPU-giờ (≈6–17h trên 7 GPU) — **ước từ smoke, chưa đáng tin**.
Cost thật đo per-type từ những ô đầu bằng `status.py`, mục `[measured cost]`, rồi
mới báo ETA thật.

## 7. Còn mở
* Chọn độ dài: 24GB đủ cho 8K/16K/32K thoải mái; 64K cần đo lại; **128K
  Qwen3-4B không vừa một A5000** (KV ~19 GiB + weight 8 GiB).
* Chưa có mirror kết quả về hub / chưa có mục HANDOFF.

## Editing a script while the pool is running

`run_cell.sh` cost seven cells to learn this: bash reads a script by BYTE
OFFSET as it executes, so rewriting the file in place under a running worker
makes it resume at the wrong offset and die with a syntax error -- after the
model has finished the whole cell. The work was complete and correct in the
jsonl; only the marker said `failed`.

So: never write a shell script in place while `run_pool.sh` is up. Write a temp
file next to it and `os.replace()` / `mv` it, which swaps the directory entry
and leaves running processes reading the old inode. Same rule for `run_pool.sh`
itself. Python files are safe -- the interpreter reads them once at import.

## Smoke-test a new method before it reaches the queue

Adding `m51w` cost 67 cells to a single missing argparse choice: `run_cell.sh`
grew a `--m51_pq_warm_iters` flag, `eval_acc.py` had not grown the matching
argument, and every worker burned a claim on an instant usage error. Nothing was
wrong with the science and nothing was lost but time -- the markers were cleared
and the cells re-run -- but the pool is not the place to find out that two files
disagree about a flag.

Before enqueuing a new method or a new flag, run one cell by hand with a tiny
sample count and a throwaway results root:

    SHADOWKV_RESULTS_ROOT=$SHADOWKV_RESULTS_ROOT/_smoke_m51w \
      repro/shadowkv/run_cell.sh qwen3 8192 niah_single_1 m51w 512 8 0.999 8 0 2 0

Invoke it WITHOUT `bash`, the way run_pool.sh does. A smoke test run as
`bash run_cell.sh` passed while the pool failed every cell with "Permission
denied": rewriting the script through mkstemp + os.replace had dropped its
execute bit, and only the pool's invocation needs it. A smoke test that does not
use the real entry point is not a smoke test.

Two samples take under a minute and exercise the whole path: cell naming, the
flag list, the model constructor, the cache, and the scorer.
