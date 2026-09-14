# sparse_attention — định nghĩa chuẩn

Đổi chậm. Thứ đổi mỗi lần có kết quả thì nằm ở `RECORDS.md`.
Luật chung về cách điều hành thực nghiệm nằm ở skill `research_conduction`;
file này là bản cụ thể hoá cho project này.

## 1. Project làm gì

So các phương pháp **sparse attention / KV retrieval** trên **một forward path
chung**, để khác biệt trong bảng là khác biệt của thuật toán chứ không phải của
harness. Model chính: Qwen3-4B-Instruct-2507 và Llama-3.2-3B-Instruct.

## 2. Method — tên CLI chính xác

| nhãn bảng | `--method` | cài đặt | index khi decode |
|---|---|---|---|
| **Ours** | `adaptive_centroid_lse_streaming_prefix4_querymean` | `models/adaptive_centroid_streaming_cache.py` | **8 token** (block seal) |
| Quest | `quest_streaming` | `models/quest_streaming_cache.py` | **16 token** (page seal) |
| ParisKV | `pariskv_author_common` | `models/pariskv_author_streaming_cache.py` + `cache_hub.polar_cache` của tác giả | **8 token** |
| ShadowKV-GPU | `shadowkv` | `models/kv_cache.py` `ShadowKVCache` | **8 token** (wave seal) |
| ShadowKV-CPU | `shadowkv_cpu` | `models/kv_cache.py` `ShadowKVCache_CPU` | **8 token** (wave seal) |
| PQCache | `pqcache_author_common` | `models/pqcache_author_cache.py` | không |
| MagicPIG | `magicpig_author_common` | `models/magicpig_author_cache.py` | không |
| RetroInfer | `retroinfer_reference_streaming` | `models/retroinfer_streaming_cache.py` | **1024 token** (`update_segment`) |
| Full | `full` | — | — |

### 2b. Bảng sự thật về từng method — ĐÃ KIỂM BẰNG CODE 2026-09-13

| method | đơn vị retrieve | index token decode | KV nằm ở | offload |
|---|---|---|---|---|
| ours | block 8 | 8 token (block seal) | pinned CPU khi bật cờ | cờ `--streaming_offload` |
| Quest | page 16 (paper) / 8 (ta) | 16 token | như trên | cờ |
| ParisKV | block 8 | 8 token | như trên | cờ |
| PQCache | **từng token** | **mỗi token** (`_assign_codes`) | `device="cpu", pin_memory` **cứng** | luôn bật |
| ShadowKV-CPU | chunk 8 (paper) / 4 (ta) | **8 token** (wave seal) | `device='cpu'` **cứng** | luôn bật |
| ShadowKV-GPU | chunk 8 | **không** | GPU | — |
| MagicPIG | **từng token** (LSH) | **một lần**, dựng lại từ 0 | server CPU tác giả | — |
| RetroInfer | **cụm IVF** (~16) | 1024 token (`update_segment`) | **GPU** | **KHÔNG CÓ** |

Ba điều dễ kết luận sai (tôi đã sai cả ba trong một phiên, vì grep nông thay vì
đọc hàm):

* PQCache **có** index token decode — `_assign_codes` chạy mỗi `update_kv_cache`.
* RetroInfer **có** mở rộng cụm — `_append_index` trong `update_kv_cache`, và nó
  attend **toàn bộ đuôi chưa index** chính xác (`k_cache[indexed_end:total]`),
  tới 1088 token ở `update_segment=1024`. Ở B=1024 nó dùng gần gấp đôi ngân sách
  các method khác.
* `ShadowKVCache_CPU` **có** wave-sealing token decode — comment trong code tự
  nhận *"Extend the official CPU lifecycle to generated tokens"*. Đây là **mở
  rộng của ta**, không phải paper. ParisKV đánh ShadowKV là **NA** cho GPQA vì
  *"does not support long-generation scenarios"*.

MagicPIG: `server.build_table(layer_idx, 0, length)` **sort lại từ đầu**, không
nhận offset, và chỉ gọi một lần khi context vượt `sink+local`. Token sinh sau đó
không vào bảng. Đúng paper; nhưng ở benchmark sinh dài nó mất nội dung, nên
**để NA cho reasoning** (ParisKV dùng nhãn `MagicPig++` cho trường hợp này nhưng
**không định nghĩa** `++` ở đâu trong repo của họ — đừng mượn nhãn đó).

### 2b-bis. Hai bản ShadowKV — khác nhau ở đúng một chỗ

Cả hai nằm trong `models/kv_cache.py`, chọn bằng `--method`:

| | `shadowkv` (`ShadowKVCache`) | `shadowkv_cpu` (`ShadowKVCache_CPU`) |
|---|---|---|
| V toàn phần nằm ở | **GPU** | **pinned host** |
| K | low-rank `U/SV` trên GPU, dựng lại mỗi bước | giống hệt |
| chọn chunk | softmax fp32 → **ép bf16** → max theo group | softmax fp32 → `amax` theo (group, q) |
| PCIe mỗi bước decode | 0 | V của budget |
| tác giả tự mô tả | *"only for accuracy measurement, not for efficiency"* | *"the efficient implementation"* |

Ba điều dễ hiểu nhầm:

* Ở bản GPU, thuộc tính tên là `v_cache_cpu` nhưng cấp phát `device=self.device`
  — **nó nằm trên GPU**. Tên là di chứng của code tác giả, không phải mô tả.
* Hai công thức chọn khác nhau về mặt chữ nhưng **ở decode `q_len=1` chúng
  trùng nhau**, chỉ lệch do bản GPU ép bf16 trước khi `max`. Nên accuracy về
  cơ bản như nhau; runtime và bộ nhớ thì khác hoàn toàn.
* Bản GPU **cũng** seal landmark trong decode (`_append_stream_landmarks` trong
  `update_kv_cache`). Bảng §2 trước đây ghi ShadowKV "không index khi decode" —
  sai cho cả hai bản.

Chọn cho campaign: **`shadowkv_cpu`**. Nó là bản tác giả gọi là hiệu quả, nó
khớp với khung offload đang chuẩn hoá, và nó là bản duy nhất lên được 128K.

### 2c. Khung token dùng chung (`streaming_blocks.py`)

```
[ sink ][ retrievable: đã index ][ local ][ buffer 0..interval )
  exact        chọn top-k          exact    exact, chưa index

pending_buffer = (total − update_origin) % update_interval
recent_tokens  = local_tokens + pending_buffer        # đuôi exact, ĐỘNG
active_blocks  = min(sealed, (total − recent_tokens) // block_size)
```

`update_origin` = độ dài prompt, đặt ở `reset()`, nên buffer đếm **token sinh
ra**. Flush tự xuất hiện từ số học, không cần code riêng.

Một dòng điều khiển cả bảy method:
```bash
QUEST_PREFIX_TOKENS=32  STREAMING_RECENT_TOKENS=256  STREAMING_UPDATE_INTERVAL=256
```
Không đặt gì thì **mọi method giữ default gốc của tác giả** (RetroInfer 4/64/1024,
MagicPIG 4/64, ShadowKV 32/chunk). Cấu hình thoái hoá `local=recent,
interval=block_size` tái tạo **bit-exact** vòng đời per-block cũ — pin bằng
`tests/test_streaming_frame.py`.

### 2d. Bộ nhớ theo độ dài — Qwen3-4B, card 24 GB

| context | KV | + 8 GB weights | vừa? |
|---|---:|---:|---|
| 32K | 4.50 GB | 12.50 GB | ✅ |
| 64K | 9.00 GB | 17.00 GB | ✅ sát |
| 128K | **18.00 GB** | **26.00 GB** | ❌ OOM |

Ở 128K offload là **bắt buộc**. **RetroInfer sẽ OOM** — nó cấp phát `k_cache`
thẳng trên GPU và không có đường offload. Ba lựa chọn khi tới đó: viết offload,
chạy trên m3 (L40 46 GB), hoặc để NA.

**Ranh giới claim.** Ours / Quest / ParisKV kế thừa `StreamingBlockCache`
(`models/streaming_cache.py` + `models/streaming_blocks.py`): block được *seal*
khi đủ `block_size` token theo thứ tự thời gian — **kể cả token vừa sinh ra** —
rồi *active* khi rơi ra ngoài cửa sổ recent exact. RetroInfer có cơ chế riêng
theo đoạn 1024. ShadowKV (cả hai bản) seal landmark theo đợt trong decode —
**đây là mở rộng của ta**, không phải paper. PQCache gán mã PQ mỗi bước
(`_assign_codes`). Chỉ **MagicPIG** dựng index **một lần ở prefill** và không
bao giờ dựng lại; token sinh ra sau đó chỉ nằm trong cửa sổ recent. Đúng paper,
nhưng trên benchmark sinh dài (AIME 32K, GPQA 16K, MATH-500 4K) nó nghĩa là
MagicPIG mất nội dung nó tự sinh ra — phải ghi rõ, hoặc để NA.

ParisKV dùng **author components** chạy trên common path. RetroInfer là **bản tự
cài lại của ta** (`models/retroinfer_reference.py`), không phải code tác giả —
nhãn phải là *reference reimplementation*.

### 2e. Residency — một chính sách cho mọi method: **luôn offload**

KV nằm ở đâu là **lựa chọn triển khai, không phải thuật toán**. Nên nó phải
giống nhau giữa các method, nếu không bảng runtime đang so hai chế độ chứ không
so method.

⚠ **SỬA LẠI 2026-09-13.** Bản cũ của mục này ghi "≤64K để KV trên GPU, 128K mới
offload". **Sai cả hai vế, và chưa từng được kiểm.**

* **Không vừa.** Ở 64K GPU-resident, ours / Quest / ParisKV **OOM** trên card
  24 GB (một process dùng 23.31 GiB). Chỉ ShadowKV và RetroInfer sống.
* **Và cũng không nhanh hơn.** Ở 32K, nơi cả năm method đều vừa, offload vẫn
  **nhanh hơn**: ours prefill **15.05 s** so với 27.48 s, decode median
  **43.11 ms** so với 49.91, đỉnh bộ nhớ **10.91 GiB** so với 15.47 — mà lần
  chạy GPU-resident còn được card nguội hơn (40°C so với 49°C).

⚠ **SỬA LẠI 14/09/2026 lần hai — vế "và cũng không nhanh hơn" ở trên là SAI.**
Đo lại 4 lần xen kẽ thứ tự, cùng xuất phát 53–54 °C: GPU-resident **prefill
10.0 s so với 15.6 s** (nhanh hơn 35%), decode median 41.6–43.2 so với
36.5–37.7 ms (chậm hơn 13%). Với RULER, gen ~128 bước nên **prefill quyết định**
⇒ resident nhanh hơn ~24% mỗi mẫu. Con số 27.48 s trong bảng trên là **một lần
đo không lặp lại**, và chênh decode 16% nằm dưới ngưỡng 20% mà kỷ luật đo bắt
phải lặp mới tin.

Decode chậm hơn thì có cơ chế: gather UVA hợp nhất và **tái dùng chéo bước chỉ
chạy trên đường offload** (`_can_reuse_selected_blocks()` đòi `self.offload`).
Prefill nhanh hơn thì chưa có lời giải đã kiểm chứng.

⇒ **Vẫn luôn offload, nhưng vì BỘ NHỚ, không phải tốc độ**: ở 64K GPU-resident
ours/Quest/ParisKV **OOM** trên card 24 GB. Trên card 46 GB của m3 thì resident
đáng đo lại — nhưng **không trộn hai chế độ trong một bảng**.

⇒ **`STREAMING_OFFLOAD=1` cho mọi độ dài**, đã là default của `run_cell.sh`.
ShadowKV do đó dùng **`shadowkv_cpu`** ở mọi độ dài — trùng luôn với quyết định
cho RULER (ở `gen_len` ≤ 128 phần mở rộng không bao giờ kích hoạt vì flush cần
256 token, nên nó trùng method đã công bố).

`STREAMING_OFFLOAD` điều khiển **cả PQCache** (trước đây hard-code pinned CPU).
Đã xác minh hai chế độ cho ra prediction **bit-exact** — cùng md5, cell
`qwen3_8192_niah_single_1_pqcache_author_common_…_x32_l256` (2 mẫu).

**MagicPIG không có lựa chọn** — attention lấy mẫu chạy trên CPU của tác giả;
đó *là* method.

### 2e-bis. Bản mặc định của mỗi method cho benchmark

Chốt 2026-09-13. Mỗi method mặc định về **bản nhanh nhất của chính nó**, và mọi
knob làm đổi số đều nằm trong tên ô.

| method | tên gọi | đường nhanh | công tắc |
|---|---|---|---|
| ours | `adaptive_centroid_lse_streaming_prefix4_querymean` | router **Triton** (`packed_block_logits`) | `STREAMING_ROUTER_BACKEND`, **default đổi torch → triton** |
| Quest | `quest_streaming` | `quest_page_scores` (Triton) | không có — luôn Triton |
| ParisKV | `pariskv_author_common` | kernel tác giả (SRHT, collision, radix top-k, RaBitQ) | không có |
| RetroInfer | `retroinfer_author_common` | kernel tác giả (`batch_gemm_softmax`, `gather_copy_vectors`, `weighted_flash_decoding`) | không có |
| ShadowKV | `shadowkv_cpu` | khung chung + kernel gather chung | tên method |

Tất cả đều chạy **offload** (§2e) và **có reuse chéo bước** (`STREAMING_GATHER_REUSE=1`).

**Router backend của ta: tất định nhưng hai bản KHÁC NHAU.** Đo ở 32K b1024:
mỗi backend chạy hai lần cho **cùng md5**, nhưng torch ≠ triton. Khác biệt
nằm trong nhiễu — cwe 0.785 (torch) so với 0.780 (triton) trên 20 mẫu,
`niah_multikey_3` 1.000 cả hai — còn tốc độ thì triton **nhanh 1.6×** (decode
median 43.0 so với 70.5 ms ở 64K B=4096). Nên default là triton, và **backend
nằm trong tên ô** (`_rbtriton` / `_rbtorch`) để không bao giờ trộn hai bản vào
một bảng. Ô cũ (không có hậu tố) là ô đo trước khi phân biệt này tồn tại.

⚠ **BUG 14/09/2026 (đã sửa `1446d17`)**: `_reset_metadata` không xoá
`empty_cluster`, nên prompt ngắn sau prompt dài thừa hưởng cờ cụm "sống" cũ và
truy hồi vào slot của mẫu trước. Lộ ra ở LongBench-v2: RetroInfer **0.039** so
với 0.356 của `full`. Mọi số RetroInfer trước ngày này **không dùng được**.

⚠ **RetroInfer KHÔNG tất định.** Chạy hai lần cùng cấu hình cho **md5 khác
nhau** (32K cwe b1024 8 mẫu: 0.825 và 0.850) vì k-means của tác giả dùng
`tl.atomic_add`. Giống ParisKV. **Không dùng hai method này làm cổng
bit-exact**, và chênh lệch ≤1 điểm giữa hai lần chạy của chúng là nhiễu.

### 2f. Reuse giữa hai bước — và nó công bằng tới đâu

Kernel block của ta giữ lại block mà bước trước đã nạp và chỉ kéo phần thay
thế. **Đo thật ở 32K, budget 1024, block 8, `niah_single_1`:**

| | block tái dùng | prediction |
|---|---|---|
| ours | **62.8%** (532 405 / 847 872) | md5 giống hệt bản tắt reuse |
| Quest | **60.4%** (512 304 / 847 872) | md5 giống hệt bản tắt reuse |

Hai điều rút ra:

* **Reuse không đụng tới accuracy** — bật hay tắt cho ra cùng một byte. Nên
  campaign accuracy cứ bật (nhanh hơn ~5.8×); chỉ bảng runtime mới cần tắt.
* **Giữa ours và Quest nó đối xứng** (62.8 vs 60.4), nên không method nào được
  lợi. Bất đối xứng nằm ở chỗ khác: **method chọn theo block có reuse, method
  chọn theo token thì không** — ParisKV và PQCache đi nhánh per-token và ta
  chưa cài reuse cho chúng.

> **Quyết định: BẬT reuse, mọi lúc, mặc định.** Reuse giữa hai bước là kỹ
> thuật đã công bố của chính dòng literature này — ShadowKV có nó trong paper
> (`reorder_keys_and_compute_offsets` + `gather_copy_with_offsets`). Tắt nó đi
> để tránh một lời chê giả định sẽ làm số của ta **tệ hơn số ShadowKV đã công
> bố**, và đó mới là bảng sai.

Cách xử lý đúng khi baseline có thể nhanh hơn nếu được tối ưu là **nói ra**,
không phải tự làm chậm mình. Trong bảng ghi rõ method nào có reuse:

| method | reuse | vì sao |
|---|---|---|
| ours, Quest | ✅ ~60% | chọn theo block → id địa chỉ hoá được, so 128 id/bước |
| ShadowKV-CPU | ✅ **54.4%** | dùng kernel block của ta với `block_size = chunk_size` (§2g) |
| ParisKV | ❌ (có thể có — §2h) | tác giả không cài; độ trùng thật **58.3%** |
| PQCache | ❌ | tác giả không cài; chưa đo độ trùng |
| RetroInfer, ShadowKV-GPU | — | không có PCIe |

`STREAMING_GATHER_REUSE=0` giữ lại làm **cột chẩn đoán**: nó tách riêng hiệu ứng
bề rộng record (2 KB/block so với 256 B/token, 11.31 vs 2.96 GB/s) khi có người
hỏi. Đó là câu trả lời cho lời chê, không phải bảng chính.

Đã trả lại reuse cho ShadowKV-CPU — xem §2g.

`STREAMING_REUSE_STATS=1` in ra tỉ lệ tái dùng khi kết thúc tiến trình.

### 2g. ShadowKV-CPU dùng chung kernel offload

Kho pinned của ShadowKV là `[B,H,chunks,chunk*D]` — **đúng cùng số byte, cùng
thứ tự** với `[B,H,tokens,D]`, vì trong một record các token nằm liên tiếp. Nên
kernel block của ta đọc thẳng nó với `block_size = chunk_size`, và **chunk id
của tác giả chính là block id của kernel**. Khoá không đi qua đâu cả: ShadowKV
dựng K từ low-rank trên GPU, nên có thêm entry point chỉ-values
(`gather_blocks_reuse_values_uva`) để không kéo K qua PCIe rồi vứt đi.

Kernel của ta so khớp id **trên GPU ngay trong launch**, không dựa vào sổ sách
neo ở prefill như `reorder_keys_and_compute_offsets`. Vì thế nó reuse được dù
streaming id đã phá sổ sách đó — cái đã buộc bản cũ phải gọi `cnts.zero_()`.

Đo ở 32K, budget 1024 (ms mỗi layer, chuyển 2 MB khi cold):

| | 0% | 60% | 90% |
|---|---|---|---|
| chunk 4, kernel của ta | **0.211** | 0.120 | 0.061 |
| chunk 8, kernel của ta | **0.184** | 0.086 | 0.038 |
| chunk 8, kernel tác giả | 0.358 | — | 0.051 |
| `torch.gather` trên CPU (bản cũ) | 0.630 | — | — |

Ba điều:

* chunk 4 giờ **có offload tử tế**: 0.211 ms thay vì 0.630 — nhanh **3.0×**, và
  5.3× ở mức reuse thật.
* Kernel của ta **nhanh hơn kernel tác giả ngay ở chunk 8 gốc** (0.184 vs
  0.358). Không còn đánh đổi giữa "đúng paper" và "chạy nhanh".
* Reuse thật của ShadowKV ở 32K: **54.4%** (922 580 / 1 695 744), chunk 4.

Xác minh **bit-exact**: ô `qwen3 8192 niah_single_1 shadowkv_cpu b1024` cho md5
`8599ed6a3c74` — giống hệt đường `torch.gather` cũ. Đây là đổi **cách chuyển dữ
liệu**, không đụng thuật toán.

`clear()` phải `stage_ids.fill_(-1)`: id cũ còn sót sẽ khiến kernel chép lại
hàng của mẫu trước cho một block id tình cờ trùng.

### 2h. ParisKV — hạ tầng offload của chính tác giả

ParisKV **có** offload riêng, và nó gần trùng với của ta:

* `cache_hub/polar_cache.py`: `unified_keys_cpu` / `unified_values_cpu` là
  pinned CPU giữ toàn bộ KV; sink + local ở lại GPU (`unified_keys_gpu`).
  `enable_offload` mặc định True.
* Fetch bằng kernel CUDA riêng `h2d_gather_kv`
  (`cache_hub/gather_trans/trans_h2d.cu`). Comment của họ: *"UVA H2D Gather:
  GPU indices + pinned CPU src -> GPU dst"*.
* Kernel: grid `(bs, heads, topk)`, **32 thread/block** — thread 0–15 chép K
  (16 × `uint4` = 256 B), 16–31 chép V. **Per-token, record 256 B.**

Tức là hai bên viết độc lập ra **cùng một thiết kế**. Bản của ta faithful.

**Không có reuse** ở bản chính thức: không lưu id bước trước, không so khớp,
không bank. Mỗi bước kéo lại toàn bộ top-k.

Và **đơn vị truy hồi của ParisKV thật sự là token**: `collision_based_topk_batch`
trả `topk_indices: [bs, kv_heads, final_topk]`, RaBitQ rerank từng key. Config
Qwen của họ có `cache_unit_size: 8` nghe như block, nhưng `polar_cache.py`
**nhận tham số đó rồi không dùng ở đâu** (chỉ có ở dòng signature 119) — knob chết.

**Nhưng reuse vẫn giúp được nó.** Độ trùng tập token giữa hai bước decode, đo ở
32K: **58.3%** (3 782 420 / 6 488 064) — cùng vùng với ours 62.8%, Quest 60.4%,
ShadowKV 54.4%. Độ ổn định của lựa chọn **không** phụ thuộc vào đơn vị truy hồi.

Và kernel reuse hiện có chạy được luôn ở mức token — `block_size=1` biến block
id thành token id, **không cần viết kernel mới**:

| | ms/layer |
|---|---|
| `gather_kv_uva` per-token, không reuse | 1.215 |
| kernel reuse, `block_size=1`, 0% trùng | 1.159 |
| kernel reuse, `block_size=1`, 57% trùng | **0.629** |

Vòng so khớp O(K) nối tiếp trong thread 0 **không** thành nút cổ chai ở K=1024
— nó bị latency PCIe che đi (1.159 vs 1.215 khi không có gì để tái dùng).

⇒ ParisKV và PQCache **có thể có reuse với ~1.9× nhanh hơn**, dùng đúng kernel
đang có. Chưa cài. Đây là quyết định về mức độ trung thành với baseline, không
phải vấn đề kỹ thuật.

### 2i. RetroInfer — code tác giả ĐÃ có, bản của ta thiếu phần cốt lõi

Repo chính thức là **`microsoft/RetrievalAttention`**, không phải
`microsoft/RetroInfer` (URL đó không tồn tại). Đã clone về
`/home/baonn/upstream-kv-methods/RetrievalAttention`, commit `75829e63`
(2026-07-30). **Cấu trúc giống hệt ParisKV-official** (`cache_hub`, `attn_hub`,
`model_hub`, `config`) — ParisKV xây trên chính codebase này, nên adapter ta
viết cho ParisKV dùng lại được.

**SỬA LẠI: RetroInfer CÓ offload — đó là toàn bộ điểm của paper.** README của
họ: *"rethinks the KV cache as vector storage within a GPU–CPU co-execution
setup"*, *"the wave buffer coordinates KV cache placement and overlaps
computation and data transfer across GPU and CPU"*. Họ có hai cài đặt:
`cache_hub/retroinfer_cache.py` (953 dòng, wave-buffer GPU–CPU) và
`retroinfer_cache_gpu.py` (744 dòng, chỉ GPU). Config có `gpu_only: false` —
**mặc định là offload**.

Method thật, đọc từ `sparse_attention()`:

1. `batch_gemm_softmax(queries, centroids)` trên **8192 centroid**, cộng dồn qua
   query head chung KV head, mask cụm rỗng, rồi `topk`.
2. **Vùng ước lượng**: `gather_copy_vectors` chỉ lấy **centroid, `value_sum`,
   `cluster_size`** — ba vector mỗi cụm, không phải token.
3. **Vùng truy hồi**: `nprobe` cụm đầu, nạp token thật.
4. Gộp bằng `weighted_flash_decoding` với `previous_out`/`previous_lse` — **một
   softmax duy nhất**.

`WaveBufferCPU` là class **C++** trên thread pool
(`library/retroinfer/retroinfer_kernels/src/wave_buffer_cpu.cpp`), giữ **LRU
cache trên GPU**: mỗi bước chia cụm cần dùng thành **hit** (đã ở GPU) và **miss**
(kéo từ CPU). `gather_copy_and_concat` nạp theo **đơn vị cụm kích thước biến
thiên**, không phải token rời rạc; `gather_copy_and_scatter` nạp page vừa dùng
vào GPU cache cho bước sau.

**Ngân sách của họ là tỉ lệ, không phải số token.** Config Qwen2.5-7B:

| tham số | của họ | bản của ta |
|---|---|---|
| `static_pattern_start` / `_end` | 4 / 64 | prefix 4 / recent 64 ✅ |
| `estimation_budget` | 0.232 | `estimation_ratio` 0.232 ✅ |
| **`retrieval_budget`** | **0.018** | ❌ dùng `sparse_budget` cố định |
| `n_centroids` / `n_segment` | 8192 / 16 | ❌ `average_cluster_size` 16 |
| `cache_ratio` / `buffer_cluster_num` / `pages_per_cluster` | 0.05 / 32 / 2 | ❌ không có |

Ở 64K, `retrieval_budget` của họ là 0.018 × 65536 ≈ **1180 token exact**. Đặt
RetroInfer vào cột "B=4096" là chạy một cấu hình không có trong paper.

**Bản `retroinfer_reference.py` của ta chỉ giữ phần toán.** Không wave buffer,
không LRU, không offload, không thread pool. Docstring của chính nó vẫn ghi
*"Do not register it as a production retroinfer method until its cluster choices
and outputs have been compared on captured model tensors"* — chưa ai gỡ, chưa
thấy bằng chứng đã đối chiếu.

### 2i-bis. RetroInfer đã lên common path — `retroinfer_author_common`

**ĐÃ LÀM (2026-09-13).** `ShadowKV/models/retroinfer_author_streaming_cache.py`,
`StreamingRetroInferAuthorCache(StreamingBlockCache)`, method
`retroinfer_author_common`. Bản `retroinfer_reference_streaming` vẫn còn, làm
test tương đương, **không đứng trong bảng**.

**Component của họ được nạp, không chép — cả bốn phần.**

| phần của method | code chạy |
|---|---|
| k-means cầu phân đoạn, `value_sum`, reverse index | `cache_hub/kmeans.py` (Triton) của họ |
| sắp lại KV theo cụm trên kho offload | ý của `construct_func` trong `wave_buffer_cpu.cpp` |
| chấm điểm cụm | `retroinfer_kernels.batch_gemm_softmax` (CUTLASS) |
| gom centroid/`value_sum`/`cluster_size` vùng ước lượng | `retroinfer_kernels.gather_copy_vectors` |
| attention hai vùng + gộp một softmax | `weighted_flash_decoding` (fork của họ) |

Build: cutlass v3.5.1 clone vào `library/`, `CUDA_HOME` trỏ **conda env** (nvcc
12.4 — nvcc hệ thống là 13.0, torch build bằng 12.4 nên pip từ chối). Fork
flash-attention cài dưới tên **`weighted_flash_decoding`** + ext
`weighted_flash_attn_cuda`; `find_packages` của nó **chỉ lấy**
`weighted_flash_decoding`, nên `flash_attn` 2.7.4 mà sáu method kia đang dùng
**không bị đụng** — đã kiểm sau khi cài.

`cache_hub/kmeans.py` nạp theo **đường file** chứ không `import cache_hub.kmeans`:
`__init__.py` của họ kéo cả `retroinfer_cache`, cần kernel **và** fork, nên
import theo package sẽ hỏng ở máy chưa build.

**Hai ràng buộc của kernel họ, phải tuân:**

* `VECTOR_SIZE_CP` trong `copy_kernel.cuh` **ghim head_dim = 128**. Model khác
  sẽ bị đánh sai chỉ số chứ không báo lỗi — adapter chặn bằng `ValueError`.
* trục cụm là chiều GEMM của `batch_gemm_softmax`, phải canh biên bội số
  `lcm(8, n_segment)`. Không canh thì **không fail lúc launch** mà fault
  `misaligned address` giữa chừng decode. Nên trục cụm ở đây luôn bằng **sức
  chứa** của index, cụm chưa dựng là cụm rỗng và đã bị mask bằng `dtype_min` —
  tránh luôn việc cấp phát lại mỗi flush như cache của họ làm.

**Vận chuyển là của họ.** `construct_func` sắp lại vật lý KV trên CPU để vector
của một cụm nằm liền nhau, rồi đường miss đọc **page `page_size` vector** từ
`CPUStartIndex`. Ở đây vùng chỉ mục của kho backing cũng được ghi theo thứ tự
cụm, nên bản ghi PCIe là **page 2 KB**, không phải token rải 256 B. Đo được
gather của nó **nhanh hơn Quest** (95 so với 112 ms / 8 bước). Cụm xếp khít
không đệm, và **page được chấm điểm** (lấy max điểm các cụm trong page) thay vì
chọn theo cụm — nhờ đó ngân sách đúng bằng B token, không slot đệm nào bị
attend, không page nào lấy hai lần.

**Không port** LRU wave buffer C++: `gather_blocks_reuse_uva` dùng chung đã giữ
lại page bước trước lấy, đúng vai của LRU đó.

**Đo được estimation zone có tác dụng** (qwen3, 16K, B=64, 20 mẫu, offload, ô
phân biệt được): cwe **0.825** so với **0.705** khi tắt; fwe 0.967 so với 0.950.

**Tốc độ decode** (64K, B=4096, offload, đúng env protocol, tuần tự một card):

| | decode mean | median |
|---|---|---|
| Quest | 43.48 | 43.10 |
| ours | 46.22 | 43.00 |
| **RetroInfer** | **54.43** | **53.64** |

Đường đi tới đó: bản PyTorch đầu tiên **108.87 ms** → bỏ đồng bộ GPU→CPU mỗi
layer **74.53** → kernel tác giả **54.43**.

**Ba chỗ lệch paper, không chỗ nào im lặng:**

1. **Ngân sách.** Paper đặt `retrieval_budget` là **tỉ lệ cụm** (0.018), số token
   thả nổi. Ở đây B là **số token mỗi KV head**, dùng chung cả bảng — nên lấy
   nguyên cụm theo thứ tự điểm cho tới khi đủ B token, cụm vắt ngang biên bị cắt.
   Cụm vắt ngang đó **bị loại khỏi estimation zone**: ước lượng nó chồng lên
   phần token đã truy hồi từ chính nó là **đếm hai lần**.
2. **Khung.** Steady zone của paper là `static_pattern_start + _end`, index lớn
   thêm mỗi `UPDATE_SEGMENT`=1024 token sinh. Ở đây sink/local và nhịp flush là
   của khung chung, nên index lớn thêm mỗi `update_interval`. Index được kéo dài
   **đúng bằng candidate range** ở mỗi flush, nên **không token nào vừa nằm
   trong cụm vừa được attend exact** — test `test_frame_partitions_the_context`
   kiểm điều này ở mọi bước.
3. **Số cụm.** Paper ghim `n_centroids`=8192 bất kể độ dài. Ở đây mặc định
   `tokens // average_cluster_size` để index tốn bộ nhớ tỉ lệ với thứ nó đánh
   chỉ mục; truyền `--retroinfer_n_centroids 8192` để dựng lại con số của họ.

**Wave buffer thì sao?** Không cài lại. LRU cache trên GPU của họ và
`gather_blocks_reuse_uva` của ta **là cùng một ý**: giữ lại thứ bước trước đã
kéo, chỉ nạp phần thay đổi. Vì `block_selection = False`, RetroInfer chạy
gather chung ở đơn vị token, đúng đường ParisKV và PQCache đi.

**Cổng kiểm**: `repro/shadowkv/test_retroinfer_author_common.py` — 4 test, cần
CUDA (k-means của họ là Triton). Phân hoạch khung; reverse index đảo ngược khớp
`cluster_size`/`value_sum` của chính họ; ngân sách phủ hết thì bằng dense; phép
gộp estimation bằng **một softmax duy nhất** trên hợp hai tập.

## 3. Đội máy — phần riêng của project này

> Đường vào, mount lớn, env, bẫy từng máy: skill `research_conduction` §3.1.
> Ở đây chỉ ghi cái riêng của project.

| | m1 `fatchoy` | m2 `sepc810` | m4 `sashimi` | m3 (chỉ user) |
|---|---|---|---|---|
| vai trò | **HUB + push** | pull, campaign | pull, campaign | **chưa migrate** |
| repo | `/home/baonn/sparse_attention` | `/home/nbnguyen/sparse_attention` | `/home/baonn/sparse_attention` | — |
| GPU dùng được | 1–7 (GPU0 của user, nóng) | 1–4 (GPU0 = T400, cấm) | 0–1 | — |
| results | `/storage/baonn/sparse_attention_results/` | `/home/nbnguyen/sparse_attention_results/` | `/storage/nbao/sparse_attention_results/` | — |
| env | `repro/shadowkv/env_m1.sh` | `env_m2.sh` | `env_m4.sh` | — |

**Chỉ m1 push.** m2/m4 `git pull` — không sửa nội dung ở máy phụ. Sửa ở m1,
commit, push, máy phụ pull.

**Đường vào m2** (chi tiết và chẩn đoán: skill `research_conduction` §3.1). m2
**không vào bằng hostname** — `sepc810.se.cuhk.edu.hk` không nối thẳng từ m1
được, và m4 cũng không. Nó nằm sau reverse tunnel do **Mac** mở vào loopback m1:

```bash
# mọi lệnh trên m2 đi qua socket này
ssh -S ~/.ssh/sockets/m2.sock -p 2222 nbnguyen@localhost '<cmd>'

# master chết mà 127.0.0.1:2222 còn listen thì tự dựng lại (password ở
# ~/.ssh/m2_askpass.sh, không bao giờ chép ra log/lệnh)
rm -f ~/.ssh/sockets/m2.sock
SSH_ASKPASS=~/.ssh/m2_askpass.sh SSH_ASKPASS_REQUIRE=force setsid -w \
  ssh -M -S ~/.ssh/sockets/m2.sock -N -f -p 2222 -o ControlPersist=yes \
      -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
      -o PreferredAuthentications=password nbnguyen@localhost
```

⚠ Cổng 2222 **còn listen mà treo ở `banner exchange`** nghĩa là **Mac chưa vào
mạng trường**, không phải tunnel chết — chờ user nối mạng, đừng dọn socket.

Ba máy đã xác minh cùng `torch 2.6.0+cu124`, `transformers 4.55.4` (2026-09-13).

**m3 chưa migrate.** Nó đang có HAI checkout (`$BASE/s2-ttt` nhánh
`machine3-48gb` và `$BASE/shadowkv-research`) và HAI env
(`envs/kvpress-torch240-cu121` tf 5.2.0, `envs/shadowkv`), chạy Qwen2.5-14B và
đang tải DeepSeek-R1-Distill-Llama-8B. `BASE=/home/zhufangzhou/workspace/sheruifeng/baonn/baonn`.
Máy chỉ-đưa-lệnh, không internet cho HF hub, git fetch được.

### 3b. Dựng một máy mới — thứ tự đã kiểm (m3, 14/09/2026)

Repo `git clone` xong **chưa build được**: `ShadowKV/3rdparty/` nằm trong
`.gitignore`, mà `ShadowKV/setup.py` lại đòi `3rdparty/cutlass/include`. Clone
sạch sẽ chết với `fatal error: cutlass/cutlass.h: No such file or directory`,
và traceback Python chỉ nói "Error compiling objects" — phải `grep 'error:'`
trong log mới thấy nguyên nhân.

```bash
BASE=<thư mục làm việc trên máy đó>
git clone https://github.com/nguyenngocbaocmt02/sparse_attention.git $BASE/sparse_attention
git clone https://github.com/microsoft/RetrievalAttention.git      $BASE/RetrievalAttention

# 1. cutlass v3.5.1 — BẮT BUỘC, bị gitignore, dùng chung cho cả hai bộ kernel
mkdir -p $BASE/sparse_attention/ShadowKV/3rdparty
git clone --depth 1 -b v3.5.1 https://github.com/NVIDIA/cutlass.git \
  $BASE/sparse_attention/ShadowKV/3rdparty/cutlass
ln -sfn $BASE/sparse_attention/ShadowKV/3rdparty/cutlass $BASE/RetrievalAttention/library/cutlass

# 2. dữ liệu RULER: symlink từ checkout cũ nếu có, đừng sinh lại
#    (cùng byte ⇒ cùng mẫu ⇒ số ghép được với các máy khác)

# 3. kernel ShadowKV
cd $BASE/sparse_attention/ShadowKV
CUDA_HOME=<cuda khớp torch> TORCH_CUDA_ARCH_LIST=<sm của card> MAX_JOBS=16 \
  $ENVP/bin/python setup.py build_ext --inplace

# 4. kernel RetroInfer + fork flash-attention (chỉ cần cho retroinfer_author_common)
export PATH=$ENVP/bin:$PATH        # thiếu dòng này pip không tìm thấy python3.10
cd $BASE/RetrievalAttention/library/retroinfer && $ENVP/bin/pip install --no-build-isolation .
cd $BASE && git clone --depth 1 -b weighted https://github.com/Starmys/flash-attention.git wfd
cd wfd && FLASH_ATTENTION_FORCE_BUILD=TRUE $ENVP/bin/pip install --no-build-isolation .

# 5. kiểm — PHẢI cd ra khỏi thư mục source trước
cd /tmp && $ENVP/bin/python -c "
import torch, retroinfer_kernels
from weighted_flash_decoding import weighted_flash_decoding
import flash_attn; print('OK', flash_attn.__version__)"
```

Ba cái bẫy, cả ba đều đã cắn:

* **`CUDA_HOME`** phải khớp **major** với torch, không phải nvcc hệ thống. m1 có
  nvcc 13.0 nhưng torch cu124 ⇒ trỏ vào nvcc 12.4 trong conda env. m3 có nvcc
  12.0 với torch cu124 — lệch minor, torch chỉ cảnh báo, build được.
* **`PATH=$ENVP/bin`** — thiếu thì pip chết với `/usr/bin/env: 'python3.10': No
  such file or directory` trong shell không tương tác (m4 hỏng vì cái này).
* **`cd /tmp` trước khi kiểm import** — đứng trong thư mục source của fork thì
  `import flash_attn` bắt phải gói cục bộ của fork và báo **sai** phiên bản.
  Tôi đã hai lần tưởng fork ghi đè mất `flash_attn` chỉ vì chuyện này.

Fork cài dưới tên `weighted_flash_decoding` + ext `weighted_flash_attn_cuda`;
`flash_attn` **không bị đụng** — đã xác minh trên cả bốn máy, đều 2.7.4.post1.

## 4. Protocol đo — hằng số thiêng

- vùng exact: **prefix 32 + recent 32**, `UPSTREAM_MATCHED_EXACT_REGIONS=1`
- budget B là **số token mỗi KV head**; ShadowKV thì cờ ≠ budget (cộng
  `outlier_chunk × chunk_size` + local) — dùng `budget_audit.py` để ghép
- sinh (mọi reasoning bench): **temperature 0.6, top_p 0.9, KHÔNG top_k**,
  `max_new_tokens` theo bench. Đây là protocol của project tiền nhiệm
  (`s2-ttt/repro/reasoning/run_reasoning_pool.sh`: `TEMPERATURE=0.6 TOP_P=0.9`,
  không có tham số top_k) và là hằng số thiêng — đổi nó là đổi nghĩa của mọi
  bảng reasoning từng báo cáo.
- `CUDA_DEVICE_ORDER=PCI_BUS_ID` bắt buộc, trong POOL chứ không chỉ launcher

⚠ **2026-09-13**: campaign 12/09 đặt GPQA/MATH500 ở `top_p 0.95 / top_k 20`
trong khi AIME25 giữ `0.9 / không top_k`. **AIME mới là cái đúng protocol**;
GPQA/MATH500 là cái lệch. Mọi số GPQA/MATH500 của campaign đó không ghép được
với AIME, và không so được với bảng reasoning của s2-ttt.

## 5. Benchmark

| bench | datalen | mẫu | ghi chú |
|---|---|---|---|
| LongBench-v2 short | 131072 | 180 | lọc bằng `SHADOWKV_LONGBENCH_V2_LENGTH_FILTER=short` |
| LongBench-v2 medium | 131072 | 215 | `=medium` |

**Nối chuỗi khi rời máy**: `run_after_pool.sh --wait <state> <queue> [--wait
...] -- <lệnh>` chờ pool xong rồi chạy lệnh. Chạy bằng `setsid nohup`, nếu
không nó chết theo phiên SSH. Điều kiện chờ là **marker**, không phải tiến
trình: `running/` rỗng **và** `done + failed >= số dòng queue`. Ô hỏng tính là
đã xong — chặn campaign sau vì ba ô hỏng nghĩa là sáng ra máy không làm gì cả.
Phải `--wait` **mọi** pool mà lệnh sau sẽ đụng state dir, vì `run_pool` giữ
`flock` độc quyền ở đó và pool mới sẽ từ chối.

Quét budget LongBench-v2: `LBV2_BUDGETS=4096,2048,1024,512
run_longbench_v2_campaign.sh`. Nó **nối thêm** vào queue đang sống chứ không
ghi đè — ô đã có marker `done` thì pool bỏ qua, nên 12 ô ở 4096 không chạy lại.
`full` chỉ sinh **một lần** cho cả quét: dense không đọc budget, thêm một ô
`full` ở budget khác là cùng một phép tính dưới tên khác, mà nó lại là ô đắt
nhất. Thứ tự là hết một budget rồi mới sang budget sau, để cắt ngang chừng vẫn
còn một bảng hoàn chỉnh thay vì ba bảng dở.
| RULER | 4K–128K | 13 task | budget thường đặt theo tỉ lệ L/32 |
| GPQA diamond | 32768 | 198 | max_new 16384 |
| MATH-500 | 32768 | 500 | max_new 4096 |
| AIME25 | 32768 | 30 | max_new 32768, **nhiều seed** |

Số seed đến từ cỡ tập test: AIME 30 câu → phương sai lớn → nhiều seed. GPQA/MATH
đủ lớn để một seed, nhưng khoảng cách ≤1 điểm vẫn là nhiễu.

### 5a. Reasoning — harness và ba cái bẫy (14/09/2026)

Ba bench này **lật ngược** bối cảnh long-context: prompt vài trăm token, còn
chuỗi mà method phải index là **sinh của chính model**, được frame gộp vào index
mỗi `STREAMING_UPDATE_INTERVAL` token. Nên một ô ở đây đo *sparsity lúc decode
trên ngữ cảnh tự sinh*, và giá của nó do generation quyết định, không phải prompt.

```bash
source repro/shadowkv/env_m3.sh
repro/shadowkv/run_reasoning_campaign.sh smoke     # 4 mẫu/ô
$PY repro/shadowkv/report_reasoning.py $ROOT/smoke --cost
repro/shadowkv/run_reasoning_campaign.sh full
$PY repro/shadowkv/report_reasoning.py $ROOT/full
```

Nút vặn: `REASONING_TASKS/METHODS/SEEDS/DATALEN/BUDGETS`,
`REASONING_SMOKE_SAMPLES`.

⚠ Sinh queue mà không chạy thì dùng `run_reasoning_campaign.sh full queue`,
**đừng** đặt `SHADOWKV_POOL_GPUS=""`. `run_pool.sh` gieo file roster
`.state/gpus.txt` từ biến đó, nên roster rỗng để lại một `gpus.txt` rỗng và
lần chạy **sau** kế thừa nó rồi khởi động 0 worker — pool thoát ngay, không in
gì giải thích, trông như campaign xong tức khắc. Đã cắn thật. run_pool giờ từ
chối roster rỗng lúc khởi động (xoá file rỗng đó rồi chạy lại); làm rỗng
`gpus.txt` **trong lúc** pool chạy vẫn là cách rút cạn pool có chủ đích.

**Cấu hình đã chốt (14/09/2026)**: 4 method (**không `full`** — trần dense đã
có sẵn; **không ShadowKV** — xem dưới), 3 task, **hai budget 1024 và 512**, seed
`aime25:4, math500:1, gpqa:1` ⇒ **48 ô**. Budget 1024 (= L/32, khớp RULER) chạy hết trước rồi mới tới 512, nên
bảng đầu tiên đọc được là bảng ở budget neo. Budget là **một chiều của bảng, không
phải một biến thể của method** — gộp hai budget lại là lấy trung bình một lần
chạy 512 với một lần 1024 rồi gọi đó là method.

⚠ **ShadowKV không chạy được ba bench này, và đó là tính chất của method.**
Chỉ mục landmark của nó là **SVD của key trong prompt**, lấy một lần lúc prefill
rồi đóng băng (`SV does not change after prefill`). Prompt reasoning vài trăm
token, ngắn hơn `sparse_budget`, nên cache ở nguyên `dense_warmup` — tức là
**exact** — và khi generation vượt budget thì không có đường dựng cơ sở, chỉ có
`RuntimeError: dynamic SVD transition is not implemented`. Method cần một
prefill dài mới tồn tại được, nên ô đúng trong bảng là **N/A kèm lý do**, không
phải một con số. `gen_reasoning_queue.py` từ chối nó kèm lý do.

⚠ Trần dense chỉ ghép được nếu nó đo ở **temperature .6 / top-p .9 / không
top-k** *và* trên `kvpress_aime25_local`. Campaign 12/09 để GPQA/MATH500 ở
`.95/20` (§4) — nếu số `full` lấy từ đó thì **không ghép được**, phải thêm `full`
lại vào `REASONING_METHODS`.

**Knob per-cell đi trong task key**: `<bench>[-g<max_new_tokens>][-s<seed>]`,
ví dụ `gpqa-g32000-s2`. Env của pool cố định cho cả một queue nên thứ gì đổi
theo ô phải đi trong một trong chín trường queue; nó đi trong task, và task key
nằm nguyên trong tên ô — đó mới là điểm chính: hai seed, hoặc hai trần sinh, mà
trùng tên là hai lần ghi đè lên một jsonl. `run_cell.sh` bóc hậu tố ra
`SHADOWKV_GENERATION_SEED` / `SHADOWKV_MAX_NEW_TOKENS`. Đã kiểm: cùng seed →
prediction trùng byte, khác seed → khác.

**GPQA chạy ở 32000 chứ không phải 16384** của dataset, khớp AIME. Một trần mà
model chạm tới là một biến gây nhiễu: trả lời sai lúc đó có thể do method mất
bằng chứng, cũng có thể chỉ do bị cắt giữa chừng, và bảng không phân biệt được.
Nâng trần gần như không tốn gì vì giá là cái model thực sự viết, không phải cái
nó được phép viết. Đổi lại, số GPQA này **không so được** với bất kỳ lần chạy
GPQA nào ở 16384 — nên trần nằm trong tên ô.

⚠ **AIME25 có HAI parquet và chúng KHÔNG cùng một bench.**
`kvpress_aime25_percontext` **bỏ mất** câu "Remember to put your final answer
within `\boxed{}`" mà `kvpress_aime25_local` có, nên scorer trích `\boxed`
đọc ra con số khác trên cùng một model. Mọi số reasoning đã ghi của project đo
dưới bản `_local`; env_m2/m3/m4 trước đây trỏ vào `percontext` ⇒ ô AIME của
chúng **không ghép được** với m1, mà triệu chứng duy nhất là "m3 thấp hơn".

Cách chữa tận gốc: **ba bench nằm trong repo**, `repro/shadowkv/data/reasoning/`
(332 kB), và cả bốn `env_m*.sh` mặc định trỏ vào đó. Máy mới lấy dữ liệu bằng
`git pull`, không còn bản sao per-machine để mà trôi. Đây là ngoại lệ duy nhất
của luật "dataset không vào repo" và `.gitignore` ghi rõ lý do.
`make_reasoning_data_bundle.sh` vẫn còn cho máy không vào được git.

⚠ **Scorer MATH500 là so khớp chuỗi nguyên văn** (`extract_boxed(pred) ==
str(answer)`, cố ý bám KVPress). Model trả `\boxed{(3, \frac{\pi}{2})}` trong
khi đáp án là `\left( 3, \frac{\pi}{2} \right)` → **chấm 0 dù đúng**. Nó phạt
mọi method như nhau nên vẫn so sánh được, nhưng nó nén dải điểm và thêm nhiễu.
Đổi sang chuẩn hoá tương đương toán học thì mất tính so được với bảng reasoning
của s2-ttt — đó là một quyết định, không phải một bug fix.

## 6. Bộ máy campaign

`repro/shadowkv/`: `run_pool.sh` (claim-pool, flock, marker done/running/failed),
`run_cell.sh` (một ô), `cell_key.py` (**định nghĩa DUY NHẤT của tên ô**),
`preflight.py`, `status.py`, `compare*.py`, `report_upstream_kv_campaign.py`,
`env_m{1,2,4}.sh`.

- Muốn ô chạy lại: **xoá marker, không xoá output dir**
- Dừng: lấy PID trước, **pool trước cell sau**, không bao giờ `pkill -f`
- **Không sửa script shell khi pool đang chạy** — bash đọc theo byte offset
- Trước khi đưa method/cờ mới vào queue: smoke 1 ô bằng **đúng entry point**
  (`repro/shadowkv/run_cell.sh ...`, không phải `bash run_cell.sh`)

## 6b. Chạy test

```bash
cd ShadowKV && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<idle> $PY -m pytest tests/ -q
```

**Phải `cd ShadowKV` trước** — test import `models` theo đường dẫn tương đối; chạy
từ gốc repo thì collection lỗi `No module named 'models'`, trông như code hỏng.
Cũng cần một GPU rảnh: vài test dựng cache thật.

Trạng thái tham chiếu 2026-09-13: **168 passed, 4 failed**. Bốn cái fail nằm ở
`test_streaming_caches.py` (`supports_four_components_on_average`,
`independent_coverage_allocator_does_not_conserve_layer_budget`,
`worst_gqa_coverage_protects_each_sibling`,
`can_shortlist_exact_components`) và **đã hỏng từ trước lần tách repo** — đã chạy
đối chứng trên cây cũ để xác nhận. Thêm fail thứ năm là do thay đổi của bạn.

## 7. Không nằm trong repo

Dataset, model weights, kết quả. Mỗi máy export path riêng trong `env_m<N>.sh`;
kết quả ghi vào `<mount lớn>/sparse_attention_results/` của máy đó — thư mục
riêng của project, để nhiều project trên cùng máy không trộn kết quả. RULER text (317 MB) là artifact per-máy.
