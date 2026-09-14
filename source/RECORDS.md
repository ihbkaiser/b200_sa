# RECORDS — nhật ký chiến dịch

Mục mới lên **đầu file**. Mỗi mục: ngày + trạng thái (⏳ đang chạy / ⭐ chốt /
⚠ SỬA LẠI). Kết luận sai **không xoá** — sửa bằng mục ⚠ để agent sau không tin
lại bản cũ. Định nghĩa chuẩn (method, protocol, máy) ở `PROJECT.md`.

---

## ⭐ 2026-09-14 — whitening theo query: hiệu ứng thật và lớn, nhưng "rank cho 90% năng lượng" là con số sai

Kiểm ý tưởng token-level Wyner–Ziv: điểm số là `q^T k`, nên méo của một phép nén
key phải đo bằng `E_q[(q^T e)^2] = e^T M_q e` với `M_q = E[qq^T]` — tức nén theo
metric mà **query** tạo ra, không phải metric Euclid mà low-rank thường tối
thiểu hoá. Tương đương: whiten `k~ = M_q^{1/2} k` rồi nén Euclid.
`repro/shadowkv/probe_query_whitened_key_spectrum.py`, qwen3-4B, RULER 32K
`niah_multikey_3`, 5 tầng × 8 kv-head = 40 head, post-RoPE, query held-out.

**Whitening thắng ở mọi rank, và thắng đậm ở rank thấp** (recall@64 trên 32768
token, cùng metadata r số/token):

| rank | recall plain | recall whitened |
|---|---|---|
| 8 | 0.271 | **0.587** |
| 16 | 0.406 | **0.676** |
| 32 | 0.606 | **0.770** |
| 64 | 0.764 | **0.874** |

Whitened rank 32 ≈ plain rank 64 ⇒ **tiết kiệm 2× metadata**. Drift (khớp `W`
trên prefill query, chấm trên decode query) làm tụt đều ~0.03–0.06 recall và
**không** thu hẹp khoảng cách — giả định calibrate đứng được.

⚠ **"90% năng lượng ở rank 5.6" là ảo.** Một hướng duy nhất giữ **72%** năng
lượng sau whitening: đó là hướng query-trung bình, mà mọi token đều có thành
phần lớn dọc theo nó — một offset theo `i`, tốn rank mà **không xếp hạng gì**.
Trừ trung bình key đi thì 90% năng lượng cần whitened rank **12.9** (plain
47.5). Con số 12.9 mới là con số được phép viết ra.

⚠ **Và cả 12.9 cũng chưa phải câu trả lời, vì năng lượng là sai đơn vị.** Ở
rank 16 — tức đã giữ 90% năng lượng đã trừ trung bình — recall@64 mới **0.676**.
Lý do có cấu trúc: 10% phương sai dư đủ để xáo lại top-64 trong 32768 token.
**Retrieval là thống kê đuôi, rate–distortion là trung bình.** Nên định lý
water-filling (tối ưu MSE) tối ưu **sai mục tiêu**; muốn rigorous thì hàm mục
tiêu phải là bề rộng bound hai má per-token, không phải MSE.

**Về rate thì chưa thắng PQ.** Recall 0.77 cần ~32 chiều mang tin; kể cả 2–3
bit/chiều là 64–96 bit/token, so với ~12 bit của PQCache. Hướng này là một
**guarantee play** (khoảng tin cậy per-token tính được), không phải rate play.


## ⏳ 2026-09-14 (tối) — cân lại shard RULER, và HAI ô RetroInfer chưa từng nằm trong queue nào

126/195 ô có điểm, 0 hỏng. Bảng không đổi hình: ours **trên quest và shadow ở
cả ba độ dài**, hoà paris ở 64K (0.927/0.927 trên 9 task), thua paris ở 128K
(0.938/0.963) và **toàn bộ khoảng cách vẫn là `niah_multikey_3`** — ours 0.690
vs paris 0.860, trong khi quest 0.540 và shadow 0.510 còn tệ hơn ours.

**Lỗ hổng trong ma trận.** Đối chiếu ba queue với 13 task × 5 method × 3 độ dài
thì thiếu 2 ô RetroInfer — `65536 niah_single_1` và `131072 niah_single_2` —
**không nằm trong queue của máy nào**, và không có marker ở bất kỳ máy nào. Tức
là chúng sẽ **không bao giờ chạy** và không ai biết, vì mọi bảng đều đọc từ ô
có sẵn chứ không đối chiếu với ma trận mong đợi. Đã thêm lại vào m1. (4 ô
non-retro cũng vắng khỏi queue nhưng đã chạy xong từ trước — dòng bị xoá trong
lần dọn ô chuyển nhầm ngày 13/09.)

**Cân lại.** Cả 31 ô RetroInfer còn lại nằm trên shard m1 vì lúc sinh queue chỉ
m1 có kernel; m1 còn 44 ô trong khi m2 còn 5 và m4 còn 2 — hai máy kia sắp nằm
không suốt 15 giờ. Đã chuyển theo tỉ lệ số GPU:

* m4 ← **13 ô 32K** retro (7.4 GPU-giờ / 2 card)
* m2 ← **4 ô 128K + 2 ô 64K** retro (17 GPU-giờ / 4 card)
* m1 giữ phần còn lại + 2 ô vá lỗ hổng

m1 từ ~15 giờ xuống ~5–6 giờ; cả ba máy về đích cùng lúc. Kiểm sau khi chuyển:
**0 dòng trùng giữa các shard**, retro đủ **39/39**, tổng 191 dòng + 4 ô đã xong
= 195.

**Cách làm an toàn** (lần trước tôi chuyển nhầm 6 ô vì chạy `cell_key.py` ngoài
khối env campaign): tính key trong đúng khối env; **bỏ m1 trước, thêm máy kia
sau** (cửa sổ ở giữa chỉ làm ô tạm không ai claim, ngược lại thì hai máy cùng
claim); kiểm lại marker **dưới `flock` claim.lock** ngay trước khi ghi; và
`os.replace` trên **cùng filesystem** để pool không bao giờ đọc queue viết dở —
`/tmp` khác device nên lần đầu ném `Invalid cross-device link`, may là ném
trước khi ghi.


## ⏳ 2026-09-14 — harness reasoning cho m3, và hai cái bẫy tìm thấy khi dựng nó

MATH500 / AIME25 / GPQA-diamond được đưa lên cùng đường ray với RULER và
LongBench-v2: `gen_reasoning_queue.py` + `run_reasoning_campaign.sh` (hai pha
smoke → full) + `report_reasoning.py`. Trước đó chúng là hai script rời chạy
tay, mỗi seed một results-root, không có queue.

**Seed đi trong task key** (`aime25-s2`), vì env của pool cố định cho cả queue.
`run_cell.sh` bóc hậu tố thành `SHADOWKV_GENERATION_SEED`; task key nằm nguyên
trong tên ô nên bốn seed là bốn ô. Kiểm bằng chạy thật trên m1 GPU7: seed 3 lặp
lại cho prediction **trùng byte**, seed 5 khác. Cổng
`test_a_seeded_reasoning_task_names_a_distinct_cell`.

**Bẫy 1 — AIME25 có hai parquet khác nhau.** `kvpress_aime25_percontext` bỏ mất
chỉ dẫn `\boxed{}` mà `kvpress_aime25_local` có. env_m2/m3/m4 trỏ vào
`percontext`, env_m1 trỏ vào `_local` — nghĩa là nếu chạy AIME trên m3 hôm nay
thì số **không ghép được** với bảng AIME của m1, mà triệu chứng sẽ chỉ là "m3
thấp hơn". Đã thống nhất cả bốn env về `_local` +
`make_reasoning_data_bundle.sh` (616 KB, sha256
`40e56fe34d6f44f6f1062873590b9355ed4622906c9c32081ba85f1366619932`).

**Bẫy 2 — scorer MATH500 so khớp chuỗi nguyên văn.** Ô thử đầu tiên: model trả
đúng `\boxed{(3, \frac{\pi}{2})}`, đáp án là `\left( 3, \frac{\pi}{2}
\right)`, chấm **0**. Cố ý bám KVPress. Phạt đều mọi method nên vẫn so được,
nhưng nén dải điểm — cần quyết định trước khi đọc bảng, không phải sau.

Trần generation của ma trận mặc định (6 method × 3 task × seed 4/1/2) là **74M
token** nếu không có ô nào dừng sớm — hàng trăm GPU-giờ. Vì thế pha smoke là bắt
buộc, và `report_reasoning.py --cost` quy nó ra giờ cho ma trận đầy đủ.

**ShadowKV bị loại khỏi reasoning — vì thiết kế, không vì cấu hình.** Ô smoke
`gpqa` chết với `dynamic SVD transition is not implemented`. Chỉ mục landmark
của ShadowKV là SVD của key **trong prompt**, lấy một lần lúc prefill rồi đóng
băng; prompt reasoning ngắn hơn `sparse_budget` nên cache ở nguyên
`dense_warmup` (exact) và không có đường chuyển sang sparse khi generation vượt
budget. Nghĩa là **ShadowKV cần prefill dài mới tồn tại** — trên bench
prompt-ngắn ô đúng là N/A kèm lý do. Ma trận còn **48 ô** (4 method × 3 task ×
2 budget). Đây là điều đáng viết vào paper, không phải một ô trống.

**Đêm 14/09 chạy tự động**: reasoning 48 ô đang chạy trên m3, và
`run_after_pool.sh` chờ **cả** pool reasoning **và** pool LongBench-v2 rồi tự
nối 30 ô quét budget LongBench-v2 (4096/2048/1024/512, `full` chỉ một lần).

**Bốn lỗi vận hành trong một buổi, cùng một hình dạng: hỏng im lặng.**

1. `SHADOWKV_POOL_GPUS=""` để sinh queue mà không chạy — idiom này tôi bịa ra và
   nó **ghi một `.state/gpus.txt` rỗng**, lần chạy thật sau kế thừa rồi khởi
   động 0 worker, pool thoát ngay không in gì. Nhìn như campaign xong tức khắc.
   Sửa: `run_reasoning_campaign.sh full queue` là dry-run thật (thoát trước khi
   `.state` tồn tại), và run_pool **từ chối** roster rỗng lúc khởi động — làm
   rỗng `gpus.txt` *trong lúc* chạy vẫn là cách rút cạn pool có chủ đích.
2. Worker đậu trên card không bao giờ trống thì **ngủ mãi**: chỗ duy nhất nó
   biết queue đã cạn nằm *sau* `gpu_free`. Pool sống nhăn mà chẳng làm gì, và
   trong `pgrep` trông hệt như pool thứ hai đang tranh card. Sửa: `claim_next`
   có chế độ `PROBE=1` — cùng một vòng quét, cùng luật bỏ qua, không ghi marker
   — và worker dùng nó để thoát khi không còn gì để claim.
3. `pgrep -af run_pool.sh` liệt kê **cả worker** (fork cùng dòng lệnh), nên một
   pool 5 card hiện 6 PID. Đừng đếm PID để kết luận có hai pool; đọc đường dẫn
   `.state` trong dòng lệnh.
4. Nối chuỗi mà chỉ `--wait` một pool là bẫy: lệnh sau khởi động pool trong
   state dir mà pool khác còn giữ `flock`, nên nó bị từ chối và chuỗi chết lúc
   3 giờ sáng. `run_after_pool.sh` cho lặp `--wait`, và điều kiện chờ là
   **marker** chứ không phải tiến trình — pool sống lâu hơn công việc của nó là
   chuyện đã xảy ra (lỗi 2).

**Bẫy 3 — biến sót trong shell, hai lần cắn.** Mọi đường dẫn là
`${VAR:-mặc_định}`, nên giá trị đã export thắng mặc định. Trên m3 một
`SHADOWKV_RESULTS_ROOT` sót lại từ lần smoke làm `report_lbv2.py` đọc thư mục
rỗng và in ra bảng trắng trơn — trông y hệt "mất sạch kết quả", trong khi
`pgrep -fc eval_acc` = 6, campaign vẫn đang chạy bình thường. Lần thứ hai, một
`SHADOWKV_MATH500_PATH` cũ sống sót qua `git pull` đã chuyển dữ liệu vào repo,
và campaign từ chối khởi động với một đường dẫn không ai gõ.

Ba lớp chữa: mỗi `env_m*.sh` **in ra** code/results/data nó vừa giải; nó **báo
động** khi giá trị kế thừa khác mặc định của repo (`_shadowkv_pin`, dùng `eval`
chứ không `${!name}` vì m3 chạy zsh); và cả ba reader **từ chối** một root không
có `cells/`. Cách chắc nhất vẫn là đừng mang biến đi: `REASONING_ROOT` đã có mặc
định đúng, nên đừng đặt `$R` rồi truyền vào — chính `$R` sót lại là thứ đã ghi
một queue reasoning vào trong root của LongBench-v2.

Kèm theo: `report_ruler5.py` — bảng RULER năm method ghép từ ba máy qua digest
vài kB (không chuyển artifact qua tunnel m2). Aggregate là **pairwise**: macro
trên các task mà **cả hai** method đã xong, vì macro equal-depth trên cả năm
còn rỗng cho tới khi method chậm nhất về.

---

## ⚠ 2026-09-14 — HAI lỗi làm hỏng số đã báo: bug rò rỉ trạng thái RetroInfer, và reader đọc sai

### 1. RetroInfer rò rỉ chỉ mục giữa các prompt — **mọi ô RetroInfer đã cách ly**

Khi đệm trục cụm lên `max_clusters` để hợp kernel GEMM của tác giả, tôi phá vỡ
giả định ghi ngay trong `_reset_metadata`: *"chỉ đọc cụm dưới `cluster_count`"*.
Từ lúc đó việc chấm điểm chạy **toàn bộ chiều rộng**, và `empty_cluster` là thứ
duy nhất giữ phần đuôi chưa dựng ra khỏi bảng xếp hạng — mà mặt nạ đó **không
được xoá giữa các prompt**.

Prompt ngắn sau prompt dài thừa hưởng cờ "sống" cũ, chọn slot đang chứa token
của mẫu trước. **LongBench-v2 short: RetroInfer 0.039** trong khi `full` 0.356,
ours 0.356, paris 0.344. Prediction thoái hoá dần — mẫu 0–2 mạch lạc, mẫu 20 trở
đi thành `'Thecdn200000000000000'`. Mẫu đầu sạch vì chưa có gì cũ để kế thừa.

Sửa ở `1446d17`: `_reset_metadata` xoá `empty_cluster` và `cluster_pages`. Cổng
`test_a_shorter_prompt_does_not_inherit_the_previous_index` — chạy prompt dài →
clear → prompt ngắn — **đã xác minh fail trên code cũ**. 8/8 pass.

**RULER cũng dính**: prompt RULER gần cùng độ dài nên vùng cũ nhỏ, nhưng nhỏ
không phải không. Đã cách ly 20 ô m1 (13×32K, 6×64K, 1×128K) + 54 file, giết 4 ô
đang chạy, chạy lại ~18 GPU-giờ. m2/m4 chưa từng chạy ô RetroInfer nên không mất.

⇒ **Mọi con số RetroInfer trong nhật ký trước 14/09 đều không dùng được**, kể cả
bảng 32K "retro 0.971 dẫn đầu".

### 2. Reader của tôi lấy trung bình các dòng thay vì dòng cuối

`avg_score` trong jsonl là **trung bình luỹ tiến**, nên điểm ô là **dòng cuối**.
`status.py` ghi đúng luật này ở docstring; tôi vẫn viết reader ad-hoc và lấy
trung bình các dòng. Lệch thật:

| ô | tôi báo | đúng |
|---|---|---|
| cwe ours 32K | 0.788 | 0.789 |
| cwe paris 32K | 0.825 | **0.849** |
| qa_1 ours 32K | 0.730 | **0.840** |
| qa_1 paris 32K | 0.721 | **0.830** |

`qa_1` lệch 0.11 và **đảo thứ hạng**. Skill đã có luật "một module đọc chuẩn,
mọi script tổng hợp phải import nó" — tôi vi phạm và dính đúng lỗi luật đó tồn
tại để chặn.

---

## ⭐ 2026-09-14 — Trạng thái campaign, m3, và LongBench-v2

**RULER** (5 method × 13 task × 3 độ dài, qwen3, 100 mẫu, budget L/32, offload):
118 ô có điểm, 0 hỏng. Chưa độ dài nào có task đủ cả 5 method ⇒ **chưa macro nào
hợp lệ**, chỉ so được đối đầu trên tập task chung:

| | 32K | 64K | 128K |
|---|---|---|---|
| ours vs paris | 0.939 / 0.942 | 0.918 / 0.918 | **0.943 / 0.973** |
| ours vs quest | 0.939 / 0.930 | 0.998 / 0.995 | 0.902 / 0.863 |
| ours vs shadow | 0.965 / 0.951 | 0.918 / 0.896 | 0.937 / 0.914 |

Ours **trên quest và shadow ở cả ba độ dài**; với paris thì thua sát ở 32K, hoà
64K, **thua rõ ở 128K**. Toàn bộ khoảng cách đó nằm ở **`niah_multikey_3`**:
ours 0.690 so với paris 0.860 (32K/64K cả hai ~1.000 — task chỉ vỡ ra ở 128K).
`cwe` 128K mới có paris **0.266** (32K 0.849 → 64K 0.656).

**m3 (`ruifeng`, 8×L40 46GB, 503 GB RAM) đã dựng xong** — xem `PROJECT.md` §3b.
Env khớp khít m1/m2/m4 (torch 2.6.0+cu124, tf 4.55.4, flash_attn 2.7.4.post1).
Đang chạy **LongBench-v2 128K**, 12 ô, `full`/ours/retro/paris ở lượt claim đầu.
Ba ô short đầu tiên: **full 0.356, ours 0.356, paris 0.344**.

---

## ⚠ 2026-09-14 — SỬA LẠI: offload nhanh hơn là **sai**, lý do thật là bộ nhớ

Đo lại 4 lần xen kẽ thứ tự (32K b1024, ours, cùng xuất phát 53–54 °C):

| | prefill | decode median |
|---|---|---|
| offload | 15.5 / 15.8 s | 36.5 / 37.7 ms |
| GPU-resident | **10.0 / 10.0 s** | 41.6 / 43.2 ms |

**Resident thắng prefill 35%**, thua decode 13%. Với RULER (gen ~128 bước)
prefill mới quyết định ⇒ resident nhanh hơn ~24% mỗi mẫu. Con số 27.48 s tôi
báo lần trước là rác — một lần đo, không lặp lại, và chênh decode 16% nằm **dưới
ngưỡng 20%** mà chính file kỷ luật của tôi bảo phải lặp mới tin.

⇒ `PROJECT.md` §2e giữ **luôn offload**, nhưng lý do là **bộ nhớ** (64K resident
OOM trên card 24 GB — đo chắc chắn), **không phải tốc độ**.

Cảnh báo về phép đo lại: nó chạy song song 4 ô campaign nên có nhiễu PCIe, mà
nhiễu đó phạt bên offload nặng hơn.

---

## ⏳ 2026-09-13 — Campaign RULER 5 method × 32K/64K/128K, ba máy

**195 ô** = 13 task × 3 độ dài × 5 method, model `qwen3`, **100 mẫu/ô** (RULER
có đúng 100; default 96 của pool bỏ mất 4). Budget khớp **L/32**: 1024 / 2048 /
4096. Không chạy `full`.

| máy | GPU | ô | trọng số |
|---|---|---|---|
| m1 | 7 | 132 | 247 / 245 |
| m2 | 4 | 46 | 140 / 140 |
| m4 | 2 | 17 | 68 / 70 |

Chia theo **trọng số độ dài** (128K ≈ 4× 32K) chứ không theo số ô. Mỗi máy tự
sinh shard của mình bằng `gen_ruler_five_queue.py` với cùng tham số — không
chép file nào giữa các máy.

**RetroInfer ghim ở m1**: m2/m4 chưa build kernel tác giả (chúng không có nvcc
12.4 trong env; m2 có CUDA 12.8 hệ thống, m4 chỉ có 13.0). Launcher **từ chối
khởi động** nếu một máy nhận ô RetroInfer mà không import được kernel, thay vì
để nó fail 39 lần liên tiếp.

**Đã qua cổng trước khi thả:**

* 32K, 64K, 128K: cả năm method chạy được (2 mẫu/ô).
* 128K ParisKV lúc đầu **OOM** — 3.13 GiB reserved-nhưng-không-dùng, tức phân
  mảnh chứ không phải thiếu chỗ. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  chữa được, và đã xác minh **không đổi kết quả** (Quest cwe 32K b1024 8 mẫu,
  md5 trùng từng byte). Giờ là default của `run_cell.sh`.
* Smoke test 32K: cả năm chạy chỉ với hằng số protocol.

⚠ **ShadowKV vẫn attend nhiều hơn ở cùng cờ budget.** `budget_audit.py` ở 32K
b1024: ShadowKV 1152 token (1024 + outlier 96 + local 32) so với Quest 1056.
Campaign chạy **cùng cờ** theo quy ước preregistered, nên bảng cuối **phải ghi
chú** chênh lệch này — hoặc ghép lại bằng audit.

⚠ **ParisKV và RetroInfer không tất định** — chênh ≤1 điểm giữa hai lần chạy là
nhiễu.

---

## ⭐ 2026-09-13 — Chốt bản mặc định cho benchmark, và **sửa lại chính sách residency**

Bảo yêu cầu: năm method (ours, RetroInfer, ParisKV, Quest, ShadowKV) phải default
về bản nhanh nhất của chúng trước khi chạy RULER 32K/64K/128K.

**Đổi hai default trong `run_cell.sh`:**

1. `STREAMING_ROUTER_BACKEND` **torch → triton**. Đo: hai backend đều **tất
   định**, nhưng **khác nhau** — cùng md5 khi lặp lại, khác md5 giữa hai bản.
   Chênh lệch trong nhiễu (cwe 0.785 / 0.780 trên 20 mẫu; `niah_multikey_3`
   1.000 / 1.000) còn triton **nhanh 1.6×**. Backend giờ nằm trong **tên ô**
   (`_rbtriton`), và cả `cell_key.py` lẫn stamp metadata đã đồng bộ.
2. `STREAMING_OFFLOAD` mặc định **1 cho mọi độ dài**.

**⚠ SỬA LẠI §2e.** Bản cũ ghi "≤64K để KV trên GPU, 128K mới offload". **Sai cả
hai vế và chưa từng được kiểm:**

* ở 64K GPU-resident, ours / Quest / ParisKV **OOM** trên card 24 GB;
* ở 32K, nơi cả năm đều vừa, offload vẫn **nhanh hơn**: ours prefill
  **15.05 s** so với 27.48, decode median **43.11** so với 49.91 ms, đỉnh bộ nhớ
  **10.91** so với 15.47 GiB — mà lần GPU-resident còn được card nguội hơn.

Lý do: gather UVA hợp nhất và **tái dùng chéo bước chỉ chạy khi offload**
(`_can_reuse_selected_blocks` đòi `self.offload`). Để KV trên GPU là mất reuse.

**⚠ RetroInfer không tất định** — hai lần chạy cùng cấu hình cho md5 khác nhau
(0.825 và 0.850), do `tl.atomic_add` trong k-means tác giả. Giống ParisKV.

**RetroInfer không còn tốn bộ nhớ thêm.** Vùng chỉ mục được **hoán vị tại chỗ**
thay vì giữ bản sao thứ hai: vị trí thời gian của một token chết ngay khi nó vào
chỉ mục, vì vùng exact chỉ là sink và đuôi. Kho backing giờ bằng mọi method khác.

**Smoke test**: cả năm method chạy ở 32K chỉ với hằng số protocol, không đặt
thêm knob nào.

---

## ⭐ 2026-09-13 — RetroInfer lên common path bằng **kernel của tác giả**

Method thứ tám, `retroinfer_author_common`. Ban đầu tôi viết lại phần method
bằng PyTorch; Bảo chặn lại: *"cứ theo implementation gốc là được"*. Đúng — bản
PyTorch chạy **108.87 ms/bước**, bản dùng kernel họ chạy **54.43**.

**Đã build và dùng code của họ:** `cache_hub/kmeans.py` (Triton),
`batch_gemm_softmax` (CUTLASS, chấm điểm cụm), `gather_copy_vectors` (gom vùng
ước lượng), `weighted_flash_decoding` (attention hai vùng + gộp một softmax).
Cutlass v3.5.1, `CUDA_HOME` trỏ conda env (nvcc 12.4; nvcc hệ thống 13.0 bị pip
từ chối). Fork flash-attention cài dưới tên riêng, **`flash_attn` 2.7.4 không
bị đụng** — đã kiểm.

**Vận chuyển cũng là của họ**: kho offload được ghi theo thứ tự cụm như
`construct_func` làm, nên bản ghi PCIe là **page 2 KB** chứ không phải token
rải 256 B. Gather của nó đo được **nhanh hơn Quest** (95 so với 112 ms/8 bước).

**Hai cái bẫy của kernel họ**, cả hai đều **không báo lỗi** mà hỏng âm thầm:
`copy_kernel.cuh` **ghim head_dim = 128**; trục cụm phải canh biên bội số
`lcm(8, n_segment)`, không thì fault `misaligned address` giữa chừng decode.
Adapter chặn cái đầu bằng `ValueError`, cái sau bằng làm tròn sức chứa index.

**Estimation zone có tác dụng** (16K, B=64, 20 mẫu, ô phân biệt được):
cwe **0.825** so với **0.705** khi tắt; fwe 0.967 so với 0.950.

**Tốc độ decode** (64K, B=4096, offload, đúng env protocol): Quest 43.48,
ours 46.22, **RetroInfer 54.43**, ShadowKV-CPU 67.21 (bản ghi 11:21).

**Cổng kiểm** `repro/shadowkv/test_retroinfer_author_common.py`, 6/6 pass;
suite ShadowKV 169 pass / 4 fail hỏng sẵn.

---

## ⚠ 2026-09-13 — "ours chậm đi 1.6×" là tôi đo sai env, không phải hồi quy

Tôi báo ours 72 ms ở 64K B=4096 trong khi bản ghi 11:21 là 48.00. Bảo hỏi lại.
**Không có hồi quy nào.** Chạy lại đúng env của `rtf.sh` thì ours ra
**46.22 / median 43.00** (bản ghi: 48.00 / 44.81) và Quest **43.48** (46.62) —
cả hai còn nhanh hơn chút.

**Cái tôi bỏ sót:** `run_cell.sh` mặc định `STREAMING_ROUTER_BACKEND=torch`,
trong khi mọi số runtime đã ghi đều đo với **`triton`**. Kèm theo đó
`UPSTREAM_MATCHED_EXACT_REGIONS=1`, `STREAMING_GATHER_BACKEND=uva`,
`SHADOWKV_RMSNORM_BACKEND=flashinfer` cũng không được đặt. Riêng router backend
làm median 43.00 → 70.51.

**Vì sao lọt lưới:** Quest và `full` vẫn khớp bản ghi cũ nên bảng *trông* hợp
lệ; chỉ ours lệch, và tôi suýt đi bisect commit để tìm một hồi quy không tồn
tại. Quest không nhạy với router backend vì nó **không có** router học được.

**Phải làm:** env đo runtime là **một khối**, chép nguyên từ `rtf.sh`, không
dựa vào default của `run_cell.sh`. Xem `PROJECT.md` §4.

---

## ⭐ 2026-09-13 — Khung token dùng chung cho cả bảy method

Bốn vùng, giống nhau cho mọi method: `sink | retrievable | local | buffer`.
Chi tiết công thức ở `PROJECT.md` §2c; bảng sự thật từng method ở §2b.

**Vì sao.** Trước đó cadence index token decode là 8 / 16 / 1 / 1024 / một-lần,
và vùng exact là 32 / 64 / `ratio×B` / 1088 — năm kiểu trong một bảng. RetroInfer
ở `update_segment=1024` attend tới **1088 token exact ngoài budget**, tức gần gấp
đôi ngân sách các method khác mà không bị tính.

**Đã kiểm.**

* bit-exact 4 ô (`niah_multikey_3`, `cwe`, `qa_1`, `vt` @32K b1024) — **4/4 trùng
  từng byte** ở cấu hình thoái hoá. Baseline chốt TRƯỚC khi sửa.
* suite **165 pass / 4 fail**, đúng 4 cái hỏng sẵn.
* `tests/test_streaming_frame.py` 6/6, gồm ca thoái hoá quét 585 độ dài × 3 block
  size, và một test **phủ sóng**: không token nào nằm ngoài cả index lẫn đuôi exact.
* Không đặt env → mọi method giữ default tác giả. Commit này không thể tự làm
  lệch số nào.

**Giá trị mặc định đã chốt** (từ `s2-ttt/repro/reasoning/run_reasoning_pool.sh`):
reasoning sinh ở `temp 0.6 / top_p 0.9 / KHÔNG top_k`.

**Đã tái dùng, không chạy lại**: full attention Qwen3 RULER 32K (**94.47**) và
64K (**90.05**), n=100, ở
`sparse_attention_results/ruler_qwen_full_reused_20260913/` kèm `PROVENANCE.md`.
Bằng chứng tái dùng được: stamp commit `4c9d0686` cùng env, và diff cho thấy
nhánh `full` trong `base.py` không đổi, thay đổi truncation nằm trong nhánh
longbench-v2, `evaluator.py` vẫn greedy mặc định, `gen_len−1` chỉ áp cho aime25.
**128K chưa có** — phải chạy mới.

---

## ⚠ 2026-09-13 — SỬA LẠI: ba kết luận sai về hành vi method

Trong một phiên tôi kết luận sai ba lần vì **grep nông thay vì đọc hàm**:

1. "PQCache không index token decode" → **sai**, `_assign_codes` chạy mỗi bước.
2. "RetroInfer không mở rộng cụm" → **sai**, `_append_index` mỗi 1024 token.
3. "ShadowKV chỉ index ở prefill" → **sai với bản CPU**, nó có wave-sealing.

Và một lần nữa về offload: "chỉ 3/7 method có offload" → đúng hơn là **6/7 để KV
ngoài GPU**, chỉ khác cơ chế. Cái tôi đếm là "có cờ `--streaming_offload`".
RetroInfer là cái duy nhất giữ KV trên GPU, và sẽ OOM ở 128K.

Luật rút ra: kết luận về hành vi method phải đọc **thân hàm**, không phải đếm
grep. Bảng §2b của `PROJECT.md` ghi kết quả đã đọc kỹ, kèm tên hàm — dùng nó thay
vì dò lại.

---

## ⚠ 2026-09-13 — Bẫy: sửa file khi job đang chạy (ba lần trong một ngày)

* `run_cell.sh` sửa khi pool chạy → 7 ô chết syntax error **sau khi** đã chạy xong.
* Đổi tên marker khi worker đang giữ tên cũ → 7 marker mồ côi.
* Sửa `eval_acc.py`/`base.py`/`run_cell.sh` khi vòng kiểm bit-exact đang chạy →
  ô `qa_1` chết không để lại log.

Bash đọc script theo **byte offset**; Python đọc file lúc import, nên tiến trình
**khởi động trong cửa sổ sửa** nhận file dở. Luật: không sửa gì khi có job đang
chạy dùng file đó; nếu buộc phải thì `mv` (đổi inode), không ghi đè tại chỗ.

Và: **chạy full suite TRƯỚC khi commit**, không phải sau. Tôi đã push hai lỗi
collection rồi mới phát hiện.

---

## ⭐ 2026-09-13 — Reset sạch: repo mới, kết quả cũ archive, ba máy đồng bộ

**Vì sao reset.** Truy nguồn bảng RULER mất gần một buổi và kết quả là bảng đó
không tái tạo được từ một root nào. Ba nguyên nhân chồng lên nhau:

1. **Code drift.** m4 chạy chậm hơn m1 **81 commit**, trong đó có
   `5d2b1929 Match official LongBench-v2 middle truncation` (đổi cách cắt
   context) và `0aae6ba0 Add stratified LongBench-v2 campaign filters` (đổi tập
   mẫu short/medium). md5 của `adaptive_centroid_streaming_cache.py`, `base.py`,
   `eval_acc.py` đều khác giữa m1 và m4. Mọi số LongBench từ m4 **không so được**
   với m1/m2 — kể cả khi nằm chung một con số: ours B=1024 medium từng được gộp
   từ 10 shard trên 3 máy, tức **một số trộn hai code base**.
2. **Ô trùng.** Campaign `ruler_qwen_fairmeta32x_20260910` có **13 ô trùng, tất
   cả ShadowKV, mỗi ô có một bản sinh đôi 0.00** (harness append `/1`, `/2`; lần
   chạy hỏng nằm cạnh lần tốt). Reader không dedupe theo mtime sẽ báo ShadowKV sai.
3. **297 biến thể "ours"** trong kết quả. Bảng RULER đang lưu hành dùng
   `adaptive_lse_prefix4_b512_...` (04/09) còn campaign LongBench/reasoning chạy
   `adaptive_lse_stream_..._qmean_..._ard1.5_costmean_gap` (12/09) — hai cơ chế
   khác nhau, không phải tinh chỉnh.

**Đã làm.**

- Dừng hết: m1 8/8, m2 4/4, m4 2/2 GPU trống.
- Archive (mv, không xoá): m1 **854** thư mục, m2 **167**, m4 **242** →
  `_archive_20260913/` trên chính mount lớn của từng máy. Không đụng
  `models/ datasets/ huggingface/ envs/`. HANDOFF.md cũ còn nguyên ở
  `/storage/baonn/kvpress_results/HANDOFF.md`.
- Repo mới `github.com/nguyenngocbaocmt02/sparse_attention` @ `8427d30`:
  **20.000 file / 470 MB → 213 file / 2.7 MB**. Bỏ `lm-evaluation-harness`
  (15.737 file) và `opencompass` (3.383) vì **không file nào** dưới `ShadowKV/`
  hay `repro/shadowkv/` import chúng; bỏ `kvpress/ s2_ttt/ trimkv/ lact/
  kv_storing/ analysis*/ FastKVzip/ paper_assets/` cùng lý do.
- Kiểm trước khi commit: 33/33 model import sạch; pytest **159 pass / 4 fail**,
  và 4 fail đó **đã hỏng sẵn trong cây gốc** (chạy đối chứng để chắc).
- Ba máy cùng `8427d30`, md5 bốn file lõi khớp tuyệt đối, cùng
  `torch 2.6.0+cu124 / transformers 4.55.4`.
- Results root sạch: `/storage/baonn/results`, `/home/nbnguyen/results`,
  `/storage/nbao/results`.

**Còn treo.**

- **m3 chưa migrate** — đang có hai checkout, hai env. Chi tiết ở `PROJECT.md` §3.
- **m4 chưa pull được từ GitHub**: deploy key của nó chỉ cấp cho repo `s2-ttt`.
  Đã seed bằng git bundle qua LAN. Cần thêm key này vào repo, hoặc để repo public:
  `ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIA/ssHPbr0XYhdCJl/WjiP5chCrA6d5v37s43ew3rg/l m4-sashimi-s2-ttt`

---

## ⚠ 2026-09-13 — Bẫy: archive làm gãy symlink của kernel CUDA

`kernels/shadowkv.cpython-310-x86_64-linux-gnu.so` **không nằm trong git**
(`.gitignore: *.so`) và phải build trên từng máy. Trên m4 nó là một **symlink**
trỏ sang `shadowkv-research/` — archive `shadowkv-research` xong thì symlink gãy,
và triệu chứng gây nhầm: `find` vẫn liệt kê file, `cp` báo *No such file*
(find liệt kê symlink, stat theo nó thì hỏng).

Build lại từ nguồn trên m4 **thất bại** (`Error compiling objects`). Bản đang
dùng được lấy từ `_archive_20260913/shadowkv-research/ShadowKV/kernels/`
(2.706.632 byte) — đúng bản m4 vẫn chạy.

Hệ quả cho lần sau: sau khi clone repo sạch, mỗi máy cần
`ShadowKV/kernels/*.so`. Lấy từ máy đó (build hoặc bản cũ), không copy giữa máy
trừ khi cùng CUDA/torch/ABI. `import torch` phải đứng trước `from kernels import
shadowkv`, nếu không sẽ báo `libc10.so: cannot open shared object file`.

RULER text cũng đã chuyển ra ngoài checkout: `ruler_shadowkv/` trên mount lớn
của từng máy, symlink vào `ShadowKV/data/ruler`. m1 52M, m2 2.5G, m4 52M.

---

## ⚠ 2026-09-13 — SỬA LẠI: hai bảng đã báo cáo không có nguồn

**MagicPIG và PQCache trên LongBench.** Bảng từng báo
`MagicPIG 37.78% (180/180)` và `PQCache 33.89% (180/180)` ở B=1024. Quét cả ba
máy: hai method này trên LongBench **chỉ tồn tại dưới dạng smoke 1 mẫu ở
64K/B=4096**. Không có ô 180 mẫu nào. Hai hàng đó không dùng được.

**Xếp hạng RULER 64K.** Bảng equal-depth ở 64K chỉ có **5/13 task** đủ cả bốn
method, và `cwe` — nơi ours hơn Quest **+25 điểm** (64.70 vs 39.70) — rơi khỏi
tập chung vì Paris/ShadowKV chưa chạy nó. Thứ hạng 64K vì thế là **artifact của
lịch chạy**, không phải của method.

---

## ⚠ 2026-09-13 — SỬA LẠI (lần 2): GPQA/MATH500 lệch, KHÔNG phải AIME

Bản ghi đầu tiên của tôi nói AIME25 (`top_p 0.9 / top_k -1`) lệch khỏi
GPQA/MATH500 (`0.95 / 20`) và đề xuất chuẩn hoá về 0.95/20. **Ngược lại.**

Project tiền nhiệm s2-ttt chạy reasoning ở `TEMPERATURE=0.6 TOP_P=0.9`, **không
có tham số top_k** (`repro/reasoning/run_reasoning_pool.sh` dòng 35–36; ba
launcher GPQA/AIME đều ghi rõ "temperature 0.6, top_p 0.9"). Vậy AIME25 đang
giữ đúng protocol; `run_gpqa_math_qwen_b1024.sh` mới là cái đưa `0.95 / 20` vào.

Hệ quả: **số GPQA/MATH500 của campaign 12/09 phải bỏ**, không phải số AIME.
Chuẩn từ nay: temp 0.6 / top_p 0.9 / không top_k cho mọi reasoning bench.

---

## ⭐ 2026-09-13 — Đo thật chi phí offload: UVA của ta vs kernel của ShadowKV

Script: `repro/shadowkv/bench_offload_gather.py`. Một layer Qwen3-4B, 32K ctx,
budget 1024, B=1, 8 KV head, head_dim 128, bf16, A5000 PCIe 4.0 x16, GPU3.
CUDA events, 50 vòng sau warmup.

| đường | ms/layer | qua PCIe | băng thông |
|---|---|---|---|
| ta: `gather_kv_uva` (per-token, K+V) | 1.320 | 4.00 MB | **2.96 GB/s** |
| ta: torch fallback (per-token) | 1.615 | 4.00 MB | 2.42 GB/s |
| ta: `gather_blocks_reuse_uva`, 0% reuse | 0.345 | 4.00 MB | **11.31 GB/s** |
| ta: `gather_blocks_reuse_uva`, 90% reuse | 0.059 | 0.40 MB | — |
| ShadowKV `gather_copy_with_offsets`, 0% hit | 0.358 | 2.00 MB | **5.45 GB/s** |
| ShadowKV `gather_copy_with_offsets`, 90% hit | 0.051 | 0.20 MB | — |
| ShadowKV `gather_copy_d2d` (nhánh K), 90% hit | 0.040 | 0 (D2D) | — |
| repo hôm nay: ShadowKV-CPU `torch.gather` trên CPU | 0.630 | 2.00 MB | 3.10 GB/s |
| `copy_` pinned liên tục 4 MB (DMA engine) | 0.535 | 4.00 MB | 7.30 GB/s |

Ba kết luận, theo **băng thông hữu ích** (chuẩn hoá theo số byte, vì ShadowKV
chỉ chuyển V còn ta chuyển cả K và V):

1. **Kernel UVA theo block của ta nhanh hơn kernel của ShadowKV gấp ~2×**
   (11.31 vs 5.45 GB/s) và **nhanh hơn cả DMA engine** (7.30 GB/s) — đọc
   zero-copy từ nhiều SM song song vượt được copy engine.
2. **Bề rộng record mới là thứ quyết định, không phải kernel.** Quét block
   4/8/16/32/64 → 10.04 / 10.92 / 11.11 / 11.20 / 10.76 GB/s: phẳng. Nhưng
   per-token (record 256 B) tụt xuống 2.96 GB/s — **chậm 3.8×**. Đây là chi phí
   thật của method truy hồi theo token, không phải lỗi cài đặt.
3. **Đường ShadowKV-CPU trong repo hôm nay đang chậm.** `torch.gather` trên CPU
   mất 0.630 ms so với 0.358 ms (cold) và 0.051 ms (90% hit) của kernel tác
   giả — **chậm 12× ở mức hit thực tế**. Ta đã đổi sang nó vì kernel tác giả
   chuyển cứng 2 KiB/record nên hỏng ở chunk 4; cờ `_native_cached_gather` tính
   ra rồi **không ai dùng** (code chết, `models/kv_cache.py:548`).

Quy ra decode: 36 layer × 1.32 ms = **47 ms/token** cho nhánh per-token
(PQCache), so với 36 × 0.06 = **2 ms/token** cho nhánh block có reuse.

---

## ⚠ 2026-09-13 — SỬA LẠI: con số "90% hit" KHÔNG áp dụng cho ShadowKV-CPU của ta

Bảng offload phía trên ghi `gather_copy_with_offsets` ở 90% hit mất 0.051 ms.
Đúng với kernel, **sai với đường chạy thật của ta**: `get_retrieval_position_ids`
của `ShadowKVCache_CPU` gọi `self.cnts.zero_()` ngay trước khi trả về
(`models/kv_cache.py`), với ghi chú *"Streaming ids are not compatible with the
authors' fixed prompt cache-reuse offsets"*. `cnts=0` nghĩa là **mọi slot đều
tính là miss**. Vì ta mở rộng ShadowKV để seal landmark trong decode, id chunk
không còn nằm trong sổ sách reuse của tác giả nữa.

Hệ quả: dù có khôi phục kernel tác giả, ShadowKV-CPU cũng chỉ đạt **0.358 ms
(0% hit)**, không phải 0.051. Còn `gather_blocks_reuse_uva` của ta so khớp block
id **trên GPU ngay trong kernel**, không phụ thuộc sổ sách prefill, nên nó vẫn
reuse được với streaming id. Đó là lý do thật để chuyển nhánh V của ShadowKV-CPU
sang kernel của ta — không phải vì nhanh hơn 2×, mà vì **đó là cách duy nhất
ShadowKV-CPU có reuse**.

Ghi thêm: cả `ShadowKVCache` (GPU) lẫn `ShadowKVCache_CPU` đều seal landmark
trong decode. Bảng §2 của `PROJECT.md` từng ghi ShadowKV "không index khi
decode" — sai cho cả hai bản, đã sửa.

---

## ⭐ 2026-09-13 — Một chính sách residency, và HAI lỗi khung 256 lộ ra khi kiểm

Quyết định (theo yêu cầu): **dùng cả hai bản ShadowKV**. Context ngắn thì chạy
trên GPU, dài thì offload — và điều đó áp cho *mọi* method, không riêng ShadowKV.
Mọi GPU dùng được của cả ba máy đều 24 GB nên một ngưỡng cho cả đội:

| context | chế độ | đặt |
|---|---|---|
| ≤ 64K | KV trên GPU | `STREAMING_OFFLOAD=0`, ShadowKV `shadowkv` |
| 128K | KV pinned host | `STREAMING_OFFLOAD=1`, ShadowKV `shadowkv_cpu` |

**PQCache giờ nghe theo cùng một cờ.** Trước đây nó hard-code `device="cpu",
pin_memory=True`. Đó là lựa chọn triển khai chứ không phải thuật toán — mã PQ và
router vẫn trên GPU ở cả hai chế độ. Thêm `offload=` vào
`PQCacheAuthorCommonCache`, nối qua `base.py` / `eval_acc.py` / `run_cell.sh`.
Cũng thêm `PQCACHE_AUTHOR_ROOT` và `MAGICPIG_AUTHOR_ROOT` vào `env_m{1,2,4}.sh`
— **trước đó thiếu hẳn**, nên mọi ô PQCache/MagicPIG sẽ chết ngay khi khởi động.

### Kiểm bằng ô thật đã lôi ra hai lỗi của khung 256

Sáu ô `qwen3 / 8192 / niah_single_1 / b1024 / n=2`, frame `x32 l256 i256`:

| | trước khi sửa | sau khi sửa |
|---|---|---|
| ours, quest — `OFFLOAD=0` | **crash** `gather buffer capacity was underestimated` | score 1 |
| ours, quest — `OFFLOAD=1` | **score 0, không báo lỗi** | score 1 |
| shadowkv, shadowkv_cpu, pqcache | score 1 | score 1 |

**Lỗi 1 — `gather_capacity` thiếu chỗ cho buffer.** Nó tính
`budget + prefix + recent + block_size`, đúng **chỉ khi** interval *bằng*
block_size. Với interval 256 vùng exact dài thêm một token mỗi bước sinh cho tới
lần flush. Mô phỏng `StreamingBlockState` ở 8K/b1024: cần **1567**, cấp **1320**
— vỡ ở token sinh thứ 9. Sửa: cộng thêm `update_interval`.

**Lỗi 2 — fallback đọc block id như token id.** `select_key_value_cache` truyền
**block id** cho `get_key_value_cache` khi kernel reuse đủ điều kiện (kernel ăn
block id trực tiếp). Khi kernel ném lỗi và backend là `auto`, `except` nuốt lỗi
rồi rơi xuống nhánh portable — nhánh này hiểu tham số là **token id**. Kết quả:
gather token `b` cho mỗi block `b`. Không crash, không cảnh báo, chỉ ra **đáp án
sai**. Đây chính là hai ô score 0 ở trên: lỗi 1 làm kernel ném, lỗi 2 biến cú ném
đó thành số liệu trông bình thường.

Bài học: **một lỗi capacity đã tự bộc lộ ở nhánh này lại bị nhánh kia che đi.**
Nếu chỉ chạy cấu hình offload — tức là cấu hình cho 128K — campaign sẽ ra một
bảng đầy đủ, chạy trơn, và sai toàn bộ với ours và Quest.

Sửa xong: hai test mới (`test_gather_buffer_holds_the_widest_exact_region`,
`test_block_id_fallback_gathers_the_same_rows_as_the_reuse_kernel`) — **đã xác
minh cả hai FAIL trên code cũ**. Suite: **167 passed / 4 failed** (bốn cái fail
là bốn cái cũ). Sáu ô chạy lại: cả sáu score 1 và prediction **md5 giống hệt
nhau** — GPU và offload cho ra cùng một byte, ShadowKV GPU và CPU cũng vậy.

---

## ⭐ 2026-09-13 — Reuse: đo thật 60%, và quyết định để bảng runtime công bằng

Thêm `STREAMING_GATHER_REUSE` (mặc định 1) và `STREAMING_REUSE_STATS`. Bốn ô
`qwen3 / 32768 / niah_single_1 / b1024 / n=2`, frame `x32 l256 i256`, offload bật:

| | block tái dùng | score | md5 prediction |
|---|---|---|---|
| ours, reuse ON | **62.8%** (532 405/847 872) | 1 | `f4fe458f4a77` |
| ours, reuse OFF | 0% | 1 | `f4fe458f4a77` |
| Quest, reuse ON | **60.4%** (512 304/847 872) | 1 | `f4fe458f4a77` |
| Quest, reuse OFF | 0% | 1 | `f4fe458f4a77` |

**Reuse không đụng accuracy** — bốn ô cùng một byte. Campaign accuracy cứ bật.

**Công bằng:** giữa ours và Quest reuse đối xứng (62.8 vs 60.4). Bất đối xứng
nằm ở **block vs token**: ParisKV và PQCache chọn theo token nên đi nhánh
per-token, và ta chưa cài reuse cho nhánh đó. Đó là thiếu sót kỹ thuật của ta,
không phải tính chất method — so 1024 id thay vì 128 vẫn làm được.

⇒ **BẬT reuse, mặc định, mọi lúc.** (Tôi đã đề xuất ngược lại và bị bác — xem
mục SỬA LẠI ngay dưới.) `STREAMING_GATHER_REUSE=0` giữ làm cột chẩn đoán.

**ShadowKV có reuse không?** *Có, trong thiết kế gốc* —
`reorder_keys_and_compute_offsets` tính hit/miss so với chunk đã chọn ở bước
trước, rồi `gather_copy_with_offsets` làm D2D cho hit và H2D cho miss. Đó là
phần "cache-aware fetch" của paper. **Bản của ta không có**: bản CPU gọi
`cnts.zero_()` (mọi slot thành miss) vì streaming id phá sổ sách prefill của tác
giả, và đường V hiện dùng `torch.gather` nên bỏ qua `cnts` hoàn toàn. Bản GPU
không cần reuse vì không có PCIe.

Test mới `test_cross_step_reuse_changes_traffic_but_not_rows`. Suite:
**168 passed / 4 failed**.

---

## ⚠ 2026-09-13 — SỬA LẠI: tắt reuse cho "công bằng" là đề xuất sai

Tôi đề xuất chạy bảng runtime với `STREAMING_GATHER_REUSE=0` cho mọi method, lý
do: ParisKV/PQCache không có reuse nên bật cho ours/Quest là so kỹ thuật của ta
với kỹ thuật của họ. **Sai ở chỗ căn bản: ShadowKV có reuse ngay trong paper.**

Reuse giữa hai bước không phải tối ưu hoá ta tự nghĩ ra — nó là kỹ thuật đã công
bố của chính dòng literature này. Tắt nó đi sẽ làm số của ta **tệ hơn số
ShadowKV đã công bố**, và đó mới là bảng sai lệch.

Cách xử lý đúng khi một baseline có thể nhanh hơn nếu được tối ưu là **ghi rõ
điều đó trong bảng**, không phải tự làm chậm mình xuống bằng nó. Bảng ghi cột
"có reuse hay không" cho từng method; `=0` thành cột chẩn đoán để tách riêng
hiệu ứng bề rộng record khi có người hỏi.

Còn một việc thật sự cần làm: **trả lại reuse cho ShadowKV-CPU**. Hiện bản của
ta gọi `cnts.zero_()` nên nó mất reuse mà paper có. Đó là chỗ duy nhất ta hơn
baseline vì *ta làm hỏng baseline* — phải sửa trước khi báo cáo runtime.

---

## ⭐ 2026-09-13 — ShadowKV-CPU dùng chung kernel offload: nhanh 3×, bit-exact

Kho pinned của ShadowKV là `[B,H,chunks,chunk*D]` — cùng byte, cùng thứ tự với
`[B,H,tokens,D]`. Nên kernel block của ta đọc thẳng với `block_size = chunk_size`
và **chunk id của tác giả chính là block id**. Thêm entry point chỉ-values
(`gather_blocks_reuse_values`, template `with_keys=false`) vì ShadowKV dựng K từ
low-rank trên GPU — kéo K qua PCIe rồi vứt là phí gấp đôi.

ms mỗi layer, 32K, budget 1024, 2 MB khi cold:

| | 0% | 60% | 90% |
|---|---|---|---|
| chunk 4, kernel ta | **0.211** | 0.120 | 0.061 |
| chunk 8, kernel ta | **0.184** | 0.086 | 0.038 |
| chunk 8, kernel tác giả | 0.358 | — | 0.051 |
| `torch.gather` CPU (bản cũ) | 0.630 | — | — |

* chunk 4 **nhanh 3.0×** so với bản cũ (5.3× ở mức reuse thật).
* Kernel của ta **nhanh hơn kernel tác giả ngay ở chunk 8 gốc** — hết đánh đổi
  giữa "đúng paper" và "chạy nhanh".
* Reuse thật của ShadowKV ở 32K: **54.4%** (922 580/1 695 744), chunk 4, score 1.
  So với ours 62.8% và Quest 60.4% — ba method cùng một vùng.

**Bit-exact**: ô `qwen3 8192 niah_single_1 shadowkv_cpu b1024` ra md5
`8599ed6a3c74`, giống hệt đường `torch.gather`. Đổi cách chuyển dữ liệu, không
đụng thuật toán.

Bẫy đã xử: `clear()` phải `stage_ids.fill_(-1)` — id còn sót từ mẫu trước sẽ
khiến kernel chép lại hàng cũ cho một block id tình cờ trùng. Đây là loại lỗi
không crash và không lộ ở ô nhỏ.

Suite: **168 passed / 4 failed**.

---

## ⭐ 2026-09-13 — ParisKV: hạ tầng offload của tác giả, và reuse có giúp được nó

ParisKV chính thức **có** offload, thiết kế gần trùng của ta: pinned CPU giữ
toàn bộ KV (`unified_keys_cpu`), sink+local ở GPU, fetch bằng kernel CUDA riêng
`h2d_gather_kv` (`cache_hub/gather_trans/trans_h2d.cu`) — grid `(bs,heads,topk)`,
32 thread/block, 16 cho K và 16 cho V, mỗi thread một `uint4`. **Per-token,
record 256 B, không reuse.** Hai bên viết độc lập ra cùng một thiết kế; bản của
ta faithful.

Đơn vị truy hồi của nó **thật sự là token** — `collision_based_topk_batch` trả
`topk_indices: [bs,kv_heads,final_topk]`, RaBitQ rerank từng key. `cache_unit_size: 8`
trong config Qwen của họ nghe như block nhưng `polar_cache.py` nhận rồi **không
dùng ở đâu cả** — knob chết.

**Đo độ trùng tập được chọn giữa hai bước decode, 32K, b1024:**

| method | đơn vị | độ trùng |
|---|---|---|
| ours | block 8 | 62.8% |
| Quest | block 8 | 60.4% |
| **ParisKV** | **token** | **58.3%** |
| ShadowKV-CPU | chunk 4 | 54.4% |

**Độ ổn định của lựa chọn không phụ thuộc đơn vị truy hồi.** Bốn method nằm
trong dải 54–63%. Nên reuse không phải đặc quyền của method chọn theo block —
nó chỉ là thứ ai cài thì có.

Kernel reuse hiện có chạy luôn ở mức token với `block_size=1`, **không cần code
mới**: 1.215 ms (không reuse) → 1.159 ms (kernel reuse, 0% trùng) → **0.629 ms**
(57% trùng). Vòng so khớp O(K) nối tiếp không thành nút cổ chai vì bị latency
PCIe che.

⇒ ParisKV và PQCache có thể nhanh **~1.9×** bằng đúng kernel đang có. **Chưa
cài** — đây là quyết định về mức trung thành với baseline, không phải kỹ thuật.

---

## ⚠ 2026-09-13 — SỬA LẠI: đo runtime mà chạy song song là hỏng số

Tôi thả 4 ô runtime 128K song song trên 4 GPU của m1. Sai. Bốn tiến trình tranh
**PCIe, RAM pinned và CPU** — mà đó chính là ba thứ đang đo. RAM m1 tụt từ 175 GB
còn 39 GB (mỗi tiến trình offload giữ 18 GB pinned), và ParisKV OOM trong prefill.

Với ô **accuracy** thì song song hoàn toàn ổn — kết quả không phụ thuộc hàng xóm.
Với ô **runtime** thì không: **một method một lúc, cùng một card**. Chậm hơn
nhưng đó là điều kiện để con số có nghĩa.

Quy tắc từ nay: pool song song cho accuracy, runner tuần tự cho runtime, và ghi
rõ GPU nào trong bản ghi (số PCIe của từng slot không giống nhau).

### Điều smoke 8K đã cho thấy trước khi có bảng

`quest_streaming` 8K b1024 offload: prefill 1.90 s (4166 tok/s), decode
**56.2 ms/bước** → 17.8 tok/s. Nhưng 36 layer × 0.06 ms gather ≈ **2 ms**. Vậy
54 ms còn lại **không phải PCIe** — là router và overhead mỗi layer.

Nghĩa là mọi tối ưu offload hôm nay động vào **~4%** thời gian decode. Đừng để
bảng GB/s che mất chuyện đó.

---

## ⭐ 2026-09-13 — Profile: decode của ours **không** bị nghẽn ở GPU

64K, B=4096, offload, `STREAMING_ROUTER_BACKEND=triton`,
`SHADOWKV_RMSNORM_BACKEND=flashinfer`, A5000 GPU1, tuần tự. `torch.profiler`
trên 8 bước decode.

| mỗi bước decode | ours | Quest | full (dense) |
|---|---|---|---|
| wall (không profiler) | **124.56 ms** | 47.65 ms | 30.97 ms |
| thời gian kernel CUDA | **43.6 ms** | 36.7 ms | — |
| thời gian CPU | **154.5 ms** | **40.7 ms** | — |
| `cudaLaunchKernel` | **5 532** | **1 102** | — |
| GPU bận (theo wall thật) | **35%** | 77% | — |
| `aten::nonzero` | **144** | **0** | — |
| `aten::index` | **625** | **0** | — |
| view+reshape+select+slice | **6 755** | 2 216 | — |

**Ours chỉ làm nhiều hơn Quest 19% việc trên GPU (43.6 vs 36.7 ms) nhưng mất
2.6× thời gian tường.** Khoảng 81 ms mỗi bước — **65%** — là CPU đứng phát lệnh
và GPU chờ.

Chi phí dùng chung, để thấy phần nào *không* phải lỗi của router:

* `gather_blocks_reuse_16byte_kernel`: **391 µs/layer ở ours, 441 µs ở Quest**
  → 14.1 và 15.9 ms mỗi bước. Đây là **32–43% toàn bộ thời gian GPU của cả
  hai** — ở B=4096 việc chuyển 4096 token KV mỗi layer mỗi token sinh ra là
  chi phí thật, không tránh được. (Quest tốn hơn vì reuse thấp hơn: 60.4% so
  với 62.8%.)
* `aten::mm`: 13 ms/bước, giống hệt nhau — GEMM của model.

**Thủ phạm là `aten::nonzero` (144 lần/bước) và `aten::index` (625 lần/bước),
Quest không có cái nào.** `nonzero` có shape phụ thuộc dữ liệu nên **ép đồng bộ
device→host**: 4 lần mỗi layer, 144 lần mỗi token. Mỗi lần rút cạn hàng đợi CUDA
và phơi toàn bộ độ trễ ra. Cộng thêm `.item()` rải rác
(`adaptive_centroid_streaming_cache.py` dòng 1397, 1428, 1437, 3419, 3429,
3511, 3525, 3532, 3570, 3638, 3648) — mỗi cái là một điểm đồng bộ nữa.

Ý nghĩa: **thuật toán không đắt, cách cài đặt mới đắt.** Nếu bỏ được các điểm
đồng bộ và gộp bớt op, ours sẽ về quanh **50–55 ms**, tức ngang Quest. Đó là
việc sửa code, không phải đổi method.

Sửa lại điều tôi nói trước khi đo: tôi đoán "~260 kernel mỗi layer, launch-bound"
và rằng Triton router sẽ cứu được. Triton **đang chạy thật** (log in
`router triton | metadata compact center-int8`) và chỉ mua được **3.5%**
(129.14 → 124.56 ms), vì `_packed_lse_kernel` chỉ chiếm **4.5%** thời gian GPU.
Tối ưu đúng chỗ phải là **bỏ đồng bộ**, không phải làm router nhanh hơn.

---

## ⭐ 2026-09-13 — Tìm ra chỗ decode chậm: commit `c68b1415` xoá đường exact-seal

Đo bằng `repro/shadowkv/profile_streaming_decode.py`, 32K, B=1024, block 8,
prefix/recent 32, triton, uva, KV trên GPU, A5000 GPU1 **rảnh hoàn toàn**.

| component (median ms) | 10/09 (kho lưu) | hôm nay, mọi commit |
|---|---|---|
| attention | 2.121 | 2.191 |
| gather | 6.701 | 6.65 |
| score | 7.603 | 11.75 |
| **update** | **3.179** | **55–59** |
| **decode median** | **59.98** | **111.73** |

Chạy commit `71a83f31` (ngay trước khi xoá) với hai chế độ:

| `SHADOWKV_STREAMING_SEAL_MODE` | decode median | update | score |
|---|---|---|---|
| `adaptive` (default) | 99.96 ms | **55.34** | 6.60 |
| `exact` | **49.58 ms** | **5.30** | 6.56 |

**Commit `c68b1415` "Remove exact decode seal path" (11/09) đã xoá đường nhanh
đó**, không kèm lý do trong message. Comment của chính đoạn code bị xoá mô tả
đúng triệu chứng:

> *"Building its complete self-K hierarchy synchronously **stalls every eighth
> token**."*
> *"indexing that pinned tensor, copying it back to CUDA and reducing it to an
> unused mean **accounted for most of the remaining decode-block update
> latency**."*

Đường đó lưu block decode vừa seal thành **component đơn lẻ chính xác**
(`centers = blocks`, counts = block_size) thay vì chạy fit centroid đầy đủ, và
tránh chạm vào V cache pinned. Code cũ tự nhận nó *"strictly more faithful than
an approximate path"* — nên khôi phục không phải là đánh đổi accuracy lấy tốc độ.

Hồi quy này **có từ cây cũ**, không phải do tách repo: checkout
`shadowkv-paris-retro-selfk128k` đo hôm nay cũng cho `update` 55.57 ms.

### ⚠ SỬA LẠI trong cùng phiên: con số "score 406 ms" là do tôi làm hỏng phép đo

Tôi báo `score` tăng từ 7.6 lên **406 ms** và gọi đó là hồi quy của repo mới.
**Sai.** Lần đo đó chạy khi `bench_latency.py` vẫn đang chiếm GPU1. Chạy lại với
GPU rảnh: **11.75 ms**, giống mọi commit từ `40b0506` tới HEAD. Bisect 8 commit
đều cho 11.4–11.9.

Đây là **đúng lỗi tôi vừa ghi vào file này một giờ trước** ("ô runtime phải chạy
một mình"). Ghi lại cho rõ: **trước mỗi phép đo tốc độ, kiểm `pgrep` và
`nvidia-smi` — không kiểm thì số đo vô nghĩa.**

---

## ⭐ 2026-09-13 — Cài nốt phần "defer" mà khung đã hứa: decode 124.6 → 46.2 ms

Khung bốn vùng chép từ ParisKV (`dynamic_update_interval: 512`, ta dùng 256).
**Toàn bộ lý do con số đó tồn tại là để việc dựng chỉ mục thành một phép theo
lô** — ParisKV gọi `_update_polar_index()` → `batch_encode()` cho cả 512 token
một lần (`cache_hub/polar_cache.py:875`). Ta lấy cách kế toán vùng nhưng giữ
nguyên fit từng block mỗi 8 token. Hàm `_build_or_defer_blocks` có docstring ghi
*"or queue it while the block is recent"* và thân hàm gọi thẳng `_build_blocks`.

**Vì sao hoãn là an toàn theo cấu trúc:** metadata của một block chỉ cần khi
block được phép truy hồi, và `active_blocks = min(sealed, (total − local −
buffer) // block)` đảm bảo điều đó không xảy ra trước flush. Cùng block được
chọn, cùng centroid.

Hai chỗ phải sửa, chỗ thứ hai suýt bị bỏ sót:

1. `StreamingBlockCache._build_or_defer_blocks` — xếp hàng đợi, flush khi
   `pending_buffer` quấn về 0. Phải `clone()` key vì staging bị ghi đè sau 8 token.
2. **`StreamingAdaptiveCentroidLSECache` đã override hàm đó** bằng cơ chế rải
   riêng (`layer_idx % block_size`). Nên sửa (1) xong đo lại **không đổi gì**
   (124.56 → 124.64). Cơ chế rải chỉ trải đều chi phí: mỗi token vẫn trả một
   phần hoá đơn 36 layer. Cho override gọi `super()` khi frame có buffer.

### Kết quả, 64K B=4096, offload, A5000 GPU1 rảnh

| | mean | median | p90 |
|---|---|---|---|
| trước | 124.56 | 129.47 | 140.09 |
| sau, **576 bước (2 chu kỳ flush)** | **46.20** | **43.79** | **48.34** |

**Nhanh 2.7×**, và đuôi biến mất (p90 204 → 48).

Chi phí flush quy ra: mean 576 bước 46.20 so với 43.21 khi đo 72 bước (chưa
flush lần nào) ⇒ **~864 ms mỗi flush** cho 36 layer × 32 block = **0.75 ms mỗi
block**, so với **12.8 ms mỗi block** ở đường rải. **Gộp lô hiệu quả hơn 17×.**

**Accuracy bit-exact**: ô `qwen3 16384 niah_single_1 ours b1024` cho md5
`5a1cbf92bc1f` trước và sau. Test mới
`test_deferred_block_build_selects_exactly_what_immediate_build_selects`
(đã xác minh FAIL khi tắt flush). Suite **169 passed / 4 failed**.

### ⚠ Bẫy đo đếm: interval 256 mà chỉ đo 64 bước là đo phần rẻ

Lần đo đầu sau khi sửa cho 43.21 ms — nhưng 72 bước không đủ chạm lần flush nào
(cần 256). Đo đủ 576 bước mới ra 46.20. **Mọi ô runtime của method có batch
flush phải chạy ít nhất hai chu kỳ `update_interval`**, nếu không con số là giả.

---

## ⭐ 2026-09-13 — Reuse cho method chọn theo token: sai một lần rồi sửa

Theo yêu cầu, cài reuse cho ParisKV và PQCache. Không cần kernel mới về nguyên
tắc: đơn vị truy hồi là token thì "block id" chính là "token id", tức
`block_size=1`.

* **PQCache** — thêm hai bank K/V theo layer + bank id, gọi
  `gather_blocks_reuse_uva(block_size=1)`.
* **ParisKV** — đi qua `StreamingBlockCache`, nên tổng quát hoá thay vì viết
  riêng: thêm `retrieval_unit` (= block_size nếu chọn theo block, **1** nếu
  chọn theo token), cấp bank id theo đơn vị đó, bỏ điều kiện `block_selection`
  khỏi `_can_reuse_selected_blocks()`. Vùng exact vẫn đi qua `exact_ranges` của
  kernel nên phần chọn động giữ bề rộng cố định mỗi bước — **điều kiện cần để
  so khớp id hợp lệ**.

### ⚠ Lần đầu làm bảng TỆ ĐI, vì tôi ngoại suy sai

| 64K B=4096 | không reuse | reuse, match nối tiếp | reuse, match song song |
|---|---|---|---|
| ParisKV | 204.70 | **295.06** | **84.57** |
| PQCache | 177.15 | **339.23** | **86.79** |

Tôi đo kernel ở `block_size=1` với **K=1024** (1.159 ms, không phạt gì) rồi suy
ra nó dùng được ở budget 4096. Sai: vòng so khớp chạy **nối tiếp trong thread 0**
và quét cả K slot, nên tổng là **O(K²)** — K gấp 4 thì so khớp gấp **16×**. Ở
ours/Quest, K = 4096/8 = **512** nên rẻ; ở đơn vị token K = 4096–4384 nên nó ăn
hết phần tiết kiệm rồi ăn thêm.

**Sửa trong kernel**: quét trải đều trên cả thread block, `atomicMin` giữ match
nhỏ nhất (đúng ngữ nghĩa `break` cũ), và nới số thread lên 256 khi bề rộng chép
hẹp để phần quét có chỗ song song — thread thừa thoát trước khi chép.

⇒ ParisKV **2.42×**, PQCache **2.04×** so với không reuse.

**Bit-exact**: ô `qwen3 16384 niah_single_1 b1024` cho md5 `5a1cbf92bc1f` với
ParisKV / PQCache / ours, cả `STREAMING_GATHER_REUSE` 0 và 1, cả trước và sau
khi đổi kernel. Suite **169 passed / 4 failed**.

**Bài học lặp lại lần thứ tư hôm nay**: ngoại suy từ một điểm đo sang một chế độ
khác là đoán, không phải đo. Ba lần trước: "không phải băng thông" (sai),
"budget gây ra" (sai), "score hồi quy 53×" (tự làm nhiễu phép đo).

---

## ⚠ 2026-09-13 — ShadowKV-CPU chưa bao giờ ở trên khung chung

`ShadowKVCache_CPU` **hard-code** `recent_tokens = 32` và `update_tokens = 8`;
chỉ bản GPU đọc `STREAMING_RECENT_TOKENS` / `STREAMING_UPDATE_INTERVAL`. Log của
mọi ô hôm nay in `rolling recent 32 | stream update 8` trong khi các method khác
chạy `recent 256 | interval 256`.

Hệ quả nếu không phát hiện: bảng accuracy sẽ so ShadowKV-CPU (vùng exact **32**
token, seal mỗi **8** token) với mọi method khác (vùng exact **256–511**, gộp
**256**). Đó là hai giao thức khác nhau, không phải hai method khác nhau.

Sửa: đọc env như bản GPU. Kéo theo hai lỗi kích thước buffer:

* đuôi sinh cố định 128 token nhỏ hơn đệm flush 256 → `IndexError` ở token 129;
* `target_local = recent_tokens + prompt % chunk_size`, nên vùng local có thể
  dư tới `chunk_size − 1` token (prompt 64 902 lẻ 2 → local 258 chứ không phải
  256) → `IndexError` lần hai.

Đuôi giờ là `max(128, update_tokens) + chunk_size`.

**Runtime 64K B=4096:** 67.21 ms (khung cũ) → **57.84 ms** (khung 256).

Không kiểm bit-exact được ở đây, và đó là điều đúng: đổi vùng exact từ 32 lên
256 **là** đổi giao thức, nên kết quả phải đổi. Mọi số accuracy của
`shadowkv_cpu` trước thời điểm này đều ở khung sai — may là campaign accuracy
chưa chạy.

Câu hỏi của user đã lôi ra chỗ này: *"tôi tưởng cũng phải 256 lần mới làm 1 lần?"*

---

## ⚠ 2026-09-13 — Bảng runtime bị nhiễu nhiệt: vị trí trong hàng đợi đổi kết quả 20%

Cùng ô `shadowkv_cpu 64K B=4096`, cùng commit, cùng GPU1, cùng 576 bước:

| | decode mean | p90 |
|---|---|---|
| chạy riêng, card vừa nghỉ | **57.84** | 60.06 |
| chạy thứ 6 trong bảng, card nóng | **69.34** | 69.57 |

GPU1 đo được **82–85°C**, và một mẫu bắt được SM clock **1005 MHz trên trần
2100**. Ô `full` luôn chạy **đầu tiên** nên rất ổn định (30.93 / 30.97 / 31.25),
còn ô chạy sau bị phạt — tức **thứ tự tôi đặt trong script đang quyết định thứ
hạng**.

Đây không phải nhiễu ngẫu nhiên mà là sai lệch có hệ thống, và nó lớn hơn khoảng
cách giữa ours và Quest (43.75 so với 44.13). Mọi bảng runtime trước thời điểm
này đều mang sai lệch đó.

**Sửa trong `measure_runtime`:**

* **cổng làm nguội** trước khi đo — chờ tới `SHADOWKV_RUNTIME_COOL_C` (mặc định
  55°C), tối đa `SHADOWKV_RUNTIME_COOL_TIMEOUT` giây, rồi mới prefill;
* **ghi bối cảnh nhiệt vào mỗi ô**: `start_temperature_c`, `start_clock_mhz`,
  `max_temperature_c`, `min_clock_mhz`, `mean_clock_mhz` (lấy mẫu mỗi 64 bước).

Con số không kèm nhiệt độ và xung nhịp thì không so được với con số đo ở vị trí
khác trong hàng đợi. Từ nay mọi ô runtime đều mang theo bối cảnh đó, và nếu
cooldown hết giờ thì ô tự in cảnh báo.

---

## ⭐ 2026-09-13 — ShadowKV-CPU: PCIe rẻ đúng như kỳ vọng, nghẽn nằm ở CPU

Profile `shadowkv_cpu` 64K B=4096, khung 256, card làm nguội trước.

| op | % thời gian GPU |
|---|---|
| GEMM của model | 28.6% |
| **`torch.cat` landmark** | **30.2%** |
| GEMM dựng lại K + chấm landmark | 21.3% |
| topk | 10.1% |
| **gather V qua PCIe** | **15.0%** |

**Việc chỉ chuyển V qua PCIe đúng là có lãi** — gather chiếm 15%, so với 32% của
ours. Kỳ vọng ban đầu không sai.

**Đã sửa `torch.cat`**: landmark sinh trong decode nằm ở pool riêng nên mỗi bước
mỗi layer phải nối lại với pool prompt (16 225 landmark ở 64K/chunk 4). Pool chỉ
đổi khi flush, nên nhớ lại kết quả nối. Xác minh bằng profile: `aten::cat` và
`CatArrayBatched` **biến mất khỏi top op**, tổng thời gian kernel GPU
**80.24 → 64.63 ms/bước (−19.5%)**. Giá: ~1.2 GB giữ bản đã nối.

**Nhưng wall không đổi** (199.9 → 222.2 ms dưới profiler), và **GPU bận tụt
0.401 → 0.291**. Vì ràng buộc không ở GPU:

| | số lần / bước |
|---|---|
| op indexing nhiều nhất | **4 290** |
| `cudaLaunchKernel` | **1 984** |
| ba op indexing tiếp theo | 1 476 + 1 296 + 1 265 |
| `copy_` (ring buffer) | 686 |

~12 000 dispatch mỗi bước decode, phần lớn là indexing tensor trong Python:
`self.k_cache_buffer[layer_idx, :, :, slot]` × 8 biểu thức × 36 layer, mỗi
biểu thức sinh vài op metadata. So sánh: Quest **1 102** launch/bước, ours 5 532.

⇒ Muốn ShadowKV-CPU nhanh hơn thì phải **giảm số dispatch**, không phải giảm
việc GPU. Cụ thể: bỏ ring buffer, dùng đuôi nối thêm như `StreamingBlockCache`
(2 copy mỗi layer thay vì 19), và dồn sổ sách vào flush.

**Cả hai chi phí này là của phần mở rộng streaming của TA**, không phải ShadowKV.
Bản paper không index token decode nên không có pool thứ hai để nối, và `recent`
chỉ 32 token nên ring rẻ.

---

## ⭐ 2026-09-13 — ShadowKV-CPU lên khung chung: 67.2 → 46.1 ms, bit-exact từng bước

Theo yêu cầu "phải thống nhất": đưa mọi việc per-token của phần mở rộng streaming
về nhịp flush 256, giống mọi method khác.

| bước | decode mean | GPU bận | launch/bước | CUDA/bước |
|---|---|---|---|---|
| khung cũ (recent 32, seal mỗi 8) | 67.21 | — | — | — |
| lên khung 256 | ~67 | 0.401 | 7 254 | 80.2 |
| + bỏ `torch.cat` landmark | 67.44 | 0.291 | 7 039 | **64.6** |
| + bỏ ring buffer | 49.11 | 0.379 | 6 391 | 64.8 |
| + gộp projection | **46.10** | **0.438** | **5 527** | 63.4 |

**Ba việc đã gộp:**

1. **`torch.cat` landmark** — pool decode nối với pool prompt mỗi bước mỗi layer.
   Nhớ lại kết quả, dựng lại khi flush.
2. **Ring buffer** — 8 biểu thức indexing mỗi token mỗi layer. Thay bằng **đuôi
   nối thêm**: attention trên key đã RoPE không phụ thuộc thứ tự trong buffer,
   nên vùng exact chỉ cần "cũ nhất ở đầu, mới nhất ở đuôi". Một phép chép slice
   mỗi bước; sổ sách đuổi token dồn vào flush.
3. **`_project_decode_keys`** — chạy mỗi token. `U` chỉ bị đọc khi chunk đã vào
   chỉ mục, mà chunk decode chỉ vào lúc flush, nên chiếu cả đợt 256 key ở đó.
   Kèm theo: `sv.float()` và `sv.square().sum(-1)` được tính lại **mỗi token**
   dù `SV` đóng băng sau prefill — giờ tính một lần.

**Bit-exact ở cả ba bước**: ô `qwen3 16384 niah_single_1 shadowkv_cpu b1024` giữ
md5 `5a1cbf92bc1f`, suite 169 passed / 4 failed.

### Điều hai phép đo này dạy

Bỏ `cat` giảm **19.5% việc GPU** mà wall **không nhúc nhích**. Bỏ ring **không
giảm việc GPU chút nào** (64.8 ms trước và sau) nhưng wall giảm **27%**.

Ở chế độ decode, thứ quyết định là **số dispatch**, không phải khối lượng tính
toán. Tối ưu FLOP hay băng thông khi GPU mới bận 29% là tối ưu sai chỗ.

---

## ⭐ 2026-09-13 — Kéo code RetroInfer về, và phải sửa lại hai điều tôi đã nói

**Repo là `microsoft/RetrievalAttention`**, không phải `microsoft/RetroInfer`
(URL đó không tồn tại). Clone về `/home/baonn/upstream-kv-methods/RetrievalAttention`,
commit `75829e63`. Chi tiết method và so sánh config: `PROJECT.md` §2i.

**SỬA LẠI 1: "RetroInfer không có đường offload" — SAI.** Offload *là* method.
Họ có `retroinfer_cache.py` (wave-buffer GPU–CPU, đầy `pin_memory`) và
`retroinfer_cache_gpu.py`; config mặc định `gpu_only: false`. Câu "RetroInfer
OOM ở 128K" đúng với **bản tự cài lại của ta**, không đúng với method.

**SỬA LẠI 2: RetroInfer nạp theo CỤM, không theo token.** `gather_copy_and_concat`
nhận `miss_unit_idices` / `miss_unit_sizes` — đơn vị là cụm kích thước biến
thiên. Cộng LRU cache thường trú trên GPU (`WaveBufferCPU`, class C++ trên thread
pool) chia hit/miss mỗi bước. Tức nó có **cả lợi thế bề rộng record lẫn reuse**,
không nằm cùng nhóm với ParisKV/PQCache như tôi từng xếp.

**Vùng ước lượng chỉ tốn ba vector mỗi cụm** (centroid, `value_sum`,
`cluster_size`), không phải token — nên "ngân sách" của nó không so trực tiếp
được với budget token của các method khác.

### Trạng thái từng method trước khi compact

| method | code tác giả | vị thế |
|---|---|---|
| ours | — | của ta |
| Quest | — | ta cài lại, 88 dòng, đã đối chiếu |
| ShadowKV | repo này là fork của họ | + mở rộng streaming của ta |
| ParisKV | `/home/baonn/ParisKV-official` | router của họ trên common path; **chưa đối chiếu với số công bố** |
| PQCache | `upstream-kv-methods/PQCache` | + reuse của ta |
| MagicPIG | `upstream-kv-methods/MagicPIG` | server CPU của họ |
| **RetroInfer** | **`upstream-kv-methods/RetrievalAttention`** (vừa kéo) | **bản tự cài lại, thiếu wave buffer/LRU/offload** |

### Số runtime đã đo (64K B=4096, chỉ để định hướng)

Đo ở nhiều thời điểm, nhiều trạng thái code và nhiệt khác nhau — **chưa có bảng
chạy một lượt**:

| | decode mean |
|---|---|
| full (dense, không offload) | 31.25 |
| ours | 43.75 |
| Quest | 44.13 |
| ShadowKV-CPU | 46.10 |
| PQCache (có reuse) | 79.16 |
| ParisKV (có reuse) | 90.98 |
| MagicPIG | 191.82 |
| ours @128K | 52.80 |

### Việc còn treo

1. **Bảng runtime 13 ô chạy một lượt** với cổng làm nguội — script sẵn ở
   `$SCRATCH/final.sh`, chưa chạy trọn.
2. **Campaign accuracy RULER 32K** — 91 ô, chưa thả.
3. **Chốt RMSNorm backend cho accuracy**: runtime dùng `flashinfer`, nhưng
   README ghi `torch` là default tham chiếu chất lượng và `compat.py` nói rõ hai
   backend cho số khác nhau. Đề xuất: `torch` cho accuracy, `flashinfer` cho
   runtime, ghi rõ trong PROJECT.md. **Chưa xác minh hai backend thật sự khác
   nhau** — hai ô 16K là đủ.
4. **Tích hợp RetroInfer author components** lên common path (§2i).
5. **Đối chiếu ParisKV** với số công bố của họ.
6. Chọn bản ShadowKV cho RULER: đã chốt **`shadowkv_cpu`**, vì ở `gen_len ≤ 128`
   phần mở rộng không chạy lần nào (flush cần 256) nên nó trùng bản paper.
