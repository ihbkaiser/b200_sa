"""Per-decode-step KV fetch cost: our UVA gather vs the ShadowKV author kernels.

Run from ``ShadowKV/``::

    CUDA_VISIBLE_DEVICES=3 python ../repro/shadowkv/bench_offload_gather.py


Geometry is one Qwen3-4B layer at 32K context, budget 1024: B=1, 8 KV heads,
head_dim 128, bf16.  Everything is timed with CUDA events on the current
stream, after a warmup.
"""
import os, sys, time
import torch

sys.path.insert(0, os.path.abspath("."))
from models.offload_gather import gather_kv_uva, gather_kv_torch, gather_blocks_reuse_uva
from kernels import shadowkv

DEV = "cuda"
B, H, D = 1, 8, 128
CTX = 32768
BUDGET = 1024
DT = torch.bfloat16
ITERS = 50


def timed(fn, iters=ITERS, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


def report(name, ms, megabytes):
    print(f"{name:<46} {ms:8.3f} ms   {megabytes:7.2f} MB   "
          f"{megabytes / 1024 / (ms / 1000):7.2f} GB/s")


print(f"device: {torch.cuda.get_device_name()}")
print(f"{'path':<46} {'latency':>11}   {'PCIe':>10}   {'eff. BW':>9}")
print("-" * 84)

# ---------------------------------------------------------------- ours: K+V
src_k = torch.zeros(B, H, CTX, D, dtype=DT, pin_memory=True)
src_v = torch.zeros(B, H, CTX, D, dtype=DT, pin_memory=True)
dst_k = torch.zeros(B, H, BUDGET, D, dtype=DT, device=DEV)
dst_v = torch.zeros(B, H, BUDGET, D, dtype=DT, device=DEV)
pos = torch.randint(0, CTX, (B, H, BUDGET), device=DEV, dtype=torch.int64)

kv_mb = B * H * BUDGET * D * 2 * 2 / 2**20
report("ours UVA  gather_kv (K+V, per token)",
       timed(lambda: gather_kv_uva(src_k, src_v, pos, dst_k, dst_v)), kv_mb)
report("ours torch fallback (K+V, per token)",
       timed(lambda: gather_kv_torch(src_k, src_v, pos, dst_k, dst_v), iters=10), kv_mb)

# ------------------------------------------------- ours: block reuse variant
BLK = 8
NBLK = BUDGET // BLK
prev_k = torch.zeros(B, H, BUDGET, D, dtype=DT, device=DEV)
prev_v = torch.zeros_like(prev_k)
dst_blocks = torch.zeros(B, H, NBLK, dtype=torch.int64, device=DEV)

for reuse in (0.0, 0.9):
    base = torch.randint(0, CTX // BLK, (B, H, NBLK), device=DEV, dtype=torch.int64)
    cur = base.clone()
    keep = int(NBLK * reuse)
    cur[..., keep:] = torch.randint(0, CTX // BLK, (B, H, NBLK - keep),
                                    device=DEV, dtype=torch.int64)
    ms = timed(lambda: gather_blocks_reuse_uva(
        src_k, src_v, prev_k, prev_v, base, cur, dst_k, dst_v, dst_blocks,
        block_size=BLK, exact_ranges=()))
    report(f"ours UVA  gather_blocks_reuse ({int(reuse*100)}% reuse)",
           ms, kv_mb * (1 - reuse))

# ------------------------------------------------------- ShadowKV author kernels
CH = 8
SETS = BUDGET // CH
GPU_ROWS = BUDGET + 128 + (48 + 4) * CH
v_cpu = torch.zeros(B, H, CTX // CH, CH * D, dtype=DT, pin_memory=True)
v_buf = torch.zeros(B, H, GPU_ROWS, D, dtype=DT, device=DEV)
k_buf = torch.zeros_like(v_buf)
temp = torch.zeros(B, H, SETS, CH * D, dtype=DT, device=DEV)
offsets = torch.zeros(B * H * SETS, dtype=torch.int32, device=DEV)
cnts = torch.zeros(B * H, dtype=torch.int32, device=DEV)
signals = torch.zeros(B * H, dtype=torch.int32, device=DEV)
cached = torch.full((B, H, SETS), -1, dtype=torch.int64, device=DEV)

cpu_v_len = CTX * D
gpu_v_len = BUDGET * D
kernel_offset = 0
kernel_stride = GPU_ROWS * D

v_mb = B * H * BUDGET * D * 2 / 2**20

for reuse in (0.0, 0.9):
    base = torch.randint(0, CTX // CH, (B, H, SETS), device=DEV, dtype=torch.int64)
    sel = base.clone()
    keep = int(SETS * reuse)
    sel[..., keep:] = torch.randint(0, CTX // CH, (B, H, SETS - keep),
                                    device=DEV, dtype=torch.int64)

    def step():
        cached.copy_(base)
        shadowkv.reorder_keys_and_compute_offsets(cached, sel, offsets, cnts, B, H, SETS)
        shadowkv.gather_copy_with_offsets(
            v_cpu, v_buf, temp, offsets, cnts, signals, B, H,
            cpu_v_len, gpu_v_len, kernel_offset, kernel_stride, SETS)

    report(f"ShadowKV author gather_copy_with_offsets ({int(reuse*100)}% hit)",
           timed(step), v_mb * (1 - reuse))

    def step_k():
        cached.copy_(base)
        shadowkv.reorder_keys_and_compute_offsets(cached, sel, offsets, cnts, B, H, SETS)
        shadowkv.gather_copy_d2d_with_offsets(
            k_buf, offsets, cnts, B, H, gpu_v_len, kernel_offset, kernel_stride, SETS)

    report(f"ShadowKV author gather_copy_d2d (K side, {int(reuse*100)}% hit)",
           timed(step_k), 0.0 if reuse == 1 else 0.0)

# ------------------------- what our repo actually runs today for ShadowKV-CPU
gather_temp = torch.empty(B, H, SETS, CH * D, dtype=DT, pin_memory=True)


def torch_gather_path():
    ids = cached.to(device="cpu")
    torch.gather(v_cpu, 2, ids.unsqueeze(-1).expand(-1, -1, -1, v_cpu.shape[-1]),
                 out=gather_temp)
    v_buf[:, :, :BUDGET].copy_(gather_temp.view(B, H, BUDGET, D), non_blocking=True)


report("repo today: ShadowKV-CPU torch.gather on CPU",
       timed(torch_gather_path, iters=10), v_mb)
