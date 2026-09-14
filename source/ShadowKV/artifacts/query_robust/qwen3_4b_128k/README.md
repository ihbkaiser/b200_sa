# Qwen3-4B Query-Robust 128K vertices

This directory vendors the frozen BF16 QR-vertices asset from the public
`ihb-sparse` branch `query-robust-paper-b200`:

`remote_artifacts/modal_20260913_145407_qwen3_qr_128k_final`

The asset is calibration metadata, not model weights. It is intended for the
`Qwen/Qwen3-4B-Instruct-2507` configuration with 36 layers, 8 KV heads, head
dimension 128, and `M=32` support vertices. The calibration used 20 tokenized
Pile contexts of exactly 131072 tokens, with 3000 captured query samples per
query head. It does not itself establish a RULER accuracy or throughput claim.

The runtime validates shape, BF16 storage, model identity/fingerprint, RoPE
metadata when supplied, padding semantics, and the checksum in
`manifest.json` before loading the asset.
