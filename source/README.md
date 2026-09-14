# sparse_attention

Sparse-attention KV retrieval methods on a single common forward path, with the
campaign machinery used to benchmark them.

```
ShadowKV/models/     the caches: ours, Quest, ParisKV, ShadowKV, PQCache,
                     MagicPIG, RetroInfer, full attention
ShadowKV/test/       eval_acc.py + evaluator
ShadowKV/data/       benchmark loaders (LongBench-v2, RULER, GPQA, MATH-500, AIME)
repro/shadowkv/      pool, preflight, cell naming, status, reports, per-machine env
```

Datasets, model weights and results are NOT in this repo. Each machine exports
its own paths from `repro/shadowkv/env_m<N>.sh`; results go to that machine's
large mount. See `PROJECT.md` for the machine roster and `RECORDS.md` for the
running campaign log.
