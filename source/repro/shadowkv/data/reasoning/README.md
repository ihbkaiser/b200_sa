# Reasoning benchmark data — committed on purpose

The repository's rule is that datasets stay out of git, because they are
per-machine and reach the code through `env_m*.sh`. These three are the
exception, and the reason is the bug that produced them.

AIME25 exists on this project as two parquet files. `kvpress_aime25_percontext`
drops the "Remember to put your final answer within \boxed{}" instruction that
`kvpress_aime25_local` carries, so the extraction scorer reads a different
number off the same model. m1 pointed at one, m2/m3/m4 at the other, and an
AIME cell run on m3 would not have merged with m1's — with no symptom beyond
"m3 scores lower". Per-machine copies of a small eval set drift, and a drift
this quiet is worth 320 kB of git.

So these are canonical: every `env_m*.sh` resolves the three benchmarks here by
default, and a machine gets them by `git pull` rather than by a copy someone
has to remember to make. Override the env var if you deliberately want another
variant; nothing else should.

| directory | rows | max_new_tokens | sha256 of the parquet |
|---|---|---|---|
| `kvpress_aime25_local`  |  30 | 32000 | `b22cc024ea0fa4412c3649ab4a22957ec48219602d5d3e0cf74d8a617608fb57` |
| `kvpress_math500_local` | 500 |  4096 | `5649d8811e7c7c99413fc60848d6cc13630a113dee9b2096131daf579d1ce58d` |
| `kvpress_gpqa/diamond`  | 198 | 16384 | `1cf910c4124d471610b65a2a4115f010c28d61f011ef3ffef8ba371fc3eec906` |

GPQA is shipped as `kvpress_gpqa/diamond` because the loader takes the parent
directory and selects the subset with `data_dir=`. The other GPQA subsets
(main, extended, the probe splits) stay out: nothing here reads them.

`make_reasoning_data_bundle.sh` still exists for a machine that cannot reach
git at all.
