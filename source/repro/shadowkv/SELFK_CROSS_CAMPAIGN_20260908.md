# Query-independent self-K center campaign (2026-09-08)

## Question

Can a clean query-independent rule improve the placement and allocation of
1.25--1.50 centers per positional block of eight tokens enough to match
ParisKV at a matched 2B-to-B proposal budget?

All diagnostics use Qwen3-4B-Instruct-2507, post-RoPE K, prefix=32,
recent=32, B=512, and eight fixed RULER-32K traces (three CWE, two
MultiKey-3, QA-1, QA-2, and VT).  Future queries are used only for evaluation.

## Falsified directions

- Cross-block mean/CVaR/entropic self-K risk at 16- and 64-block scales.
- Block exposure, novelty, leverage and uncertainty bonuses.
- Budget-free MDL/Lagrangian stopping: competitive mass required 6--8
  components/block.
- RoPE-transported self-K proxies.
- Low-rank covariance/log-MGF summaries.
- Spherical-Ward and support-function center objectives.
- Residual-singleton and 512-token KeyDiff-pool singleton placement.

The current-query finite-family oracle closes 56% of the original
self-K-to-exact mass gap, so a useful low-order partition exists.  The
query-free features cannot predict it reliably enough over future queries.

## Best diagnostic result

| selector | retained mass | annotated target recall |
|---|---:|---:|
| exact block-8 total mass | 89.214% | 1.003% |
| angular isolation r=1.50 + exact 2B rerank | **88.382%** | 0.974% |
| residual isolation r=1.50 + exact 2B rerank | 88.382% | 0.970% |
| KeyDiff-pool isolation r=1.50 + exact 2B rerank | 88.300% | **0.992%** |
| deployed self-K r=1.25 + exact 2B rerank | 87.822% | 0.933% |
| ParisKV, exactly 2B candidates | 86.828% | **1.278%** |
| ParisKV, 2B candidates + exact rerank | 86.881% | 1.302% |

Paris' default 10% candidate pool retains 88.883% mass but is approximately
6.4B at B=512, not a matched 2B comparison.  At matched 2B, ours recovers more
aggregate attention mass while Paris recalls more rare answer-bearing tokens.
This separation predicts the generation result.

## CWE-32K generation gate (30 samples)

| selector | B=512 | B=2048 |
|---|---:|---:|
| deployed self-K r=1.25 + exact 2B rerank | 65.94 | **81.66** |
| angular r=1.25 + exact 2B rerank | 63.86 | 78.57 |
| angular r=1.50 + exact 2B rerank | **67.97** | 78.51 |
| ParisKV, exactly 2B candidates | **80.96** | **83.54** |

For reference, live exact block-8 LSE routing scores 68.60 on the same first
30 samples at B=512 (68.42 over 100).  Angular r=1.50 is already within 0.63
point of that block-level ceiling.  Consequently the remaining 13-point gap
to Paris cannot be closed by further optimizing the number or placement of
centers while preserving this routing unit and objective.

## Conclusion

This campaign is a clear negative result for the proposed search space.
Angular isolation is a cheap experimental option (+2.03 CWE points over
self-K at B=512 and 1.43x faster fitting), but it regresses at B=2048 and must
not replace the default.  KeyDiff-pool increases answer-token recall only
marginally and remains far below Paris.  No variant passes the precommitted
32K gate, so running an expensive 128K sweep would not be scientifically
justified.

Closing the gap requires changing the candidate semantics or granularity
(token/coherent-cluster retrieval), not another allocator or center objective
inside fixed positional blocks of eight.  That is intentionally outside this
campaign.

## Reproducibility

- Diagnostic implementation: `repro/shadowkv/diagnose_router_mass_full_trace.py`
- Launcher: `repro/shadowkv/run_cross_block_center_probe.sh`
- Optional angular production path: `SHADOWKV_CENTER_PLACEMENT=angular`
- Main artifacts: `/storage/baonn/selfk_cross_campaign_20260907/`
- Machine-2 artifacts: `/home/nbnguyen/selfk_cross_campaign_20260908/`
- Machine-4 artifacts: `/storage/nbao/selfk_cross_campaign_20260908/`
- Detailed live ledger: `/storage/baonn/selfk_cross_campaign_20260907/RESULTS.md`
- Final code commit: `36ff7e14`
