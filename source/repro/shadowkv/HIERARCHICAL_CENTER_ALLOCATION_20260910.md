# Smooth hierarchical center allocation (2026-09-10)

## Objective

For each KV head, let (R_G(r)) be the measured self-K tail-CVaR routing
distortion of temporal block (G) when represented by (r\in\{1,\ldots,8\})
centers, and let (c_h=|\mathcal G_h|^{-1}\sum_G(r_G-1)).  The allocator
minimizes

\[
  |\mathcal G_h|^{-1}\sum_G R_G(r_G)
  +\lambda_0c_h+\alpha\log(\beta+c_h).
\]

The resulting marginal center price is

\[
  \lambda_h(c_h)=\lambda_0+\frac{\alpha}{\beta+c_h}.
\]

It therefore falls smoothly when a head has coherent evidence for several
useful upgrades, but never below the floor \(\lambda_0\).  There is no task
label, hard easy/hard threshold, or fixed mean-center quota.  Concavity of the
log gives an MM update: at iteration (t), solve every block's exact discrete
priced problem with price \(\lambda_h(c_h^{(t)})\), then update (c_h).  Each
MM update does not increase the stated objective.  The implementation uses a
fixed 12 vectorized updates and the better of the one-center and full-order
initializations.

Selected parameterizations keep the initial price at 3 and the floor at 1.5:

- conservative: \(\lambda_0=1.5,\beta=0.25,\alpha=0.375\);
- stronger compression: \(\lambda_0=1.5,\beta=0.5,\alpha=0.75\).

## Validated 32K result

Qwen3-4B-Instruct-2507, RULER 32K, B=1024, temporal block 8, 30 samples per
task, no reranking.  All methods use self placement, tail-CVaR 0.25,
gap-correction 0.25, and INT8 centers.

| allocator | CWE | QA1 | QA2 | MultiKey-3 | mean centers (same order) |
|---|---:|---:|---:|---:|---|
| fixed price 1.5 | 79.68 | 58.60 | 58.64 | 100.00 | 1.994, 1.644, 1.680, 1.635 |
| smooth, beta=0.5 | 78.89 | 58.60 | 58.64 | 100.00 | 1.545, 1.284, 1.310, 1.336 |
| smooth, beta=0.25 | 79.56 | 58.60 | 58.64 | 100.00 | 1.682, 1.362, 1.396, 1.402 |

The beta=0.5 row reduces mean centers by 21.2% over the four tasks for a
0.20-point mean-score change.  The conservative beta=0.25 row reduces mean
centers by 16.0% while changing the four-task mean by only -0.03 point.  Its
CWE change is -0.12 point and the other three task scores match exactly.

Pure log complexity without the linear floor is rejected.  At a comparable
CWE mean it spends too many upgrades at orders 7--8: the alpha=4 pure-log run
used 71,300 eight-center blocks, whereas the beta=0.5 floor run used only 580.
The floor reallocates capacity toward broadly useful second centers.

## Result roots

- fixed price: `/storage/baonn/free_rd_validate30_20260910`
- beta=0.5 validation: `/storage/nbao/hfrd_l15_h3_b05_validate30_20260910`
- beta=0.25 validation: `/storage/baonn/hfrd_l15_h3_b025_validate30_20260910`
- 13-task beta=0.5 run: `/storage/baonn/hfrd_l15_h3_b05_full13_10_20260910`
- 128K beta=0.5 pilot (machine 2):
  `/home/nbnguyen/hfrd_l15_h3_b05_128k_10_20260910`
- 128K beta=0.25 chained pilot (machine 2):
  `/home/nbnguyen/hfrd_l15_h3_b025_128k_10_20260910`

Code commits: `f0c1c176` (log-prior allocator), `97c4f34d` (bounded marginal
price).
