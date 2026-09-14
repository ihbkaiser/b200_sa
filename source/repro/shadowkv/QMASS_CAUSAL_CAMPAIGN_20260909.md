# Causal prompt-query-mass campaign (2026-09-09/10)

## Question

Can the adaptive block-8 router place and allocate its `r=1..8` centroids
from prompt queries, without observing future decode queries, and thereby close
the hard-task gap to ParisKV at the same retained-token budget?

This note separates complete end-to-end results, short screening prefixes, and
offline diagnostics. Scores from different rows are not compared unless model,
context length, token budget, sample prefix, GQA reduction, and reranking mode
match.

## Default evaluated method

- Model: Qwen3-4B-Instruct-2507.
- Temporal block size: 8 tokens.
- Retained-token budget: B=512 at 32K and B=4096 at 128K.
- Exact regions: first 32 and most recent 32 tokens, added outside B.
- Adaptive centroid count: `r in [1,8]`, mean `r=1.5` over candidate blocks.
- Center storage: INT8 compact metadata; exact KV backing is CPU pinned.
- Routing: one shot, no token rerank unless explicitly named.
- GQA: mean query before routing.
- Placement/allocation: a causal bank of the final four prompt queries;
  `global_mass` minimizes missed exact prompt-query block mass along each
  nested centroid path, then concave marginal allocation spends the global
  centroid budget.

## Cluster resource rule

- Never schedule four concurrent 128K jobs on machine 1: its host RAM is not
  sufficient for four CPU-offloaded exact-KV caches.
- Use machine 1 GPU 0--3 concurrently for 32K screens.  Prefer machines 2 and
  4 for 128K; if machine 1 is unavoidable, run at most one 128K job and verify
  host-RAM headroom first.
- Code moves between machines through Git.  Models, datasets, environments,
  and large result artifacts are installed or downloaded independently.

## End-to-end results

### Qwen3-4B-Instruct-2507, RULER 128K, B=4096

The query-mass run completed on machine 4. All rows retain B=4096 tokens, but
the ParisKV row uses its 2B candidate stage whereas the first two rows are
one-shot; it is therefore a quality target, not a one-stage cost match.

| Method | CWE | MultiKey-3 | Samples | Notes |
|---|---:|---:|---:|---|
| self-K, r=1.5 | 17.0 | 60.0 | 100/task | one shot |
| causal query-mass, bank 4, r=1.5 | 31.0 | 56.67 | 30/task | one shot |
| ParisKV | 40.0 | 84.0 | CWE 10; MK3 100 | B=4096, candidate pool 2B |

On the exactly matched first ten CWE samples, query mass scores 38.0 and
ParisKV 40.0. Thus query mass closes most of the CWE prefix gap, but it loses
3.33 points to self-K and 27.33 points to ParisKV on MultiKey-3.

Result roots:

- query-mass samples 0--4: `/storage/nbao/qmass_recent4_128k_20260909`
- query-mass samples 5--29: `/storage/nbao/qmass_recent4_128k_expand_20260909`
- self-K r=1.5: `/home/nbnguyen/current_selfk_128k_r15_b4096_20260909`
- Paris CWE: `/storage/baonn/center_campaign_20260907/e2e128_m1/cc_m1_paris_cwe_b4096`

### Qwen3-4B-Instruct-2507, CWE 32K, B=512

The full 30-sample causal-query result already available for the one-query
variant is 75.67 one shot and 80.67 with a 2B token rerank. The four-query bank
was screened on the same first five samples before the 128K expansion.

| Placement/objective | GQA semantics | Score | Samples |
|---|---|---:|---:|
| causal query mass, final query, global mass | mean | 72.0 | first 5 |
| causal query mass, bank 4, global mass | all query heads | 74.0 | first 5 |
| bank 4, worst 6.25% query losses | all query heads | 72.0 | first 5 |
| bank 4, worst 25% query losses | mean | 74.0 | first 5 |
| bank 4, top-k hinge | all query heads | 60.0 | first 5 |
| bank 4, worst-query top-k hinge | all query heads | 60.0 | first 5 |

The tail-risk and top-k objectives therefore do not beat ordinary global mass
on this controlled prefix. The worst-query top-k variant also scores 80.0 on
MultiKey-3 versus 100.0 for ordinary global mass on the same first five
samples. They are not promoted to a full campaign.

### Llama-3.2-3B-Instruct cross-model screen

At the same 32K/B=512/block-8/r=1.5 one-shot setting, query-mass bank 4
scores 8.0 on CWE and 20.0 on MultiKey-3 over the first five samples. CWE is
not a useful discriminator for this model (the previously completed full-
attention 100-sample score is only 0.4), while MultiKey-3 full attention is
52.0. The matched self-K control scores 0.0 and 20.0, respectively. Query mass
therefore provides no MultiKey-3 improvement on this second model. It is also
materially slower in this screen: about 41 seconds/sample versus 16--18 for
self-K, because fitting paths against prompt-query banks dominates generation.

## Causal failure audit

RULER MultiKey-3 sample 1 at 128K is a causal counterexample. The baseline
predicts one UUID character incorrectly. Forcing only the five known evidence
blocks (2996--3000) into the same B=4096 budget changes the answer to the exact
target, so the failure is selection/ranking rather than model incapacity.

Under the deployed causal query-mass router, several retrieval heads have high
mean evidence-block recall but a catastrophic minimum over the 38 decode
steps: layer/head 24/1 has mean 94.7% and minimum 0%; 29/1 has mean 91.1% and
minimum 0%; 30/6 has mean 90.5% and minimum 20%. A single transient miss is
enough to corrupt exact-string generation.

The miss is not caused by starving these evidence blocks of centroids. Under
the global mean quota `r=1.5`, their mean component counts at heads 24/1,
24/5, 29/1, and 30/6 are 5.8, 5.8, 3.8, and 5.2; the minimum over the five
evidence blocks is 4, 3, 1, and 3. Thus the allocator already concentrates
most of its available resolution on the relevant region. Several nearly exact
block representations are nevertheless rejected late in decoding. This
separates representation/allocation error from ranking-objective error.

Artifact root:
`/storage/baonn/qmass_failure_audit_20260909/mk3_s1_bank4`.

Component-allocation audit:
`/storage/baonn/qmass_failure_audit_20260909/mk3_s1_bank4_rankaudit`.

The intervention replicates on sample 8: qmass produces an incorrect UUID,
while forcing only the literal evidence blocks at the same B=4096 recovers the
exact answer.  The same four heads reach zero evidence recall, even though the
evidence blocks receive 3.2--4.4 components on average.  This second example
confirms that the sample-1 diagnosis is not a one-off decode accident.

Second audit:
`/storage/baonn/qmass_failure_audit_20260909/mk3_s8_bank4_rankaudit`.

### Does observing the future query solve placement?

No. Re-fitting the same r=1.5 paths using all 38 future decode queries increases
top-set overlap with exact block-mass ranking, but does not remove the evidence
recall collapse. For example:

| Layer/head | Causal bank overlap | Oracle-future overlap | Causal evidence min | Oracle-future evidence min |
|---|---:|---:|---:|---:|
| 24/1 | 81.1 | 83.2 | 0 | 0 |
| 24/5 | 66.9 | 73.1 | 40 | 40 |
| 29/1 | 80.9 | 86.6 | 0 | 20 |
| 30/6 | 77.0 | 85.8 | 20 | 20 |

Thus future-query drift is real but is not the dominant remaining bottleneck.
Even oracle future-query placement optimizes aggregate mass/overlap rather than
the worst-step survival of every evidence block. Exact block-total-mass ranking
itself also omits evidence at some steps; its mean evidence recall over these
heads is only 84.7--93.7%. This is an objective mismatch, not merely a causal
query-estimation error.

A second oracle test replaced average top-k hinge by the worst 6.25% of
per-query hinge violations. It produced essentially the same selected paths
and retained the same zero-recall steps, even with all future queries. The
failure is therefore not repaired by changing the average over queries to a
short-tail minimax surrogate at the same r=1.5 budget.

Nor is GQA averaging the sole cause. On the same trace, an exact oracle that
ranks blocks by the maximum normalized mass over the four GQA query heads
raises the minimum recall from 0 to 20% at heads 24/1 and 29/1, but head 30/6
still reaches zero. Ranking by the largest individual token logit also leaves
zero-recall steps. This is a small aggregation effect, not the missing 27-point
MultiKey-3 explanation.

GQA/peak diagnostic:
`/storage/baonn/qmass_failure_audit_20260909/mk3_s1_bank4/gqa_oracle_diagnostic.json`.

### What ParisKV preserves on the same failures

ParisKV produces the exact target on both MultiKey-3 samples 1 and 8, which
the causal query-mass run misses.  Its network-wide mean answer-token recall
is only 12.1% and 10.4%, respectively, so retaining literal evidence in every
head is neither necessary nor an appropriate diagnostic.  At the retrieval
heads implicated by the causal intervention, however, ParisKV's selection is
substantially more stable.  On sample 1 its mean/minimum *evidence-block*
recalls over decoding are 96.8/40% (24/1), 99.5/80% (24/5), 97.4/60% (29/1),
and 94.7/20% (30/6).  None reaches zero.  The corresponding query-mass minima
are 0%, 40%, 0%, and 20%.

This supports a more precise diagnosis: Paris does not preserve all answer
tokens everywhere; its token-level collision selector preserves enough of
them consistently at the retrieval heads.  Query mass ranks whole block-8
units by bulk probability, and a block can disappear during one late decode
step even when its time-averaged recall is high.  A later token rerank cannot
recover a block that the coarse shortlist never admitted.

Paris audit roots:

- `/storage/baonn/qmass_failure_audit_20260909/pariskv_mk3_s1`
- `/storage/baonn/qmass_failure_audit_20260909/pariskv_mk3_s8`

### Does a component-diverse 2B shortlist repair dilution?

No on the two causal counterexamples.  A diagnostic ranked each packed
centroid separately, mapped the top responses back to distinct parent blocks,
and only then performed the same exact-token rerank over a 2B candidate pool.
This gives a singleton centroid a route into the shortlist without first
adding the other components of its block.  Both a peak score (population
log-count removed) and a mass score were tested.  Sample 1 remains wrong under
both, and sample 8 remains wrong under the peak variant.  The peak variant
also reproduces the same erroneous strings as ordinary qmass on both samples.
On the first five 32K CWE examples it scores 70.0 despite the 2B rerank, versus
74.0 for ordinary bank-4 qmass without this candidate rule. Thus
pre-shortlist component dilution is not the missing mechanism.

Artifacts:

- `/storage/baonn/qmass_component_shortlist_128k_20260909`
- `/storage/baonn/qmass_component_shortlist_mass_128k_20260909`

### Does temporal persistence repair the rare late-step miss?

No.  A causal reservoir reserved either 12.5% or 25% of the fixed B=4096
budget for blocks most frequently selected during the preceding 16 decode
steps, then filled the remaining slots from the current query ranking.  An
offline replay showed that this policy substantially raises the minimum
literal-evidence recall at the four audited retrieval heads (on sample 1,
the 25% reservoir raises three of four minima to 100% and the fourth to 80%).
Nevertheless, end-to-end generation remains wrong on all five tested
MultiKey-3 failures (samples 1, 8, 9, and 11 at 12.5%, plus sample 1 at 25%).
The 25% run reproduces exactly the same one-character error as the ordinary
router.  Two 32K CWE controls are also unchanged (0.7 and 0.8).

Thus persistence improves a hand-selected recall diagnostic without improving
the causal outcome.  It is not promoted to the method.  This is also a warning
against using the minimum recall of four post-hoc heads as an optimization
target: it is causally useful under forced-block intervention, but it is not a
sufficient statistic for successful generation.

Artifacts:

- `/storage/baonn/qmass_temporal_light_20260910`
- `/storage/baonn/qmass_failure_audit_20260909/mk3_s1_bank4/temporal_reservoir_grid.json`
- `/storage/baonn/qmass_failure_audit_20260909/mk3_s8_bank4_rankaudit/temporal_reservoir_grid.json`

### Can a calibrated prompt-query/self-K mixture recover both task types?

Not on the controlled 32K screen.  We replaced a fraction of the proxy law by
a query-free law that chooses a block uniformly and then one normalized key
inside that block.  Both terms are fractions of Jensen mass lost, so this is
a dimensionally calibrated mixture rather than a heuristic score sum.  At
mean r=1.5 and B=512, a 25% self-K mixture scores 74.0 on CWE and 100.0 on
MultiKey-3 over the first five samples—exactly the ordinary bank-4 query-mass
screen.  A 50% mixture already degrades CWE to 66.67 on the first three
samples (ordinary query mass gives 70.0 on that prefix), while leaving the
first five MultiKey-3 samples unchanged at 100.  The option was therefore
removed from production code rather than expanded to 128K.

Artifact: `/storage/baonn/qmass_selfmix_32k_screen_20260910`.

### Is centroid approximation still the 32K ceiling?

No on the same five-sample screen.  Materializing all eight singleton
components per block in FP16 makes the adaptive score exactly the block-8
log-sum-exp under the deployed mean-GQA query.  This exact representation
scores 72.0 on CWE and 100.0 on MultiKey-3 at B=512.  The r=1.5 query-mass
bank scores 74.0/100.0 on the identical prefix.  More centers therefore do
not improve this controlled outcome; approximation error is already below
the selector/budget noise floor.  An approximate router can occasionally
outscore its exact target because its ranking error acts as an accidental
regularizer, but that is not a reproducible reason to add centers.

Artifact: `/storage/baonn/qmass_exact_blockmass_32k_20260910`.

At 128K the exact streaming control separates two failure mechanisms.  Exact
block-total-mass recovers the correct UUID on MultiKey-3 sample 1, so this
sample still contains a representation/ranking gap.  The same oracle remains
wrong on sample 8, proving that no better approximation to block total mass
can repair that counterexample at the same block budget.  The latter requires
a different selection statistic or finer selection unit, not more centroids.

Artifact:
`/storage/nbao/exact_blockmass_counterexamples_128k_offload2_20260910`.

Generalization artifact:
`/storage/baonn/qmass_failure_audit_20260909/mk3_s1_bank4/generalization.json`.

## Current conclusion

Prompt-query mass is a useful CWE-biased signal, but it is not a robust default
for exact multi-key retrieval. Four causal queries, worst-query CVaR, GQA-all,
and top-k hinge have not removed the failure. The
failure is temporally sparse: mean retained mass can be high while one required
evidence block disappears for one decode step. Any next objective should be
judged directly by per-step evidence survival (or a deployable proxy for it),
not by average mass or average top-set overlap alone.

## Pending cross-checks

- finish the fair 2B-rerank Qwen 30-sample 128K run queued on machine 4;
- exact block-total-mass end-to-end control on MultiKey-3 sample 1 (the
  first 24-GB attempt did not fit because this offline control materializes
  the full 128K GPU cache; retry on a 48-GB worker);
- collect the Llama-3.2-3B-Instruct five-sample 128K screen from machine 2;
- record host/GPU cost, since qmass preprocessing is materially slower than
  self-K and ParisKV routing.
