################################################################################
#
# M51 -- Routed Multi-Anchor PQ LSE Retrieval, end to end.
#
# The probe this comes from (repro/certified_sparse/probe_routed_tangent_m51.py)
# measures one layer and one KV head on captured tensors. This runs the same
# selection rule inside the model, on every layer and head, so it can be scored
# on RULER next to Quest and ShadowKV.
#
# THE RULE
# --------
# Blocks of `leaf_size` post-RoPE keys. For each anchor u_m, the block's
# log-sum-exp mass F_G(u) = logsumexp_{k in G} (u . k) is linearised at u_m:
#
#     g_{G,m} = grad F_G(u_m) = sum_k softmax_k(u_m . k) * k     <- weighted centroid
#     a_{G,m} = F_G(u_m) - u_m . g_{G,m}
#
# A query is routed to its nearest anchor, the tangent is read for that branch
# only, and the block's score is
#
#     U_G = a_{G,m*} + q . PQ(g_{G,m*}) + eta_{G,m*,r}
#
# Blocks are then loaded in descending U until the predicted unresolved mass
# falls below 1 - coverage; loaded blocks contribute their EXACT logits, so the
# approximation decides only WHICH blocks are read, never the arithmetic on
# them.
#
# WHAT IS OFFLINE AND WHAT IS PER CONTEXT
# ---------------------------------------
#   offline, once per (layer, kv head)   anchors, calibration queries
#   per context, in prefill              tangents, PQ codebook, envelope eta
#   per decode step                      route, PQ table, scan, load, attend
#
# Anchors and calibration queries live in query space and come from a bank
# captured on OTHER prompts, so no part of the eval context is used to fit
# them. The envelope has to be per context because it is indexed per block.
#
# TWO DELIBERATE DEPARTURES FROM THE PROBE, both of which make the number worse
# rather than better, and neither of which is optional in a real model:
#
#   1. GQA. The probe scores one query at a time and reports its traffic. In a
#      real model the query heads of a GQA group share one KV cache, so the
#      loads of a group have to be unioned. Group-union traffic is reported
#      alongside per-query traffic; the union is what a deployment pays.
#   2. This class computes the selection exactly and then applies it as a mask
#      over the full logits. That yields the method's exact output, and its
#      traffic is counted honestly, but its DECODE WALL CLOCK IS NOT THE
#      METHOD'S -- it is an upper bound that still touches every key. Decode
#      cost for M51 must be read from the traffic columns, not from seconds.
#      (The same caveat applies to our unfused Quest: measured wall clock said
#      it was slower than dense while reading 13x less.)
#
################################################################################

import math

import torch

from .compat import head_dim_of


def _batched_kmeans(x, n_codes, iters=12, seed=0, init=None):
    """k-means over a batch of independent problems.

    x: [B, N, D] -> codebook [B, n_codes, D], codes [B, N] (int64)
    Deterministic: centres start from a strided slice, so a rerun of the same
    context reproduces the same codebook.

    `init` seeds the centres instead, which is what pq_mode='warm' uses: the
    offline codebook is already close, so a couple of Lloyd steps from it buy
    most of a per-context refit at a fraction of the iterations.
    """
    B, N, D = x.shape
    if init is not None:
        centres = init.clone()
    else:
        step = max(1, N // n_codes)
        centres = x[:, ::step, :][:, :n_codes, :].clone()
    if centres.shape[1] < n_codes:                       # pad by repetition
        pad = n_codes - centres.shape[1]
        centres = torch.cat([centres, centres[:, :pad]], dim=1)

    codes = torch.zeros(B, N, dtype=torch.long, device=x.device)
    for _ in range(iters):
        # [B, N, n_codes]
        d = torch.cdist(x, centres)
        codes = d.argmin(dim=-1)
        onehot = torch.zeros(B, N, n_codes, device=x.device, dtype=x.dtype)
        onehot.scatter_(2, codes.unsqueeze(-1), 1.0)
        counts = onehot.sum(dim=1)                        # [B, n_codes]
        summed = torch.einsum('bnc,bnd->bcd', onehot, x)
        nonempty = counts > 0
        new = torch.where(nonempty.unsqueeze(-1), summed / counts.clamp(min=1).unsqueeze(-1), centres)
        if torch.allclose(new, centres, rtol=0, atol=1e-7):
            centres = new
            break
        centres = new
    d = torch.cdist(x, centres)
    codes = d.argmin(dim=-1)
    return centres, codes


class RoutedPQCache:
    """M51 selection over a retained KV cache."""

    def __init__(self,
        config: object,
        batch_size: int = 1,
        max_length: int = 32*1024,
        device: str = 'cuda:0',
        dtype = torch.bfloat16,
        anchors_path: str = None,
        leaf_size: int = 8,
        n_anchors: int = 8,
        pq_subdim: int = 8,
        pq_codes: int = 64,
        coverage: float = 0.90,
        eta_quantile: float = 0.999,
        dense_layers: int = 0,
        pq_mode: str = 'offline',
        pq_warm_iters: int = 2,
        group_select: str = 'per_query',
        decode_mode: str = 'dense_mask',
        load_batch: int = 128,
        select_mode: str = 'adaptive',
        sparse_budget: int = 0,
        rank_anchors: int = 1,
        rank_reduce: str = 'max',
        ) -> None:

        assert batch_size == 1, "RoutedPQCache supports batch_size=1"
        assert anchors_path, "M51 needs --anchors_path (see repro/shadowkv/m51_fit_anchors.py)"

        self.config = config
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.dtype = dtype

        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.head_dim = head_dim_of(config)
        self.num_layers = config.num_hidden_layers

        self.leaf_size = int(leaf_size)
        self.n_anchors = int(n_anchors)
        self.pq_subdim = int(pq_subdim)
        self.pq_codes = int(pq_codes)
        self.coverage = float(coverage)
        self.epsilon = 1.0 - self.coverage
        self.eta_quantile = float(eta_quantile)
        self.dense_layers = int(dense_layers)

        # pq_mode:  'per_context' refits the codebook on every prompt, as the
        #           probe does. 'offline' loads a codebook fitted once on
        #           calibration documents and only ENCODES per context, which
        #           removes ~90% of the per-context metadata cost.
        # group_select: 'per_query' lets each query head of a GQA group pick
        #           its own blocks -- faithful to the probe, but the group
        #           shares one KV cache so a deployment pays their union.
        #           'shared' ranks once per KV head on the group-reduced score
        #           and lets each group stop at its own depth along that one
        #           order, so the union is the deepest prefix rather than a
        #           union of different sets. Quest and ShadowKV both reduce
        #           over the group before selecting; this matches them.
        # decode_mode: 'gather' reads only the selected blocks. 'dense_mask'
        #           computes every logit and masks -- exact same output, used
        #           as the reference the gather path is tested against.
        # select_mode: 'adaptive' spends a per-query depth decided by the
        #           coverage rule. Measured cost: the rule's depth is set by the
        #           envelope's slack rather than by coverage (2.7-7.8x an oracle
        #           at the same target), and per-query selection makes a GQA
        #           group pay the union of four different sets. 'fixed' ranks
        #           once per KV head on the group-reduced bound and takes a
        #           fixed top-k, exactly as Quest and ShadowKV do -- so the
        #           budget IS the traffic, and decode is one score, one top-k,
        #           one gather with no interleaved rounds.
        # rank_anchors: EVERY anchor gives a valid bound on a block's mass, so
        #           the minimum over anchors is tighter than whichever one the
        #           nearest-L2 route lands on. 1 = the routed branch only (the
        #           cheap scan); 0 = all M. Measured at 8K over six RULER tasks,
        #           all-anchor selection cuts the attention-output error by ~8%
        #           at every budget (_bench/m51_variants2.jsonl).
        # rank_reduce: 'max' compares raw bounds across the GQA group, which
        #           lets a head with a flatter distribution dominate. 'softmax'
        #           normalises each query head over blocks first -- ShadowKV's
        #           reduction -- and is worth another ~6%.
        assert rank_reduce in ('max', 'softmax'), rank_reduce
        assert select_mode in ('adaptive', 'fixed'), select_mode
        # pq_mode 'warm' is 'offline' plus a few Lloyd steps on THIS context's
        # tangents. Measured at 8K: refitting from scratch ('per_context') cuts
        # the attention-output error by 13% but costs 3.8x the prefill
        # (4.3 -> 16.2 s), which is 12x Quest. Warm-starting is the way to keep
        # the accuracy without the bill.
        assert pq_mode in ('per_context', 'offline', 'warm'), pq_mode
        assert group_select in ('per_query', 'shared'), group_select
        assert decode_mode in ('gather', 'dense_mask'), decode_mode
        self.pq_mode = pq_mode
        self.pq_warm_iters = int(pq_warm_iters)
        self.group_select = group_select
        self.decode_mode = decode_mode
        self.select_mode = select_mode
        self.sparse_budget = int(sparse_budget)
        self.rank_anchors = int(rank_anchors)
        self.rank_reduce = rank_reduce
        if self.select_mode == 'fixed':
            assert self.sparse_budget > 0, "select_mode='fixed' needs --sparse_budget"
        self.load_batch = int(load_batch)
        # codes are uint8. Past 256 codewords the index wraps and the scan
        # silently reads the wrong codebook row -- measured as RULER 0.00 on
        # niah_multikey_3, which looks like a bad method rather than a bad cast.
        assert 2 <= self.pq_codes <= 256, \
            f"pq_codes={self.pq_codes}: codes are uint8, so 256 is the ceiling"
        assert self.head_dim % self.pq_subdim == 0
        self.n_sub = self.head_dim // self.pq_subdim
        self.scale = 1.0 / math.sqrt(self.head_dim)

        shape = (self.num_layers, batch_size, self.num_key_value_heads, max_length, self.head_dim)
        self.k_cache = torch.zeros(shape, device=device, dtype=dtype)
        self.v_cache = torch.zeros(shape, device=device, dtype=dtype)

        self._load_offline(anchors_path)

        self.codes = self.codebook = self.offsets = self.eta = None
        self.kv_offset = 0
        self.prefill = 0
        self.gen_offset = 0
        self.n_leaves = 0
        self.tail = 0
        self.copy_stream = torch.cuda.Stream()

        # traffic accounting, the number that actually matters for M51
        # set to a list to collect tangent samples during prefill; used only by
        # repro/shadowkv/m51_fit_codebook.py to build an offline codebook
        self.tangent_sink = None
        self.tangent_sample = 2048

        self.loaded_blocks = 0        # per (query head) sum
        self.union_blocks = 0         # per KV head after the GQA union
        self.total_blocks = 0
        self.n_queries = 0
        self.prefill_meta_s = 0.0

    def _load_offline(self, path):
        import numpy as np
        z = np.load(path)
        anchors = torch.from_numpy(z["anchors"])          # [L, H, M, D], query-scaled
        calib = torch.from_numpy(z["calib"])              # [L, H, G, N, D], query-scaled
        assert anchors.shape[0] == self.num_layers, \
            f"anchor bank has {anchors.shape[0]} layers, model has {self.num_layers}"
        assert anchors.shape[1] == self.num_key_value_heads
        assert anchors.shape[3] == self.head_dim
        if anchors.shape[2] < self.n_anchors:
            raise ValueError(f"bank holds {anchors.shape[2]} anchors, asked for {self.n_anchors}")
        self.anchors = anchors[:, :, :self.n_anchors].to(self.device, torch.float32)
        self.calib = calib.to(self.device, torch.float32)
        self.n_calib = self.calib.shape[3]

        self.offline_codebook = None
        if self.pq_mode in ('offline', 'warm'):
            if "pq_codebook" not in z.files:
                raise ValueError(
                    f"{path} has no offline PQ codebook. Build one with "
                    f"repro/shadowkv/m51_fit_codebook.py, or pass pq_mode='per_context'.")
            cb = torch.from_numpy(z["pq_codebook"])        # [L, H, S, C, sub]
            assert cb.shape[0] == self.num_layers and cb.shape[1] == self.num_key_value_heads
            assert cb.shape[2] == self.n_sub and cb.shape[3] == self.pq_codes
            self.offline_codebook = cb.to(self.device, torch.float32)

    # -- bookkeeping ------------------------------------------------------

    def print_stats(self):
        print(f"M51 routed-PQ | leaf {self.leaf_size} | anchors {self.n_anchors} | "
              f"PQ {self.n_sub}x{self.pq_codes}(subdim {self.pq_subdim}) | "
              f"coverage {self.coverage} | eta q{self.eta_quantile} | "
              f"calib {self.n_calib}/group | dense layers {self.dense_layers} | "
              f"pq {self.pq_mode} | select {self.select_mode}"
              + (f" b{self.sparse_budget}" if self.select_mode == 'fixed'
                 else f"/{self.group_select} | decode {self.decode_mode}")
              + (f" (batch {self.load_batch})" if self.decode_mode == 'gather' else ""))

    def get_kv_len(self):
        return self.kv_offset

    def H2D(self):
        pass

    def is_dense_layer(self, layer_idx):
        return layer_idx < self.dense_layers

    def traffic(self):
        """Byte accounting, in the probe's units so the numbers line up.

        A block of KV costs leaf_size * head_dim * 4 bytes (K and V, 2 bytes
        each). Against that:
          * scan      -- one PQ branch per block: 6-bit codes + offset + eta,
                         plus the shared codebook amortised over the context;
          * resident  -- every branch, since all M are stored;
          * exact     -- the blocks actually pulled, which is the term that
                         dominates and the one the GQA union inflates.
        """
        if not self.n_queries:
            return {}
        D, G, M = self.head_dim, self.num_key_value_groups, self.n_anchors
        block_bytes = self.leaf_size * D * 4
        code_bytes = self.n_sub * 6 / 8                      # 6 bits per PQ code
        branch_bytes = code_bytes + 2 + 2                    # + offset + eta (this group)
        resident_branch_bytes = code_bytes + 2 + 2 * G       # eta is per group
        shared = self.pq_codes * D * 2 / (self.prefill * 4 * D) if self.prefill else 0.0

        per_query = self.loaded_blocks / max(1, self.n_queries * self.total_blocks)
        n_kv_queries = max(1, self.n_queries // G)
        union = self.union_blocks / max(1, n_kv_queries * self.total_blocks)
        scan = shared + branch_bytes / block_bytes
        resident = shared + M * resident_branch_bytes / block_bytes
        return dict(
            # NOTE these are what the RUNNING implementation read. dense_mask
            # stops each query at its own exact depth and so reports the
            # METHOD's traffic; gather advances in lockstep rounds and
            # overshoots, so its numbers include the batching's waste.
            per_query_block_frac=per_query,
            group_union_block_frac=union,
            summary_scan_frac=scan,
            resident_summary_frac=resident,
            total_traffic_frac=scan + union,
            prefill_meta_s=self.prefill_meta_s,
        )

    def clear(self):
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.codes = self.codebook = self.offsets = self.eta = None
        self.kv_offset = 0
        self.prefill = 0
        self.gen_offset = 0
        self.n_leaves = 0
        self.tail = 0

    # -- prefill: build the per-context metadata --------------------------

    def prefill_kv_cache(self, new_v_cache, layer_idx, key_states_roped, query=None):
        import time
        incoming = new_v_cache.shape[-2]
        self.prefill = incoming
        self.k_cache[layer_idx][:, :, :incoming].copy_(key_states_roped)
        self.v_cache[layer_idx][:, :, :incoming].copy_(new_v_cache)

        H, D, M = self.num_key_value_heads, self.head_dim, self.n_anchors
        L = incoming // self.leaf_size
        self.n_leaves = L
        self.tail = incoming - L * self.leaf_size
        self.total_blocks = L

        if L < 2:
            raise ValueError(f"prefill {incoming} gives {L} blocks; too short for M51")

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        if self.codes is None:
            # [H, n_sub, M, L]. The default scan reads EVERY anchor and wants
            # (n_sub, anchors, blocks) contiguous; storing (anchors, n_sub,
            # blocks) instead forced a permute+copy of a million codes on every
            # layer of every decode step. The routed scan pays that copy now,
            # which is the right way round -- it is no longer the default.
            self.codes = torch.zeros(self.num_layers, H, self.n_sub, M, L,
                                     device=self.device, dtype=torch.uint8)
            self.codebook = torch.zeros(self.num_layers, H, self.n_sub, self.pq_codes,
                                        self.pq_subdim, device=self.device, dtype=torch.float32)
            self.offsets = torch.zeros(self.num_layers, H, M, L, device=self.device,
                                       dtype=torch.float32)
            self.eta = torch.zeros(self.num_layers, H, M, self.num_key_value_groups, L,
                                   device=self.device, dtype=torch.float32)

        keys = key_states_roped[0, :, :L * self.leaf_size].to(torch.float32)   # [H, L*ls, D]
        leaves = keys.view(H, L, self.leaf_size, D)
        anchors = self.anchors[layer_idx]                                       # [H, M, D]

        # tangents: weighted centroid and offset, per (head, anchor, block)
        anchor_logits = torch.einsum('hmd,hltd->hmlt', anchors, leaves)
        weights = torch.softmax(anchor_logits, dim=-1)
        centers = torch.einsum('hmlt,hltd->hmld', weights, leaves)              # [H, M, L, D]
        offsets = torch.logsumexp(anchor_logits, dim=-1) \
            - torch.einsum('hmd,hmld->hml', anchors, centers)
        self.offsets[layer_idx].copy_(offsets)
        del anchor_logits, weights

        # PQ over the tangents; anchors of a head share one codebook, as in the probe.
        # offline mode only ENCODES (one cdist per subspace); per_context refits
        # the codebook, which profiling showed is ~90% of the metadata cost.
        flat = centers.reshape(H, M * L, D)
        recon = torch.empty_like(flat)
        for sub in range(self.n_sub):
            sl = slice(sub * self.pq_subdim, (sub + 1) * self.pq_subdim)
            if self.pq_mode == 'offline':
                cb = self.offline_codebook[layer_idx, :, sub]                    # [H, C, sub]
                code = torch.cdist(flat[:, :, sl], cb).argmin(dim=-1)
            elif self.pq_mode == 'warm':
                cb, code = _batched_kmeans(
                    flat[:, :, sl], self.pq_codes, iters=self.pq_warm_iters,
                    init=self.offline_codebook[layer_idx, :, sub])
            else:
                cb, code = _batched_kmeans(flat[:, :, sl], self.pq_codes)
            self.codebook[layer_idx, :, sub].copy_(cb)
            self.codes[layer_idx, :, sub].copy_(code.view(H, M, L).to(torch.uint8))
            recon[:, :, sl] = torch.gather(cb, 1, code.unsqueeze(-1).expand(-1, -1, self.pq_subdim))
        recon = recon.view(H, M, L, D)

        if self.tangent_sink is not None:
            n = min(self.tangent_sample, M * L)
            pick = torch.randperm(M * L, device=centers.device)[:n]
            self.tangent_sink.append(
                (layer_idx, centers.reshape(H, M * L, D)[:, pick].to("cpu", torch.float16)))

        del centers, flat

        # envelope: per (head, anchor, query group, block) quantile of the
        # residual, measured with the offline calibration queries against THIS
        # context's blocks. This is the 512 x T x d term.
        G, Nc = self.num_key_value_groups, self.n_calib
        calib = self.calib[layer_idx]                                           # [H, G, Nc, D]
        route = torch.cdist(calib.reshape(H, G * Nc, D),
                            anchors).argmin(dim=-1).view(H, G, Nc)              # [H, G, Nc]

        for h in range(H):
            logits = calib[h].reshape(G * Nc, D) @ keys[h].T                    # [G*Nc, L*ls]
            true_mass = torch.logsumexp(logits.view(G * Nc, L, self.leaf_size), dim=-1)
            true_mass = true_mass.view(G, Nc, L)
            del logits
            flat_q = calib[h].reshape(G * Nc, D)
            for m in range(M):
                lin = offsets[h, m].unsqueeze(0) + flat_q @ recon[h, m].T
                err = (true_mass.view(G * Nc, L) - lin).view(G, Nc, L)
                # One quantile per (anchor, group). Cells with no calibration
                # query routed to this anchor fall back to the whole group, as
                # the probe does -- an anchor no query reaches still needs a
                # finite envelope in case an eval query reaches it.
                for g in range(G):
                    sel = route[h, g] == m
                    src = err[g][sel] if bool(sel.any()) else err[g]
                    self.eta[layer_idx, h, m, g] = torch.quantile(
                        src.to(torch.float32), self.eta_quantile, dim=0)
            del true_mass

        torch.cuda.synchronize()
        self.prefill_meta_s += time.perf_counter() - t0

        if layer_idx == self.num_layers - 1:
            self.kv_offset += incoming

    # -- decode -----------------------------------------------------------

    def _upper_bounds(self, layer_idx, q):
        """U_G = a + q.PQ(g) + eta, for the routed branch only. [H, G, L]"""
        H, D, M, G = self.num_key_value_heads, self.head_dim, self.n_anchors, self.num_key_value_groups
        L = self.n_leaves
        anchors = self.anchors[layer_idx]
        route = torch.cdist(q, anchors).argmin(dim=-1)                          # [H, G]

        cb = self.codebook[layer_idx]                                           # [H, S, C, sub]
        qs = q.view(H, G, self.n_sub, self.pq_subdim)
        lut = torch.einsum('hgsp,hscp->hgsc', qs, cb)                           # [H, G, S, C]

        # Advanced indexing, not expand+gather. Expanding to [H, G, L, S, C] and
        # gathering makes every output element a random read into a 33M-element
        # stride-0 view; picking the routed branch by index touches only what it
        # needs and is the difference between ~190ms and a few ms per step.
        hh = torch.arange(H, device=q.device).unsqueeze(1)                      # [H, 1]
        gg = torch.arange(G, device=q.device).unsqueeze(0)                      # [1, G]

        codes = self.codes[layer_idx].transpose(1, 2)[hh, route]                # [H, G, S, L]
        flat_lut = lut.reshape(H * G * self.n_sub, self.pq_codes)
        flat_codes = codes.reshape(H * G * self.n_sub, L).long()
        score = torch.gather(flat_lut, 1, flat_codes).view(H, G, self.n_sub, L).sum(dim=2)

        off = self.offsets[layer_idx][hh, route]                                # [H, G, L]
        eta = self.eta[layer_idx][hh, route, gg]                                # [H, G, L]
        return off + score + eta

    def _order_and_tail(self, upper):
        """Ranking, plus logsumexp of the bound over every unloaded suffix.

        With group_select='shared' one order serves the whole GQA group, so the
        blocks a group loads are a PREFIX of a single order and their union is
        just the deepest prefix -- not a union of different sets. That is the
        difference between paying max_g k_g and paying |union|.
        """
        H, G, L = upper.shape
        if self.group_select == 'shared':
            order = upper.amax(dim=1).argsort(dim=-1, descending=True)          # [H, L]
            order_g = order.unsqueeze(1).expand(H, G, L)
        else:
            order = None
            order_g = upper.argsort(dim=-1, descending=True)
        upper_sorted = torch.gather(upper, -1, order_g)
        rev = torch.flip(torch.logcumsumexp(torch.flip(upper_sorted, [-1]), dim=-1), [-1])
        return order, order_g, rev, upper_sorted

    def _stop_depth(self, retained, rev, k):
        """gauge after k loads = sigmoid(lse(bound over unloaded) - retained)."""
        L = rev.shape[-1]
        unresolved = rev[..., k] if k < L else torch.full_like(retained, float('-inf'))
        return torch.sigmoid(unresolved - retained) <= self.epsilon

    @torch.inference_mode()
    def decode_attend(self, layer_idx, query_states):
        """Select blocks and attend, in one pass.

        Selection and attention are interleaved on purpose: the stopping rule
        needs the EXACT mass of the blocks loaded so far, so the loads have to
        happen while the rule is still deciding. Computing every logit first
        and masking afterwards gives the same answer but reads the whole cache,
        which is what the earlier reference path did.
        """
        H, G, D = self.num_key_value_heads, self.num_key_value_groups, self.head_dim
        L, ls = self.n_leaves, self.leaf_size
        end = self._end(layer_idx)
        q = query_states.view(H, G, D).to(torch.float32) * self.scale

        if self.is_dense_layer(layer_idx):
            keys = self.k_cache[layer_idx][0, :, :end]
            p = torch.softmax(torch.einsum('hgd,hnd->hgn', q, keys.to(torch.float32)), dim=-1)
            values = self.v_cache[layer_idx][0, :, :end].to(torch.float32)
            return torch.einsum('hgn,hnd->hgd', p, values).view(1, 1, H * G, D).to(self.dtype)

        if self.select_mode == 'fixed':
            return self._fixed_attend(layer_idx, q, end)

        upper = self._upper_bounds(layer_idx, q)
        order, order_g, rev, upper_sorted = self._order_and_tail(upper)
        # VIEW, not a copy: casting the whole cache to fp32 here would read every
        # key on every step, which is exactly the cost the gather path exists to
        # avoid. Only the gathered slice is promoted.
        kcache = self.k_cache[layer_idx][0, :, :L * ls]                          # [H, L*ls, D]

        if self.decode_mode == 'dense_mask':
            logits = torch.einsum('hgd,hnd->hgn', q, kcache.to(torch.float32))
            true_mass = torch.logsumexp(logits.view(H, G, L, ls), dim=-1)
            mass_sorted = torch.gather(true_mass, -1, order_g)
            retained = torch.logcumsumexp(mass_sorted, dim=-1)
            unresolved = torch.cat([rev[..., 1:], torch.full_like(rev[..., :1], float('-inf'))], -1)
            ok = torch.sigmoid(unresolved - retained) <= self.epsilon
            first = torch.where(ok.any(-1), ok.float().argmax(-1),
                                torch.full_like(ok[..., 0].long(), L - 1))
            k_g = (first + 1).clamp(min=1, max=L)                                # [H, G]
            k_head = k_g.amax(dim=1) if self.group_select == 'shared' else k_g
            rank = torch.empty_like(order_g)
            rank.scatter_(-1, order_g, torch.arange(L, device=order_g.device).expand_as(order_g))
            keep = k_head.unsqueeze(-1).unsqueeze(-1) if self.group_select == 'shared' \
                else k_head.unsqueeze(-1)
            loaded = rank < keep
            block_logits = logits.masked_fill(
                ~loaded.unsqueeze(-1).expand(H, G, L, ls).reshape(H, G, L * ls), float('-inf'))
            tok = torch.arange(L * ls, device=q.device).unsqueeze(0).expand(H, L * ls)
            # per-query = what each query head reads; union = what the shared KV
            # cache actually has to be read for, which under per_query selection
            # is the union of four different sets, not the deepest of them
            n_per_query = int(loaded.sum())
            n_union = int(loaded.any(dim=1).sum())
            per_query_tok = False
        elif self.group_select == 'per_query':
            # Per-query gather: every (head, group) walks its OWN order, so the
            # blocks it reads are the ones it selected -- 26% of the cache at 8K
            # rather than the 62% union a shared cache would have to serve. The
            # union is still reported, because that is what a real kernel pays.
            arange_ls = torch.arange(ls, device=q.device)
            bound_retained = torch.logcumsumexp(upper_sorted, dim=-1)
            bound_unres = torch.cat(
                [rev[..., 1:], torch.full_like(rev[..., :1], float('-inf'))], -1)
            pred_ok = torch.sigmoid(bound_unres - bound_retained) <= self.epsilon
            pred = torch.where(pred_ok.any(-1), pred_ok.float().argmax(-1) + 1,
                               torch.full_like(pred_ok[..., 0].long(), L))
            # Rounds are lockstep, so a first batch sized by the DEEPEST group
            # drags every group down to that depth: measured 42.2% of blocks
            # against the 25.0% the selection actually asks for. Size it by the
            # median instead and let the deep groups take another round.
            first_batch = int(pred.float().median().clamp(min=1, max=L))

            k, chunks = 0, []
            retained = torch.full((H, G), float('-inf'), device=q.device)
            done = torch.zeros(H, G, dtype=torch.bool, device=q.device)
            k_g = torch.full((H, G), L, dtype=torch.long, device=q.device)
            kexp = kcache.unsqueeze(1).expand(H, G, L * ls, D)
            while k < L and not bool(done.all()):
                B = min(first_batch if k == 0 else self.load_batch, L - k)
                blk = order_g[:, :, k:k + B]                                     # [H, G, B]
                tok_b = (blk.unsqueeze(-1) * ls + arange_ls).view(H, G, B * ls)
                kk = torch.gather(kexp, 2, tok_b.unsqueeze(-1).expand(H, G, B * ls, D))
                lg = torch.einsum('hgd,hgnd->hgn', q, kk.to(torch.float32))
                chunks.append(lg)
                mass = torch.logsumexp(lg.view(H, G, B, ls), dim=-1)
                retained = torch.logaddexp(retained, torch.logsumexp(mass, dim=-1))
                k += B
                newly = self._stop_depth(retained, rev, k) & ~done
                k_g = torch.where(newly, torch.full_like(k_g, k), k_g)
                done = done | newly
            k_g = torch.where(done, k_g, torch.full_like(k_g, k))
            block_logits = torch.cat(chunks, dim=-1)                             # [H, G, k*ls]
            blocks_seen = block_logits.shape[-1] // ls
            blk_idx = torch.arange(blocks_seen, device=q.device).view(1, 1, blocks_seen)
            keep = blk_idx < k_g.unsqueeze(-1)                                   # [H, G, seen]
            block_logits = block_logits.masked_fill(
                ~keep.unsqueeze(-1).expand(H, G, blocks_seen, ls).reshape(H, G, blocks_seen * ls),
                float('-inf'))
            tok = (order_g[:, :, :blocks_seen].unsqueeze(-1) * ls + arange_ls).view(
                H, G, blocks_seen * ls)
            # union across the group, which is the read a shared KV cache pays
            sel = torch.zeros(H, G, L, dtype=torch.bool, device=q.device)
            sel.scatter_(2, order_g[:, :, :blocks_seen], keep)
            n_per_query = int(sel.sum())
            n_union = int(sel.any(dim=1).sum())
            per_query_tok = True

        else:
            # rounds over one shared order: every group reads the same prefix
            assert self.group_select == 'shared', "gather decode needs an order policy"
            arange_ls = torch.arange(ls, device=q.device)
            # First round is sized by a free prediction rather than a fixed
            # batch. Using the bound in place of the (not yet known) retained
            # mass gives a depth estimate from a prefix scan we already have,
            # so a query that needs 20% of blocks does not crawl there in eight
            # rounds of 128. Later rounds fall back to load_batch, so an
            # optimistic prediction costs extra rounds, never extra traffic.
            bound_retained = torch.logcumsumexp(upper_sorted, dim=-1)
            bound_unres = torch.cat(
                [rev[..., 1:], torch.full_like(rev[..., :1], float('-inf'))], -1)
            pred_ok = torch.sigmoid(bound_unres - bound_retained) <= self.epsilon
            pred = torch.where(pred_ok.any(-1), pred_ok.float().argmax(-1) + 1,
                               torch.full_like(pred_ok[..., 0].long(), L))
            first_batch = int(pred.float().median().clamp(min=1, max=L))

            k, chunks = 0, []
            retained = torch.full((H, G), float('-inf'), device=q.device)
            done = torch.zeros(H, dtype=torch.bool, device=q.device)
            k_head = torch.full((H,), L, dtype=torch.long, device=q.device)
            while k < L and not bool(done.all()):
                B = min(first_batch if k == 0 else self.load_batch, L - k)
                blk = order[:, k:k + B]                                          # [H, B]
                tok_b = (blk.unsqueeze(-1) * ls + arange_ls).view(H, B * ls)
                kk = torch.gather(kcache, 1, tok_b.unsqueeze(-1).expand(H, B * ls, D))
                lg = torch.einsum('hgd,hnd->hgn', q, kk.to(torch.float32))        # [H, G, B*ls]
                chunks.append(lg)
                mass = torch.logsumexp(lg.view(H, G, B, ls), dim=-1)
                retained = torch.logaddexp(retained, torch.logsumexp(mass, dim=-1))
                k += B
                newly = self._stop_depth(retained, rev, k).all(dim=1) & ~done
                k_head = torch.where(newly, torch.full_like(k_head, k), k_head)
                done = done | newly
            k_head = torch.where(done, k_head, torch.full_like(k_head, k))
            block_logits = torch.cat(chunks, dim=-1)                             # [H, G, k*ls]
            blocks_seen = block_logits.shape[-1] // ls
            blk_idx = torch.arange(blocks_seen, device=q.device).view(1, 1, blocks_seen)
            keep = (blk_idx < k_head.view(H, 1, 1)).expand(H, G, blocks_seen)
            block_logits = block_logits.masked_fill(
                ~keep.unsqueeze(-1).expand(H, G, blocks_seen, ls).reshape(H, G, blocks_seen * ls),
                float('-inf'))
            tok = (order[:, :blocks_seen].unsqueeze(-1) * ls + arange_ls).view(H, blocks_seen * ls)
            # one shared prefix per KV head, so per-query and union coincide
            n_per_query = int(k_head.sum()) * G
            n_union = int(k_head.sum())
            per_query_tok = False

        # the trailing partial block and every generated token are always local
        if end > L * ls:
            keys_tail = self.k_cache[layer_idx][0, :, L * ls:end].to(torch.float32)  # tiny
            block_logits = torch.cat(
                [block_logits, torch.einsum('hgd,hnd->hgn', q, keys_tail)], dim=-1)
            n_tail = end - L * ls
            tail_tok = torch.arange(L * ls, end, device=q.device)
            if per_query_tok:
                tok = torch.cat([tok, tail_tok.view(1, 1, n_tail).expand(H, G, n_tail)], dim=-1)
            else:
                tok = torch.cat([tok, tail_tok.unsqueeze(0).expand(H, n_tail)], dim=-1)

        p = torch.softmax(block_logits, dim=-1)
        # gather V in the cache dtype, promote only what was gathered
        vcache = self.v_cache[layer_idx][0, :, :end]
        if per_query_tok:
            values = torch.gather(vcache.unsqueeze(1).expand(H, G, end, D), 2,
                                  tok.unsqueeze(-1).expand(H, G, tok.shape[-1], D))
            out = torch.einsum('hgn,hgnd->hgd', p, values.to(torch.float32))
        else:
            values = torch.gather(vcache, 1, tok.unsqueeze(-1).expand(H, tok.shape[-1], D))
            out = torch.einsum('hgn,hnd->hgd', p, values.to(torch.float32))

        self.loaded_blocks += n_per_query
        self.union_blocks += n_union
        self.n_queries += H * G
        return out.view(1, 1, H * G, D).to(self.dtype)

    def _all_anchor_bounds(self, layer_idx, q, n):
        """min over the n nearest anchors of a + q.PQ(g_m) + eta. [H, G, L]

        The routed bound picks one branch by L2 distance; every anchor gives a
        valid bound, so the minimum over several is tighter. Measured at 8K over
        six RULER tasks, n=2 already captures the whole gain -- all eight anchors
        are no better than the two nearest (_bench/m51_variants3.jsonl) -- so the
        scan stays 4x cheaper than it looks.
        """
        H, D, M, G = self.num_key_value_heads, self.head_dim, self.n_anchors, self.num_key_value_groups
        L, S, C = self.n_leaves, self.n_sub, self.pq_codes
        n = M if n <= 0 else min(n, M)
        cb = self.codebook[layer_idx]                                        # [H, S, C, sub]
        # Tried fp16 for this scan on the theory that it is memory bound: it was
        # 50% SLOWER (0.29 ms against 0.20 ms per layer), so the gather is not
        # bandwidth limited here and half buys nothing. A loop over
        # subquantisers moves half the bytes but costs 16 launches per layer and
        # ran 8x slower still -- at these sizes the launches dominate.
        lut = torch.einsum('hgsp,hscp->hgsc', q.view(H, G, S, self.pq_subdim), cb)
        if n == M:
            # every branch is scored, so the codes need no gather at all; cast
            # to int64 BEFORE broadcasting over the group, since .long() on an
            # expanded view materialises the expansion (33 MB per layer at 8K)
            idx = (self.codes[layer_idx].reshape(H, 1, S, M * L)
                   .long().expand(H, G, S, M * L))
            score = torch.gather(lut, 3, idx).view(H, G, S, M, L).sum(dim=2)
            off = self.offsets[layer_idx].unsqueeze(1)                       # [H, 1, M, L]
            eta = self.eta[layer_idx].permute(0, 2, 1, 3)                    # [H, G, M, L]
        else:
            hh = torch.arange(H, device=q.device).unsqueeze(1)
            gg = torch.arange(G, device=q.device).view(1, G, 1).expand(H, G, n).reshape(H, G * n)
            near = torch.cdist(q, self.anchors[layer_idx]).topk(
                n, dim=-1, largest=False).indices                            # [H, G, n]
            sel = near.reshape(H, G * n)
            # Keep the anchor axis where it lands. Selecting [H, G, n, S, L] and
            # permuting S in front forces a second full copy of the codes; giving
            # the LUT the anchor axis instead costs nothing, because expanding it
            # is a stride-0 view.
            codes = self.codes[layer_idx].transpose(1, 2)[hh, sel].view(
                H, G, n, S, L).long()
            lut_n = lut.unsqueeze(2).expand(H, G, n, S, C)
            score = torch.gather(lut_n, 4, codes).sum(dim=3)                 # [H, G, n, L]
            off = self.offsets[layer_idx][hh, sel].view(H, G, n, L)
            eta = self.eta[layer_idx][hh, sel, gg].view(H, G, n, L)
        return (score + off + eta).amin(dim=2)

    def _rank_scores(self, layer_idx, q):
        """One score per (KV head, block) for the fixed-budget selection."""
        upper = (self._upper_bounds(layer_idx, q) if self.rank_anchors == 1
                 else self._all_anchor_bounds(layer_idx, q, self.rank_anchors))
        if self.rank_reduce == 'softmax':
            # softmax over blocks is monotone within a query head, so it changes
            # nothing about that head's own order -- it changes which head wins
            # the group max, by putting every head on a common scale first.
            upper = upper.sub_(upper.logsumexp(dim=-1, keepdim=True))
        return upper.amax(dim=1)

    def _fixed_attend(self, layer_idx, q, end):
        """One score, one top-k, one gather -- Quest's decode shape.

        The group reduction happens BEFORE the top-k, so a KV head has a single
        block set and the group's cost is that set, not the union of four. That
        is the whole reason Quest and ShadowKV have no GQA union tax, and the
        measurement that motivated this path says the trade is worth taking:
        per-query selection buys lower error per head but charges ~2x the
        tokens, and at equal tokens the shared selection wins.
        """
        H, G, D = self.num_key_value_heads, self.num_key_value_groups, self.head_dim
        L, ls = self.n_leaves, self.leaf_size
        k = max(1, min(L, self.sparse_budget // ls))
        sel = self._rank_scores(layer_idx, q).topk(k, dim=-1).indices        # [H, k]
        arange_ls = torch.arange(ls, device=q.device)
        tok = (sel.unsqueeze(-1) * ls + arange_ls).view(H, k * ls)
        if end > L * ls:
            tok = torch.cat([tok, torch.arange(L * ls, end, device=q.device)
                             .unsqueeze(0).expand(H, end - L * ls)], dim=-1)

        kk = torch.gather(self.k_cache[layer_idx][0, :, :end], 1,
                          tok.unsqueeze(-1).expand(H, tok.shape[-1], D))
        p = torch.softmax(torch.einsum('hgd,hnd->hgn', q, kk.to(torch.float32)), dim=-1)
        vv = torch.gather(self.v_cache[layer_idx][0, :, :end], 1,
                          tok.unsqueeze(-1).expand(H, tok.shape[-1], D))
        out = torch.einsum('hgn,hnd->hgd', p, vv.to(torch.float32))

        self.loaded_blocks += k * H * G
        self.union_blocks += k * H
        self.n_queries += H * G
        return out.view(1, 1, H * G, D).to(self.dtype)

    def update_kv_cache(self, new_k_cache, new_v_cache, layer_idx):
        incoming = new_k_cache.shape[-2]
        start = self.prefill + self.gen_offset
        self.k_cache[layer_idx][:, :, start:start + incoming].copy_(new_k_cache, non_blocking=True)
        self.v_cache[layer_idx][:, :, start:start + incoming].copy_(new_v_cache, non_blocking=True)
        if layer_idx == self.num_layers - 1:
            self.kv_offset += incoming
            self.gen_offset += incoming

    # -- attention over the selected blocks -------------------------------

    def _end(self, layer_idx):
        """Valid cache length for this layer during a decode step.

        update_kv_cache advances kv_offset only on the last layer, so every
        earlier layer has already written this step's token while the counter
        still reads one short. Same convention as ShadowKVCache.
        """
        return self.kv_offset if layer_idx == self.num_layers - 1 else self.kv_offset + 1
