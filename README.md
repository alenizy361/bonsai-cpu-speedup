# Bonsai-2 27B on a single-channel AVX2 laptop CPU: 3.9 → 8.3 tok/s, bit-exact

**ملخص عربي:** نموذج 27B ثلاثي (Ternary Bonsai-2, تكميم PrismML بعرض 2.125 بت) على معالج لابتوب i7-13620H بلا GPU وبلا AVX-512 وبرام أحادية القناة (28 GB/s فعلياً). بلا أي تدريب أو تغيير في الأوزان، ارتفع الفك من 3.9 إلى 8.3 توكن/ث للكود (6 للنثر) وقراءة البرومبت من 6.7 إلى 16، **بمخرجات مطابقة بايتاً بايتاً**. المكونات: كيرنلات AVX2/AVX-VNNI للتكميم PQ2_0، ضرب مصفوفات مبلّط بتفعيلات Q8_K مع جدولة ديناميكية بين أنوية P وE، تحديث حالات GDN «في المكان» داخل llama.cpp (+40% للتوليد متعدد التسلسلات لكل النماذج الهجينة)، إصلاح خادم المضاربة، ورأس MTP مطعّم. كل رقم أدناه مقاس على الجهاز نفسه في 18–19 سبتمبر 2026، مع النتائج السلبية التي لم تنجح.

## Setup (everything measured here)

| | |
|---|---|
| CPU | Intel i7-13620H — 6 P + 4 E cores, AVX2 + AVX-VNNI, **no AVX-512 / AMX** |
| Memory | 15.6 GB single-channel DDR5-5200; 36.4 GB/s peak (own stream test), ~28 GB/s effective in inference |
| Model | `Ternary-Bonsai-2-27B` (GGUF arch `qwen35` = ternary Qwen3.8-27B, 64 blocks = 48 GDN + 16 attention), PQ2_0 2.125 bpw, 7.2 GB; uncensored community build (dealignai CRACK), fine-tuned MTP head (ProCreations) grafted |
| Runtime | PrismML llama.cpp fork, branch `prism` @ 1a07bfa, CPU only, `-t 6 -tb 12`, KV q8_0, flash-attn |
| Power | OEM firmware clamps the package at 25 W; held at 65 W by a root systemd loop (package peaks 66 °C, no throttling) |

The wall: decode is one sweep over the weights per token. 7.2 GB at ~28 GB/s → ~3.9 tok/s for any single-token method. Any lossless encoding of ternary weights is ≥ 1.585 bpw (measured code entropy 99.998 % of the maximum), and this model is not even purely ternary (0.89 % of codes are +2), so ~5 tok/s is the ceiling for *every* exact single-token path on this bus. Going past it requires more than one token per sweep — and a verify pass of N tokens costs ≈ max(one DRAM sweep, N × compute-per-column).

## Results (greedy, byte-identical outputs)

| Step | decode code | decode prose | decode Arabic | prompt eval | evidence |
|---|---|---|---|---|---|
| Fork as shipped on this CPU (generic scalar PQ2_0) | 1.3 | 1.3 | 1.3 | 2 | measured |
| + AVX2/AVX-VNNI vec_dot | 4.2 | 4.3 | 4.3 | 6.7 | measured |
| + tiled GEMM PQ2_0×Q8_0 | — | — | — | 9.6 | batch N=2 442→295 ms |
| + Q8_K activations + PQ2_0×Q8_K kernels | — | — | — | 12.5 | PPL 7.0542→7.0583 (±0.58) |
| + grafted MTP head, `--spec-draft-n-max 2` | 6.76 | 5.96 | 6.01 | 11.1 | acceptance 81 / 65 / 69 % |
| + staged activations, row-tile jobs, dynamic P/E scheduling, 65 W | **7.1–8.3** | 6.0–6.3 | 4.7–6.0 (prompt-dependent) | **15–16** | batch cost N=16: 88.8→54 ms/token |
| + in-place GDN state (multi-sequence only) | — | — | — | — | 8 seqs: 10.1→13.3, 16 seqs: 9.5→13.5 tok/s aggregate |

Exactness: every configuration reproduces the same greedy tokens (md5 `cacc21fd31be` on the 120-token code prompt across kernel/GEMM/in-place/legacy paths); perplexity identical to all printed digits with `PQ2_SGEMM=0/1` and `GGML_GDN_STATE_GATHER=0/1`.

## What each piece is

1. **AVX2/AVX-VNNI `ggml_vec_dot_pq2_0_q8_0/q8_K`** — the fork's fast PQ2_0 kernels are gated on AVX-512. 2-bit codes are unpacked with `srlv`+`and`, multiplied with `dpbusd`, and the unsigned-code offset is removed with a lane-wise `sum(y)` so per-lane partials stay exact. (PR: PrismML-Eng/llama.cpp#206)
2. **Q8_K activations for PQ2_0** — one activation scale per 256 instead of per 32: per 128 weights, 4 dpbusd + sub + cvt + fmadd instead of 4×(2 dpbusd + sub + cvt + fmadd).
3. **Tiled GEMM (`tinyBLAS_PQ2K_AVX`)** — a 128-weight block is unpacked once per row tile and reused across RN token columns; activation permute + lane sums staged once per call per thread; a job is a row tile with all its column tiles (each weight row streams from DRAM exactly once), claimed dynamically from the threadpool counter in runs of 8 so P- and E-cores share the compute-bound loop. RM=1,RN=8 was measured L2-bound (128 B of staged activations per (row, col) block vs 64 B) and rejected. Float order equals the vec_dot path → bit-identical.
4. **MTP head graft** — Qwen3.8-27B's `blk.64.nextn.*` (ProCreations' fine-tuned head) grafted onto the model file; needs the community 14-line Hadamard-inverse fix (now PrismML-Eng/llama.cpp#205). The fork's `--spec-type draft-mtp` with per-token GDN state rollback does the rest.
5. **In-place GDN state update (`ggml_gated_delta_net_cache`)** — llama.cpp's recurrent path gathers every sequence's state with `get_rows` and writes it back with `cpy` on every step (27 % of a 16-sequence decode step: memmove 18 % + get_rows 8 %). The new op updates the cache row in place when the ubatch's sequences map to their own cells (`rs_inplace_ok`), with a per-sequence rollback-snapshot budget so speculative slots keep their planes; 360-case byte-exact unit test. Upstream has an open request for the single-sequence case (ggml-org/llama.cpp#28841); this covers any n_seqs. (PR: https://github.com/PrismML-Eng/llama.cpp/pull/207)
6. **Lazily zeroed recurrent buffer** — anonymous mmap instead of an eager memset, so `--parallel 16` with speculation no longer pins 4.5 GB of untouched snapshot planes (boot RSS 12.8 → 8.4 GB; single stream 0.89 → 7.0 tok/s on this 15.6 GB machine).
7. **Server speculative fixes** — the fork's per-request `speculative.n_max` was dead code (`#if 0`), so every slot drafted and equal-length batching broke; now honored, with per-sequence snapshot budgets (opt-out slots use one plane), and an honest rollback refusal instead of a stale-plane read. (PR: https://github.com/PrismML-Eng/llama.cpp/pull/208)

## Measured dead ends (so nobody repeats them)

- **PTQ1_0 (1.75 bpw)** — a bit-exact AVX2 kernel + Q8_K tiled GEMM exist, but the base-3 decode is compute-bound on AVX2 (N=1 902 ms/token with the Q8_K path, N=16 105 vs 62 ms/token for PQ2_0) and the packing is lossy for this model (0.89 % of codes are +2).
- **FR-Spec restricted draft head** (K=16 384 rows of `output.weight` + `d2t` scatter) — exact, works, no gain: acceptance fell 81/65/69 → 76/48/55 % and the draft's lm_head is only ~12 ms of a ~300 ms round.
- **E-cores in the verify batch with a static split** (`-tb 16`) — −15 %; fixed by the dynamic row-tile claiming.
- **A stock same-vocab draft** (Qwen3.5-0.8B) — 19 % acceptance. **PrismML's DSpark drafter for the Qwen3.6-based Bonsai 27B on Bonsai-2** — 0.7 % acceptance (drafters do not transfer across base generations; PrismML ships none for Bonsai-2). **ProCreations' DFlash2 for Bonsai-2** — 50 % acceptance at T=1, no gain.
- **n_max=3** — more rejected drafts than the extra column pays for on prose.
- **Parallel "sections" answers** (one answer generated as 8 concurrent sections) — aggregate throughput is real (16 seqs = 13.5 tok/s) but the fixed costs (plan + per-section prompt processing at ~16 tok/s) eat it on this dense model; +10 % on a 3B-active MoE.
- **Activation sparsity** (TEAL/Prox-style) — blocked structurally: the model's Hadamard rotation is block-diagonal with block 1024 (`prism.hadamard.block_size`) and applies to *all* 401 rotated weights including `ffn_down`, so a sparse intermediate becomes dense before `down`; only `gate/up` rows (42 % of bytes) are skippable, and re-clustering neurons into rotation blocks would break the ternary quantization.
- **iGPU / SSD** — the iGPU shares the same DDR5 bus; the NVMe tops at 5.2 GB/s total regardless of parallel streams (measured 1/2/4 streams: 4.5 / 5.0 / 5.2 GB/s).

## The law that sets the ceiling

A verify pass is one weight sweep (~250 ms) plus ~54 ms of compute per extra token column (after this work; 88 ms before). So tok/s ≈ accepted_tokens_per_round / (max(sweep, N × column) + draft cost). With the MTP head's acceptance (84 % first token on code, decaying along the chain) the exact ceiling on this machine is ~8–10 tok/s; 20+ needs a block drafter trained for Bonsai-2 (acceptance ≥ 0.85 over 6–8 positions) **and** ≤ 40 ms per column — or a second memory channel.

## Reproduce

```bash
git clone --branch prism https://github.com/PrismML-Eng/llama.cpp && cd llama.cpp && git checkout 1a07bfa
git am ../patches/*.patch                                   # kernels + GEMM (PR #206), in-place GDN state, server fixes
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF && cmake --build build -j --target llama-server
python3 tools/graft_mtp_local.py Bonsai-2-27B-PQ2_0.gguf bonsai2-mtp.gguf Ternary-Bonsai-2-27B-MTP-Q8_0.gguf   # head source: ProCreations
build/bin/llama-server -m bonsai2-mtp.gguf --no-repack -t 6 -tb 12 -c 8192 -ctk q8_0 -ctv q8_0 --flash-attn on \
  --spec-type draft-mtp --spec-draft-n-max 2 --jinja --reasoning-format deepseek --chat-template-kwargs '{"enable_thinking":false}'
tools/watch-speed.sh "Write a Python function that parses a CSV file ..."    # live tokens + tok/s + acceptance
```

Credits: PrismML (fork, PQ2_0, hybrid speculative rollback); BoldingBuilds / decent-jawfish / ProCreations (MTP graft, Hadamard-inverse fix, fine-tuned head); FR-Spec (arXiv 2502.14856); EAGLE-3 (d2t pattern). Kernels, Q8_K path, tile design, in-place recurrent state, server fixes, tooling and all measurements: Abdulaziz Alanizy (with Claude Code), 2026-09-18/19.
