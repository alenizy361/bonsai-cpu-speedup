Title: Ternary Bonsai-2 27B on a single-channel AVX2 laptop CPU: 3.9 → 8.3 tok/s, bit-exact (no GPU, no AVX-512, no retraining) — kernels, tiled GEMM, in-place recurrent state, MTP

Body:
Machine: i7-13620H (6P+4E, AVX2 + AVX-VNNI, AVX-512 fused off), 15.6 GB single-channel DDR5-5200 (~28 GB/s effective), no GPU. Model: PrismML Ternary Bonsai-2 27B (PQ2_0, 7.2 GB, ternary Qwen3.8-27B), uncensored community build, community fine-tuned MTP head grafted.

What changed (all measured, greedy outputs byte-identical across every step):
- The fork's fast PQ2_0 kernels are AVX-512 only → wrote AVX2/AVX-VNNI vec_dot kernels: 1.3 → 4.2 tok/s.
- Q8_K activations + a tiled PQ2_0×Q8_K GEMM with staged activations and dynamic P/E-core scheduling: batch cost per token N=16 89 → 54 ms, prompt processing 6.7 → 16 tok/s. This makes the MTP verify pass ≈ one DRAM sweep.
- Grafted MTP head (--spec-type draft-mtp, n=2): 84 % acceptance on code → 7.1–8.3 tok/s code, ~6 prose.
- llama.cpp's recurrent (GDN) state path gathers + writes back every sequence's state every step (27 % of a 16-seq decode step). New in-place op: 8 seqs 10.1 → 13.3 tok/s, 16 seqs 9.5 → 13.5. Plus a lazily zeroed state buffer (--parallel 16 + speculation went from swapping at 0.9 tok/s to 7.0).
- Server: per-request speculative.n_max was dead code in the fork; fixed with per-sequence snapshot budgets.

Dead ends, measured so you don't repeat them: PTQ1_0 (1.75 bpw) is compute-bound on AVX2 and lossy for this model (0.89 % of the "ternary" codes are +2); FR-Spec restricted draft head: exact, zero gain; PrismML's DSpark drafter (Qwen3.6-based Bonsai 27B) on Bonsai-2: 0.7 % acceptance; DFlash2 community drafter: 50 %, no gain; activation sparsity (TEAL/Prox) is structurally blocked by the block-1024 Hadamard on ffn_down; the iGPU shares the bus; the SSD tops at 5 GB/s no matter how many parallel streams.

The physics: decode = one weight sweep per token (~250 ms here), verify of N tokens ≈ max(sweep, N × 54 ms). With this MTP head the exact ceiling on this box is ~8–10; 20+ needs a real block drafter for Bonsai-2 or a second memory channel.

PRs on the PrismML fork: #206 (kernels + GEMM), #207 (in-place recurrent state), #208 (server). Write-up, patches, tools, raw numbers: https://github.com/alenizy361/bonsai-cpu-speedup
Written with AI assistance (Claude Code) under my direction; every number is from my machine.
