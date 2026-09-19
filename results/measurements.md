# Raw measurements (i7-13620H, single-channel DDR5-5200, 2026-09-18/19)

## Batch cost, ms per token (llama-batched-bench PP, one sequence, t=6 unless noted)
| N | fork as shipped (AVX2 vec_dot) | + staged GEMM + dynamic scheduling | t=12 @ 65 W |
|---|---|---|---|
| 1 | 249 | 237 | 237 |
| 2 | 158 | 136 | — |
| 3 | 133 | 105–114 | 99 |
| 4 | 123 | 76–88 | 87 |
| 8 | 99 | 68–73 | 67 |
| 16 | 89 | 61–65 | 54 |
| 64 | 88.5 | 67–69 | — |

## Multi-sequence generation, tok/s aggregate (llama-batched-bench TG, npp 4, ntg 24, t=12)
| sequences | legacy gathered state | in-place GDN state |
|---|---|---|
| 1 | 3.44 | 3.92 |
| 8 | 10.09 | 13.34 |
| 16 | 9.52 | 13.50 |
perf at 16 sequences: memmove 18 % → 0.8 %, get_rows 8 % → 0.

## Live server, greedy, `--spec-type draft-mtp --spec-draft-n-max 2`
code 7.1–8.3 tok/s (acceptance 84 %), English prose ~6, Arabic prose 4.7–6.0 (acceptance 39–69 %, prompt dependent); prompt eval 15–16 tok/s.
`--parallel 16` + MTP: boot RSS 12.8 GB → 8.4 GB with the lazily zeroed RS buffer; single stream 0.89 → 7.0 tok/s.

## SSD (WD SN740, direct reads)
1 stream 4.5 GB/s; 2 parallel 2.5+2.5; 4 parallel 4×1.3 = 5.2 GB/s. DRAM 36.4 GB/s peak.

## Model facts
PQ2_0 code histogram (4 sampled layers): code0 33.88 %, code1 31.80 %, code2 33.43 %, code3 0.89 %. Dead rows: 3 of 2,228,224. `prism.hadamard.block_size` = 1024; 401 rotated weights incl. all 64 `ffn_down`.
