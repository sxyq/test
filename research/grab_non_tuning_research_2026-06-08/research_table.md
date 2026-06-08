# GRAB Non-Tuning Research - 2026-06-08

## Scope

Current online best remains `V42-SILENT-RUNNER-sxyq`:

| score_all | latency | auc | pcoc |
|---:|---:|---:|---:|
| 62.7338 | 69.87686s | 0.75254 | 1.05913 |

V99 recovered online at `62.40069 / 71.30446s`, so this research round
explicitly avoids another no-pred/H2D/output-stack variant.

## External Anchors

| source | stable technical point | local implication |
|---|---|---|
| PyTorch `torch.repeat_interleave` documentation | The `output_size` argument can avoid stream synchronization needed to calculate output shape. | Supports the V71/V121 hypothesis that token-user expansion and shape calculation can be a real CUDA-side cost. V121 goes further by supplying `token_user_ids` as cached-batch metadata. |
| PyTorch `scaled_dot_product_attention` / `sdpa_kernel` documentation | SDPA can dispatch to different optimized kernels and `sdpa_kernel` can select backends. | Supports exact A800 gates for V90/V93/V101 instead of changing attention semantics or splitting batches. |
| PyTorch CUDA Graphs documentation | CUDA Graph replay can reduce CPU overhead but requires stable shapes, stable memory addresses, and careful warmup/capture behavior. | Supports V117/V112 -> regional graph/compile gate sequencing, not blind `torch.compile` packages. |
| FlashInfer SegmentGEMM / MoE-style grouped backend documentation | Segment/ragged GEMM backends target variable expert workloads. | Supports V119 as a real SMoE backend branch only after A800 dependency and route-histogram evidence. |
| FBGEMM_GPU / TorchRec TBE documentation | Recommendation embedding speedups are often values/offsets/layout/backend problems. | Supports sparse data-layout branches V98/V100/V102/V118 rather than more output-stack tuning. |

## Direction Table

| rank | direction | local evidence | required next evidence | package now |
|---:|---|---|---|---|
| 1 | Token-major/cache-real sparse data path: V122 has now combined V98 layout, V100 consume source, V102 shard staging, and V121 token-user metadata into one source/schema gate. | V98 `5/5` exact and full-model positive; V100 `5/5` exact; V102 `5/5` exact with staged tensor count ratio `0.2`; V121 exact; V122 `5/5` exact with ordinary V100/ordinary V122/staged V122 all zero-drift vs V99, staged tensor count `4` vs ordinary V122 `20`, storage sharing for values/offsets/user_offsets/token_user_ids, and mask exact. | Real `dataset/cached_batches + ckpt.pt + A800`; prove flat token-major values/offsets and token_user_ids are supplied outside timed forward; prove full `time_sum`, H2D, memory, and output bytes/order. | No |
| 2 | Attention backend exact path: V123 connects V121/V122 token-user metadata to V90 bool2d SDPA mask representation; V93/V101 remain separate branches. | V90 bool2d/additive exact; V93 backend-priority API exact locally; V101 single-user causal exact; V121 metadata exact; V123 `5/5` exact on V122 staged path, bool2d vs V99 and bool2d vs bool4d both zero-drift, and attention calls confirm rank-2 bool masks. | A800 kernel/backend selection, full `time_sum`, memory, real single-user/user-shape frequency, and no silent backend fallback. | No |
| 3 | SMoE backend replacement: V119 SegmentGEMM/FlashInfer/Triton/grouped-mm branch. | V119 pack/scatter contract exact, but backend missing locally; multi-agent round 121 still sees SMoE backend as high-upside. | V104 timescope proves SMoE share, V116 proves backend dependency, real route histogram is suitable, full `time_sum` positive. | No |
| 4 | TBE/FBGEMM embedding backend: V127 joins V104 RepEncoder timescope, V116 backend selector, and V118 TBE contract. | V118 shared token-major contract exact, token direct view `10/10`, int32 safe `10/10`; V127 local smoke passes with `package_allowed=false`; replicated-table `28x` is explicitly forbidden. | Real A800 V104 proves RepEncoder is a hotspot, V116 proves `fbgemm_gpu`/TorchRec/TBE backend, full `time_sum`/H2D/memory positive, clamp marker safe. | No |
| 5 | Graph/cache scheduling: V128 joins V112 output contract, V117 shape locality, and V116 compile/CUDA Graph backend selector. | V112 safe `3/4` when predicted logids unique, duplicate negative stop works, median synthetic switch reduction `0.9367`; V128 local smoke passes with `package_allowed=false`; V117 still `missing_cached_batches`. | Real shape locality, no duplicate pred logids, A800 compile/CUDA Graph backend, first-run cost amortization, full `time_sum` positive. | No |
| 6 | True calibrated low-precision expert Linear. | V67 readiness only; local torchao/backend missing; previous FP16 family has already delivered but deeper/unsafe variants risk drift. | A800 backend and calibration, quant/dequant included in timing, AUC/PCOC guard. | No |

## V124 Decision Layer

V124 adds a machine-readable decision layer over the table above:

| field | value |
|---|---|
| gate | `V124-A800-CACHE-BACKEND-DECISION-GATE` |
| local status | `local_semantic_gate_passed_real_a800_required` |
| chosen branch | `V122_CACHE_REAL_FIRST_V123_BOOL2D_SECOND` |
| package allowed | `false` |
| next | `RUN_REAL_A800_V122_CACHE_REAL_THEN_V123_BOOL2D_SDPA` |

Interpretation: local exactness is strong enough to justify a real A800 gate,
but not strong enough to justify a scoring zip. The missing proof is real
cached-batch/A800 full `time_sum`, H2D and memory non-regression, and positive
backend evidence.

## V125 Preflight Layer

V125 turns the V124 decision into one executable real-A800 preflight:

| field | value |
|---|---|
| gate | `V125-A800-CACHE-BACKEND-PREFLIGHT-GATE` |
| local status | `synthetic_smoke_passed_package_blocked` |
| subgates | `V105 suite -> V116 backend selector -> V124 decision` |
| chosen branch | `V122_CACHE_REAL_FIRST_V123_BOOL2D_SECOND` |
| package allowed | `false` |
| next | `RUN_REAL_A800_PREFLIGHT` |

Real command:

```bash
.venv_shorttest/bin/python tools/run_v125_a800_cache_backend_preflight_gate.py \
  --cached-dir dataset/cached_batches \
  --ckpt ckpt.pt \
  --device cuda:0 \
  --max-shards 1 \
  --max-batches 16 \
  --compare-batches 3 \
  --diag-batches 3 \
  --repeat 5 \
  --warmup 1
```

## V126 SMoE Backend Branch

V126 opens a separate high-upside branch for native SMoE grouped/ragged GEMM
backends:

| field | value |
|---|---|
| gate | `V126-SMOE-BACKEND-PREFLIGHT-GATE` |
| local status | `smoe_backend_contract_ready_a800_evidence_missing` |
| branch | `V119_SEGMENT_GEMM_TRITON_SMOE_BACKEND` |
| local route contract | `exact=true`, segment candidates `6` |
| missing proof | real A800 V104 timescope, grouped/Triton/FlashInfer backend, positive full warmed `time_sum` |
| package allowed | `false` |

Real command:

```bash
.venv_shorttest/bin/python tools/run_v126_smoe_backend_preflight_gate.py \
  --refresh-v104 \
  --refresh-v116 \
  --refresh-v119 \
  --cached-dir dataset/cached_batches \
  --ckpt ckpt.pt \
  --device cuda:0 \
  --max-shards 1 \
  --max-batches 16 \
  --diag-batches 3 \
  --repeat 5 \
  --warmup 1
```

## V127 TBE Embedding Backend Branch

V127 opens a conditionally useful TBE/FBGEMM embedding backend branch, but it is
not the next first-priority A800 gate.  Two reviewer agents agreed it should
remain behind V125/V122/V123 unless the A800 backend and hotspot evidence
become positive.

| field | value |
|---|---|
| gate | `V127-TBE-EMBEDDING-BACKEND-PREFLIGHT-GATE` |
| local status | `tbe_contract_ready_a800_backend_or_hotspot_missing` |
| branch | `V118_SHARED_TOKEN_MAJOR_TBE_BACKEND` |
| local contract | shared token-major exact `true`, direct view `10/10`, int32 safe `10/10` |
| forbidden path | `28x` replicated embedding table default |
| missing proof | real A800 RepEncoder hotspot, `fbgemm_gpu`/TorchRec backend, positive full warmed `time_sum` |
| package allowed | `false` |

Real command:

```bash
.venv_shorttest/bin/python tools/run_v127_tbe_embedding_backend_preflight_gate.py \
  --refresh-v104 \
  --refresh-v116 \
  --refresh-v118 \
  --cached-dir dataset/cached_batches \
  --ckpt ckpt.pt \
  --device cuda:0 \
  --max-shards 1 \
  --max-batches 16 \
  --diag-batches 3 \
  --repeat 5 \
  --warmup 1
```

## V128 Graph/Cache Scheduling Branch

V128 opens a conditionally useful system scheduling branch.  It should stay
behind V125/V122/V123 unless real cached-batch shape locality and A800
compile/CUDA Graph evidence become positive.

| field | value |
|---|---|
| gate | `V128-GRAPH-CACHE-SCHEDULING-PREFLIGHT-GATE` |
| local status | `shape_order_contract_ready_real_a800_shape_backend_missing` |
| branch | `V112_V117_GRAPH_CACHE_SCHEDULING` |
| local contract | V112 safe `3/4`, duplicate-logid negative stop `true`, median switch reduction `0.9367` |
| missing proof | real cached-batch shape reuse, no duplicate pred logids, real A800 compile/CUDA Graph backend |
| package allowed | `false` |

Real command:

```bash
.venv_shorttest/bin/python tools/run_v128_graph_cache_scheduling_preflight_gate.py \
  --refresh-v112 \
  --refresh-v117 \
  --refresh-v116 \
  --cached-dir dataset/cached_batches \
  --device cuda:0 \
  --max-shards 8 \
  --max-batches 256 \
  --repeat 5 \
  --warmup 1
```

## Stop Rules

- Stop any direction that uses infer-time sampling or truncation.
- Stop if an optimization changes output rows, output order, Top-2 routing,
  attention semantics, or produces any drift not already accepted by a gate.
- Stop if the required real inputs are missing and the decision would be a
  scoring package rather than a gate.
- Stop if a backend dependency requires heavy online installation or falls back
  silently.
- Stop if full `time_sum` or full-script wall-time is not positive on A800.
- Stop if peak/reserved CUDA memory regresses materially.

## Immediate Recommendation

Do not make another local-only scoring zip from synthetic evidence.  The next
serious scoring candidate should be unlocked by a real A800 cache-real gate:

```bash
python tools/run_v105_next_a800_gate_suite.py \
  --cached-dir dataset/cached_batches \
  --ckpt ckpt.pt \
  --device cuda:0 \
  --max-shards 1 \
  --max-batches 16 \
  --compare-batches 3 \
  --diag-batches 3 \
  --repeat 5 \
  --warmup 1
```

Local `V122-TOKEN-CACHE-REAL-GATE` is now complete and remains non-package
evidence.  Further local-only work should either improve the real-A800 gate
instrumentation for V122/cache-real, or open a separate exact branch such as
attention backend or SMoE backend contracts without mixing it into the sparse
data-path scoring candidate.
