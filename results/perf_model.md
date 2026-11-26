# Two-term decode model: step time = t0 + k × users

| Config | t0 (ms) | k (ms/user) | R² | effective BW (GB/s) | % of peak BW | critical batch t0/k |
|---|---|---|---|---|---|---|
| E3_vllm_fp16 | 28.6 | 0.613 | 0.998 | 218 | 68% | 47 |
| E4_vllm_awq | 10.6 | 0.681 | 0.998 | 198 | 62% | 16 |
| E7a_vllm_fp16_tp2 | 17.1 | 0.437 | 0.998 | 182 | 57% | 39 |

## AWQ vs fp16 speedup: model vs measured

| Users | Model | Measured |
|---|---|---|
| 8 | 2.09× | 2.06× |
| 16 | 1.79× | 1.76× |
| 32 | 1.49× | 1.53× |
| 64 | 1.25× | 1.22× |
| 128 | 1.10× | 1.05× |

Predicted crossover (fp16 faster than AWQ) above ≈ 266 concurrent sequences (prediction, beyond the measured range).

## Tensor parallel overhead
Ideal TP=2 fixed cost 14.3 ms; measured 17.1 ms → 2.8 ms per step ≈ 39 µs per all-reduce (72 all-reduces per step). Per-user cost 0.437 vs 0.613 ms (ideal 0.306).

## Which SLO binds?
TPOT alone would allow ≈ 117 users (fp16) and ≈ 131 users (AWQ) before TPOT > 100 ms; measured goodput peaks far earlier, so first-token latency (TTFT) is the binding constraint.

## At 128 users (fp16)
Step 107 ms; compute utilisation (MFU) ≈ 11%; reading all KV caches at peak bandwidth would take only ≈ 3.5 ms. The per-user cost is therefore not KV bandwidth but per-sequence kernel and software overhead.

## Energy per output token (J)

| Users | C3_control_nocache | C5_control_cache | E0_hf_fp16 | E3_vllm_fp16 | E4_vllm_awq | E5_vllm_fp16_prefix_cache | E6_vllm_fp16_eager | E7a_vllm_fp16_tp2 | E7b_vllm_fp16_2replicas |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.911 | 1.963 | 2.926 | 3.374 | 0.881 | 1.983 | 2.810 | 2.328 | 4.908 |
| 2 | 1.008 | 1.006 | 2.931 | 1.028 | 0.461 |  |  |  |  |
| 4 |  |  | 2.889 | 0.538 | 0.253 |  |  |  |  |
| 8 | 0.282 | 0.276 | 2.882 | 0.295 | 0.144 | 0.276 | 0.380 | 0.344 | 0.533 |
| 16 |  |  | 2.880 | 0.172 | 0.098 |  |  |  |  |
| 32 |  |  | 2.903 | 0.118 | 0.077 | 0.099 | 0.118 | 0.149 | 0.172 |
| 64 |  |  |  | 0.080 | 0.066 | 0.063 |  | 0.105 | 0.116 |
| 128 |  |  |  | 0.063 | 0.060 |  |  | 0.084 | 0.080 |
