Reference load: 32 concurrent users (or the closest measured). SLO: p95 TTFT <= 1.0s

| Config | out tok/s | p95 TTFT (s) | mean TPOT (ms) | peak GPU mem (GiB) | max users within SLO | throughput x vs baseline |
|---|---|---|---|---|---|---|
| E0_hf_fp16 (u=32) | 22.9 ± 0.2 | 173.53 | 42.9 | 6.0 | 1 | 1.0x |
| E3_vllm_fp16 (u=32) | 562.2 ± 7.1 | 1.55 | 50.4 | 13.3 | 16 | 24.5x |
| E4_vllm_awq (u=32) | 857.4 ± 19.4 | 1.46 | 31.4 | 13.3 | 16 | 37.4x |
| E6_vllm_fp16_eager (u=32) | 561.0 ± 6.2 | 1.59 | 50.3 | 13.1 | 8 | 24.5x |
| E7a_vllm_fp16_tp2 (u=32) | 893.3 ± 10.8 | 0.88 | 32.6 | 12.6 | 32 | 39.0x |
| E7b_vllm_fp16_2replicas (u=32) | 766.4 ± 12.6 | 0.95 | 37.6 | 13.1 | 32 | 33.4x |
