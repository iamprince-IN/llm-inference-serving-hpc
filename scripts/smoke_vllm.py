"""Day-1 feasibility test: does vLLM run on this GPU? (offline mode, tiny model)"""
import sys, time
from vllm import LLM, SamplingParams
model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"
quant = sys.argv[2] if len(sys.argv) > 2 else None
t = time.time()
llm = LLM(model=model, dtype="half", quantization=quant, gpu_memory_utilization=0.85, max_model_len=2048)
print(f"loaded in {time.time()-t:.0f}s")
out = llm.generate(["Explain the KV cache in one sentence."], SamplingParams(max_tokens=64, temperature=0))
print("OUTPUT:", out[0].outputs[0].text)
print("SMOKE TEST PASSED for", model, quant or "")
