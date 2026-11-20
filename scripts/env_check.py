"""Day-1 check: GPU visible? libraries installed? prints what to put in the report's hardware section."""
import json, subprocess, sys
sys.path.insert(0, ".")
from benchmarks.common import env_info
info = env_info()
print(json.dumps(info, indent=2))
try:
    print(subprocess.check_output(["nvidia-smi"], text=True))
except Exception as e:
    print("nvidia-smi not available:", e)
try:
    import torch
    print("torch.cuda.is_available():", torch.cuda.is_available())
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"GPU{i}: {p.name}, {p.total_memory/2**30:.1f} GiB, compute capability {p.major}.{p.minor}, SMs={p.multi_processor_count}")
except Exception as e:
    print("torch check failed:", e)
try:
    print("CPU:", subprocess.check_output("lscpu | grep -E 'Model name|^CPU\\(s\\)'", shell=True, text=True))
    print(subprocess.check_output("free -g | head -2", shell=True, text=True))
except Exception:
    pass
