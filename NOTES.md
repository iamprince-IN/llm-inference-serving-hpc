# Lab notebook
Write an entry EVERY time you run something. This becomes the "Approaches attempted" and
"Challenges" sections of the report, so record failures too. newer huggingface_hub requires dataset ids as namespace/name (wikitext -> Salesforce/wikitext); vLLM 0.34 enables prefix caching by default, so it is disabled in the main configs and measured separately as E5.
Show me the diff before saving.

## Template
### YYYY-MM-DD HH:MM  <what>
- Platform / GPU:
- Command / config:
- Result (key numbers):
- Problems + how solved:
- Observation / why I think it happened:

---

## Hardware (paste scripts/env_check.py output)
{
  "python": "3.12.3",
  "platform": "Linux-6.18.55+-x86_64-with-glibc2.39",
  "torch": "2.14.0+cu132",
  "transformers": "5.22.2",
  "vllm": "0.34.0",
  "gpu": [
    "Tesla T4, 15360 MiB, 580.178.04",
    "Tesla T4, 15360 MiB, 580.178.04"
  ],
  "git_commit": "55fc9bf"
}
Fri Nov 20 20:56:45 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.178.04             Driver Version: 580.178.04     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   41C    P8              9W /   70W |       0MiB /  15360MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  Tesla T4                       Off |   00000000:00:05.0 Off |                    0 |
| N/A   43C    P8             10W /   70W |       0MiB /  15360MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+

torch.cuda.is_available(): True
GPU0: Tesla T4, 14.6 GiB, compute capability 7.5, SMs=40
GPU1: Tesla T4, 14.6 GiB, compute capability 7.5, SMs=40
CPU: CPU(s):                                  4
Model name:                              Intel(R) Xeon(R) CPU @ 2.00GHz

               total        used        free      shared  buff/cache   available
Mem:              31           1           0           0          29          29

## Software versions that worked
torch==2.14.0
transformers==5.20.2
vllm==0.34.0
## Entries
