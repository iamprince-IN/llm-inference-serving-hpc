"""Helpers: GPU monitoring with nvidia-smi, waiting for servers, recording the environment."""
import json
import platform
import shutil
import subprocess
import time
import urllib.request


class GpuMonitor:
    """Logs utilisation / memory / power every 500 ms to a CSV while a benchmark runs."""

    def __init__(self, path, interval_ms=500):
        self.path, self.interval_ms, self.proc, self.f = path, interval_ms, None, None

    def __enter__(self):
        if shutil.which("nvidia-smi"):
            self.f = open(self.path, "w")
            self.f.write("index,timestamp,util_gpu,mem_used_mib,power_w\n")
            self.f.flush()
            self.proc = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=index,timestamp,utilization.gpu,memory.used,power.draw",
                 "--format=csv,noheader,nounits", f"-lms={self.interval_ms}"],
                stdout=self.f, stderr=subprocess.DEVNULL)
        return self

    def __exit__(self, *a):
        if self.proc:
            self.proc.terminate()
            self.proc.wait(timeout=10)
        if self.f:
            self.f.close()


def wait_for_server(base_url, timeout=900, proc=None):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"server process exited with code {proc.returncode}; check its log")
        try:
            with urllib.request.urlopen(base_url.rstrip("/") + "/v1/models", timeout=5) as r:
                if r.status == 200:
                    return time.time() - t0
        except Exception:
            pass
        time.sleep(3)
    raise TimeoutError(f"{base_url} not ready after {timeout}s")


def _ver(mod):
    # read installed version without importing (importing vllm/torch here would be slow)
    from importlib.metadata import version, PackageNotFoundError
    try:
        return version(mod)
    except PackageNotFoundError:
        return None


def env_info():
    info = {"python": platform.python_version(), "platform": platform.platform(),
            "torch": _ver("torch"), "transformers": _ver("transformers"), "vllm": _ver("vllm")}
    try:
        info["gpu"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True).strip().splitlines()
    except Exception:
        info["gpu"] = None
    try:
        info["git_commit"] = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                                     stderr=subprocess.DEVNULL).strip()
    except Exception:
        info["git_commit"] = None
    return info


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
