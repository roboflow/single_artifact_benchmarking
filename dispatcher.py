import os
os.environ["OMP_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
os.environ["OPENBLAS_NUM_THREADS"]="1"
import json
import subprocess
import threading
import queue
from pathlib import Path
import torch

REPO_ROOT: Path = Path(".").resolve()
DATA_ROOT: Path = REPO_ROOT / "rf100-vl" 
OUTPUT_ROOT: Path = REPO_ROOT / "rf100-vl-rf-detr-fp16"
MODELS_ROOT: Path = REPO_ROOT / "medium-fixed" 

RESULTS_FILE: Path = REPO_ROOT / "final_results.json"
ERROR_LOG:   Path = REPO_ROOT / "dispatch_errors.txt"

def _visible_gpu_ids() -> list[str]:
    return [str(i) for i in range(torch.cuda.device_count())]

gpu_ids = _visible_gpu_ids()
if not gpu_ids:
    raise RuntimeError("No CUDA devices visible – aborting.")

print(f"✓ Detected {len(gpu_ids)} GPU(s): {', '.join(gpu_ids)}")

datasets = sorted([p.name for p in DATA_ROOT.iterdir() if p.is_dir()])
dataset_q: queue.Queue[str] = queue.Queue()
for d in datasets:
    dataset_q.put(d)

lock = threading.Lock()
final_results: dict[str, str] = {}

def _worker(gpu_id: str):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id

    while True:
        try:
            dset = dataset_q.get_nowait()
        except queue.Empty:
            return

        dset_dir   = DATA_ROOT / dset
        out_dir    = OUTPUT_ROOT / dset
        model_dir  = MODELS_ROOT / dset
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "python",
            "get_rfdetr_rf100_scroes.py",
            "--dataset_dir", str(dset_dir),
            "--model_dir", str(model_dir),
            "--output_dir",  str(out_dir),
        ]

        print(f"[{dset}] → GPU {gpu_id}: {' '.join(cmd)}")
        try:

            with lock:
                final_results[dset] = "ok"
                done  = len(final_results)
                total = done + dataset_q.qsize()
                print(f"[{dset}] ✓ finished  ({done}/{total})")

        except subprocess.CalledProcessError as exc:
            print(f"[{dset}] ✗ FAILED on GPU {gpu_id}: {exc}")
            with lock:
                final_results[dset] = "failed"
            with ERROR_LOG.open("a") as log:
                log.write(f"{dset}\tGPU {gpu_id}\n")
        finally:
            dataset_q.task_done()

threads = [threading.Thread(target=_worker, args=(gid,)) for gid in gpu_ids]
for t in threads:
    t.start()
for t in threads:
    t.join()

with RESULTS_FILE.open("w") as f:
    json.dump(final_results, f, indent=2)

print(
    f"\n✓ Sweep finished. Results for {len(final_results)} dataset(s) "
    f"written to {RESULTS_FILE}"
)