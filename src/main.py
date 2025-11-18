"""Main orchestrator – spawns a single training run as a subprocess obeying the
CLI contract stated in the specification.

Usage example:
  uv run python -u -m src.main run=proposed-iter1-Qwen3-0.6B-gsm8k \
      results_dir=./results mode=full"""

import os
import subprocess
import sys
from typing import List

import hydra

@hydra.main(config_path="../config", config_name="config")
def main(cfg):
    if not hasattr(cfg, "run_id") or cfg.run_id is None:
        raise ValueError("Missing run group. Call with run=<run_id>.")

    run_id = cfg.run_id

    overrides: List[str] = [f"results_dir={cfg.results_dir}", f"mode={cfg.mode}"]

    if cfg.mode == "trial":
        overrides += [
            "wandb.mode=disabled",
            "optuna.n_trials=0",
            "training.num_epochs=1",
        ]
    elif cfg.mode == "full":
        overrides += ["wandb.mode=online"]
    else:
        raise ValueError("mode must be 'trial' or 'full'")

    # Get the project root directory (parent of src)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_script = os.path.join(project_root, "src", "train.py")

    cmd = [
        sys.executable,
        "-u",
        train_script,
        f"run={run_id}",
    ] + overrides

    # Add project root to PYTHONPATH so imports work correctly
    env = os.environ.copy()
    env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")

    print("Executing:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)

if __name__ == "__main__":
    main()
