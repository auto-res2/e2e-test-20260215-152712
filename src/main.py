import os
import subprocess
import sys

from hydra import main
from hydra.utils import get_original_cwd


@main(config_path="../config", config_name="config", version_base="1.3")
def main_entry(cfg) -> None:
    results_dir = cfg.results_dir
    if not os.path.isabs(results_dir):
        results_dir = os.path.join(get_original_cwd(), results_dir)
    os.makedirs(results_dir, exist_ok=True)

    overrides = [
        f"run={cfg.run}",
        f"results_dir={results_dir}",
        f"mode={cfg.mode}",
        f"seed={cfg.seed}",
    ]

    if cfg.mode == "trial":
        overrides.extend(
            [
                "wandb.mode=disabled",
                "optuna.n_trials=0",
                "optuna.enabled=false",
                "training.epochs=1",
            ]
        )
    elif cfg.mode == "full":
        overrides.append("wandb.mode=online")

    cmd = [sys.executable, "-m", "src.train"] + overrides
    subprocess.run(cmd, check=True, cwd=get_original_cwd())


if __name__ == "__main__":
    main_entry()
