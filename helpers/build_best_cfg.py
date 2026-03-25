import optuna, shutil
from pathlib import Path
from find_repo_root import find_repo_root

"""
This script is meant to be run after you've done hyperparameter tuning with tune_model.py. It will:
1. Load the best trial from the Optuna study.
2. Extract the path to the config.yaml that was saved by that trial.
3. Copy that config.yaml to Model/configs/best.yaml for easy access in future runs.

IMPORTANT: change the STUDY_NAME to match the name of your best trial's study. 
You can find it in the Optuna dashboard or in the console output after tuning.
"""

# ───────────────────────────────────────── repo paths ─────────────────────────────────────────
REPO_ROOT = find_repo_root(Path(__file__).resolve())
DB_PATH    = REPO_ROOT / "optuna.db"                          # one DB for all studies
STUDY_NAME = "spectral_gcn_tuning_20260323_211733"            # IMPORTANT! copy from dashboard tile
OUT_YAML   = REPO_ROOT / "Model" / "configs" / "best.yaml"    # where we'll write

storage_uri = f"sqlite:///{DB_PATH}"
study = optuna.load_study(study_name=STUDY_NAME, storage=storage_uri)
best  = study.best_trial

# ─────────────────────── pick up the full YAML path saved by the trial ────────────────────────
if "config_path" not in best.user_attrs:
    raise SystemExit(
        "    The best trial did not record 'config_path'.\n"
        "    Make sure your objective() dumps cfg and calls\n"
        "    trial.set_user_attr('config_path', str(yaml_path))"
    )

full_cfg_path = Path(best.user_attrs["config_path"]).resolve()
print("Using config from best trial:", full_cfg_path)

# simply copy it (or load-modify-write if you still want tweaks)
shutil.copy2(full_cfg_path, OUT_YAML)
print("Saved tuned config →", OUT_YAML)
