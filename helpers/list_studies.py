# helpers/list_studies.py
import optuna
from pathlib import Path

from find_repo_root import find_repo_root


REPO_ROOT = find_repo_root(Path(__file__).resolve())
DB_PATH   = REPO_ROOT / "optuna.db"
storage_uri = f"sqlite:///{DB_PATH}"

print("Studies inside", DB_PATH)
for s in optuna.get_all_study_summaries(storage=storage_uri):
    print(" •", s.study_name)
