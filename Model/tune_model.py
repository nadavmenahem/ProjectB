import copy, optuna, torch, os, yaml, argparse
from pathlib import Path

from utils.data_utils   import get_data_loaders, get_config, get_graph
from utils.model_utils  import train_model, evaluate_model
from Model.model              import SpectralGCN
from Model.model_dft          import DFTSpectralGCN   # if you want to tune that too



BASE_CFG_PATH = "Model/config.yaml"          # <- your current settings file
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE        = torch.device("cpu")  # remove if you want to use GPU

# ---------- 1️⃣ command-line interface --------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", default="configs/base.yaml")
parser.add_argument("--study-name", default="spectral_gcn_tuning")
parser.add_argument("--db", default="optuna.db")
args = parser.parse_args()

# ---------- load once ----------
cfg_path   = Path(args.config)
BASE_CFG   = get_config(str(cfg_path))
STORAGE    = f"sqlite:///{args.db}"

# ------------ Optuna objective ------------------------------------------------
def objective(trial):
    # 1️⃣  take a *fresh* copy of the YAML each trial so we don’t mutate it
    cfg = copy.deepcopy(BASE_CFG)

    # 2️⃣  sample hyper-parameters
    cfg.learning_rate           = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    cfg.weight_decay            = trial.suggest_float("wd", 1e-6, 1e-3, log=True)
    cfg.model.hidden_dim        = trial.suggest_int        ("hidden_dim", 5, 8, log=True)  # 2⁵ … 2⁸
    cfg.dropout                 = trial.suggest_float      ("dropout",    0.0, 0.6)
    cfg.scheduler               = "steplr"
    cfg.num_epochs              = 120
    cfg.batch_size              = 64

    root = Path(__file__).resolve().parents[1]  # making root Code/
    data_dir = root / cfg.dataset.path
    cfg.dataset.path = str(data_dir)

    # 3️⃣  data – get loaders with the new batch-size
    dataset_path = os.path.join(cfg.dataset.path, cfg.dataset.network_name)
    train_loader, val_loader, cal_loader, test_loader, X_shape = get_data_loaders(
        dataset_path=dataset_path,
        batch_size=cfg.batch_size,
        test_size=cfg.dataset.test_size,
        val_size=cfg.dataset.val_size,
        cal_size=cfg.dataset.cal_size)

    G = get_graph(dataset_path)
    _, time_samples, num_input_features, num_nodes = X_shape

    assert num_nodes==G.number_of_nodes()

    # 4️⃣  build model *once per trial*
    if cfg.model.type == "gft":
        model = SpectralGCN(
            num_nodes       = num_nodes,
            time_samples    = time_samples,
            in_features     = num_input_features,
            out_features    = cfg.output_features,
            G               = G,
            H               = cfg.poly_order,
            num_classes     = G.number_of_edges(),
            hidden_dim      = cfg.model.hidden_dim
        ).to(DEVICE)
    else:  # DFT variant
        model = DFTSpectralGCN(...).to(DEVICE)

    params_path = os.path.join("checkpoints", f"{cfg.model.type}_trial_{trial.number}.pth")

    # 5️⃣  actual training loop
    train_model(model,
                train_loader,
                cfg,
                params_path)

    # 6️⃣  validation metric
    _, _, _, correct, total = evaluate_model(model, val_loader,
                                             loss_function="bce", k=cfg.topk)
    val_acc = correct / total

    # 7️⃣  report to Optuna & allow pruning
    trial.report(val_acc, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()

    return val_acc

# ------------ run the study ---------------------------------------------------
if __name__ == "__main__":
    study = optuna.create_study(
        study_name=args.study_name,
        storage=STORAGE,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner =optuna.pruners.MedianPruner(n_startup_trials=10)
    )
    study.optimize(objective, n_trials=50)

    print("🏆 Best trial:", study.best_trial.number,
          "val-acc =", study.best_value)
