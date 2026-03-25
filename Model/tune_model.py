import argparse
import copy
import datetime as _dt
import os
from pathlib import Path
import yaml
import optuna
import torch

from utils.data_utils import get_data_loaders, get_config, get_graph
from utils.model_utils import train_model, evaluate_model
from model import SpectralGCN
from model_dft import SpatialGCN  # if you want to tune that too


NUMBER_OF_TRIALS = 200
NUMBER_OF_EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")  # uncomment to force CPU

# --------------------------------------------------------------------------------------
# Optuna objective
# --------------------------------------------------------------------------------------
def objective(trial: optuna.trial.Trial):
    # ----- clone the base config -------------------------------------------------------
    cfg = copy.deepcopy(base_cfg)

    # ----- sample hyper‑parameters -----------------------------------------------------
    cfg.loss_function = trial.suggest_categorical("loss function", ["dkl", "bce", "Jeffreys"])
    cfg.model.type = trial.suggest_categorical("model type", ["gft", "dft"])
    cfg.optimizer = trial.suggest_categorical("optimizer", ["adam", "sgd"])
    if cfg.optimizer == "adam":
        cfg.learning_rate = trial.suggest_float("learning rate", 1e-5, 1e-3, log=True)
    else:  # sgd
        cfg.learning_rate = trial.suggest_float("learning rate", 1e-3, 1e-1, log=True)
    # cfg.learning_rate = trial.suggest_float("learning rate", 1e-5, 1e-2, log=True)
    cfg.weight_decay = trial.suggest_float("weight decay", 1e-6, 1e-3, log=True)

    hidden_exp = trial.suggest_int("log2(hidden_dim)", 5, 10, log=False)  # 2^5 … 2^8 = 32–1024
    cfg.model.hidden_dim = 2 ** hidden_exp  # convert exponent → actual dim

    cfg.dropout = trial.suggest_float("dropout", 0.0, 0.6)
    cfg.scheduler = trial.suggest_categorical("scheduler", ["steplr", "Cosine", "ReduceLROnPlateau"])
    batch_size = trial.suggest_int("log2(batch size)", 3, 7, log=False)  # 2^3 … 2^7 = 8–128
    cfg.poly_order = trial.suggest_int("GNN's poynomial order", 1, 2, log=False)  # 2^3 … 2^7 = 8–128
    cfg.batch_size = 2 ** batch_size
    cfg.num_epochs = NUMBER_OF_EPOCHS
    cfg.load_pretrained = False

    # ----- helper ----------------------------------------------------------------------
    # define the epoch callback for Optuna
    def epoch_callback(epoch: int):
        model.eval()
        with torch.no_grad():
            _, _, _, correct, total, _ = evaluate_model(
                model, val_loader, loss_function="bce", k=cfg.topk
            )
        val_acc = correct / total
        trial.report(val_acc, step=epoch + 1)  # use 1-based steps to avoid step=0 edge cases
        if trial.should_prune():
            print(f"[trial {trial.number}] PRUNED at epoch {epoch}")
            raise optuna.TrialPruned()
        return False

    # ----- dataset paths ---------------------------------------------------------------
    root = Path(__file__).resolve().parents[1]  # ProjectB/Code/
    data_dir = root / cfg.dataset.path
    cfg.dataset.path = str(data_dir)

    dataset_path = os.path.join(cfg.dataset.path, cfg.dataset.network_name)

    train_loader, val_loader, cal_loader, test_loader, X_shape = get_data_loaders(
        dataset_path=dataset_path,
        batch_size=cfg.batch_size,
        test_size=cfg.dataset.test_size,
        val_size=cfg.dataset.val_size,
        cal_size=cfg.dataset.cal_size,
    )

    # ----- build model -----------------------------------------------------------------
    G = get_graph(dataset_path)
    _, time_samples, num_features, num_nodes = X_shape
    assert num_nodes == G.number_of_nodes()

    if cfg.model.type == "gft":
        model = SpectralGCN(
            num_nodes=num_nodes,
            time_samples=time_samples,
            in_features=num_features,
            out_features=cfg.output_features,
            G=G,
            H=cfg.poly_order,
            num_classes=G.number_of_edges(),
            hidden_dim=cfg.model.hidden_dim,
            dropout=cfg.dropout
        ).to(DEVICE)
    else:  # DFT variant (fill in args as needed)
        model = SpatialGCN(
            num_nodes=G.number_of_nodes(),
            time_samples=time_samples,  # T
            in_features=num_features,  # K
            out_features=cfg.output_features,  # G
            G=G,  # graph
            H=cfg.poly_order,  # H
            num_classes=G.number_of_edges(),  # one output per power line
            hidden_dim=cfg.model.hidden_dim,
            dropout=cfg.dropout,
        )

    # ----- checkpoint path -------------------------------------------------------------
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    params_path = ckpt_dir / f"{cfg.model.type}_trial_{trial.number}.pth"

    # ----- save the exact cfg used in this trial ---------------------------------------
    trial_dir = Path("runs") / f"trial_{trial.number:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = trial_dir / "config.yaml"
    yaml_path.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    trial.set_user_attr("config_path", str(yaml_path.resolve()))

    # ----- training --------------------------------------------------------------------
    try:
        train_model(model, train_loader, cfg, params_path, epoch_callback=epoch_callback)
    except optuna.TrialPruned:
        # free GPU memory if using CUDA to avoid OOM across many pruned trials
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise

    # ----- validation metric -----------------------------------------------------------
    _, _, _, correct, total, _ = evaluate_model(
        model, val_loader, loss_function="bce", k=cfg.topk
    ) # any loss function... we only need accuracy
    val_acc = correct / total

    trial.report(val_acc, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()

    return val_acc

# --------------------------------------------------------------------------------------
# Run the study
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Spectral-GCN Optuna sweep")
    parser.add_argument("-c", "--config", default="configs/base.yaml",
                        help="Path to the YAML config to use as a template.")
    parser.add_argument("--study-name", default=None,
                        help="If omitted, a new timestamped study name is generated each run.")
    parser.add_argument("--db", default="optuna.db",
                        help="SQLite file that stores all studies.")
    args = parser.parse_args()

    if args.study_name is None:
        timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        study_name = f"spectral_gcn_tuning_{timestamp}"
        load_if_exists = False  # force a fresh study when name is auto‑generated
    else:
        study_name = args.study_name
        load_if_exists = True   # resume if the name already exists

    storage_uri = f"sqlite:///{Path(args.db).resolve()}"

    # --------------------------------------------------------------------------------------
    # Load base config once (as a dot‑access object)
    # --------------------------------------------------------------------------------------
    base_cfg = get_config(str(Path(args.config).resolve()))


    study = optuna.create_study(
        study_name=study_name,
        storage=storage_uri,
        direction="maximize",
        load_if_exists=load_if_exists,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
    )

    print(f"Running study: {study.study_name} | storage → {storage_uri}")
    study.optimize(objective, n_trials=NUMBER_OF_TRIALS)

    print("Best trial", study.best_trial.number, "val-acc =", study.best_value)
