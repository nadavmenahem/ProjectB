#===================IMPORTS==================
import os
import argparse
import random

from model import SpectralGCN
from model_dft import SpatialGCN
from utils.plot_utils import plot_test_case_probs, plot_test_case_clean
from utils.data_utils import get_data_loaders, get_config, get_graph, count_case, pick_case_cp_cover_but_not_topk
from utils.model_utils import train_model, evaluate_model, load_model, print_parameter_count
from my_conformal import *


# ==================DEVICE==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")  # remove if you want to use GPU


#==================CONFIG==================
RESULT_PLOTTING = True  # Set to True to plot the results
DEBUGGING = False  # Set to True to enable debugging mode


#=====================MAIN====================
def main():
    print(f"\nUsing device: {device}\n")

    parser = argparse.ArgumentParser(description="Spectral GCN for line outage identification")
    parser.add_argument('-c', '--config', type=str, default=None,
                        help='Path to a config.yaml file')
    args = parser.parse_args()
    config = get_config(args.config)

    dataset_path = os.path.join(config.dataset.path, config.dataset.network_name)
    # metadata = get_meta_data(dataset_path)
    G = get_graph(dataset_path)

    train_loader, val_loader, cal_loader, test_loader, X_shape = get_data_loaders(
        dataset_path=dataset_path,
        batch_size=config.batch_size,
        test_size=config.dataset.test_size,
        val_size=config.dataset.val_size,
        cal_size=config.dataset.cal_size)

    if DEBUGGING:
        case_to_check = (6, 7)

        print("Train:", count_case(train_loader, case_to_check))
        print("Val:  ", count_case(val_loader, case_to_check))
        print("Cal:  ", count_case(cal_loader, case_to_check))
        print("Test: ", count_case(test_loader, case_to_check))

    # X_shape == (B, T, K, N)
    _, time_samples, num_input_features, _ = X_shape

    if DEBUGGING:  
        print("DEBUGGING: X shape:", X_shape)
        print("DEBUGGING: num_input_features:", num_input_features)

    params_path = os.path.join(config.model.params_folder, f"{config.model.type}.pth")

    if config.model.type == 'gft':
        model = SpectralGCN(
            num_nodes=G.number_of_nodes(),
            time_samples=time_samples,           # T
            in_features=num_input_features,      # K
            out_features=config.output_features, # G
            G=G, # graph
            H=config.poly_order, # H
            num_classes=G.number_of_edges(), # one output per power line
            hidden_dim=config.model.hidden_dim,
            dropout=config.dropout
        )

    elif config.model.type == 'dft':
        model = SpatialGCN(
            num_nodes=G.number_of_nodes(),
            time_samples=time_samples,           # T
            in_features=num_input_features,      # K
            out_features=config.output_features, # G
            G=G, # graph
            H=config.poly_order, # H
            num_classes=G.number_of_edges(), # one output per power line
            hidden_dim=config.model.hidden_dim,
            dropout=config.dropout
        )

    else:
        raise ValueError(f"Unknown model type: {config.model.type}. Supported types are 'gft' and 'dft'.")

    model.to(device)

    # if DEBUGGING:
    if True:
        for name, param in model.named_parameters():
            print(f"{name:30} | requires_grad: {param.requires_grad}")
        print_parameter_count(model)

    # Training the model
    if config.model.load_pretrained:
        load_model(model=model, params_path=params_path)
    else:
        train_model(model, train_loader, config, params_path, val_loader=val_loader)


    # ---------- Conformal prediction (APS) + Evaluation + Plotting ----------

    # 1) Evaluate Top-K as you already do
    y_probs, y_true, topk_idx, correct, total, avg_time = evaluate_model(
        model, test_loader, config.loss_function, k=config.topk
    )
    print(f"\nTest Top-{config.topk} Accuracy: {(correct / total) * 100:.2f}% ({correct}/{total})")
    print(f"average forward pass time: {avg_time} seconds")

    # 2) Collect calibration & test probs/targets for CP
    P_cal, Y_cal = get_probs_and_targets(model, cal_loader, loss_name=config.loss_function)
    # Use probs/targets from evaluate_model for test (or re-compute; both fine)
    P_tst = np.asarray(y_probs)
    Y_tst = np.asarray(y_true)

    # 3) Calibrate and predict APS sets
    alpha = getattr(config, "alpha", 0.1)
    # tau = aps_calibrate(P_cal, Y_cal, alpha=alpha, randomized=True)
    # cp_sets = aps_predict_sets(P_tst, tau)
    tau = raps_calibrate(P_cal, Y_cal, alpha=alpha, lam=0.05, k0=1, randomized=True)
    cp_sets = raps_predict_sets(P_tst, tau, lam=0.05, k0=1)

    # (helper) turn list-of-indices → boolean mask [N, L] for plotting ease
    num_labels = P_tst.shape[1]
    cp_mask = np.zeros((len(cp_sets), num_labels), dtype=bool)
    for i, S in enumerate(cp_sets):
        if len(S) > 0:
            cp_mask[i, S] = True

    # 4) CP metrics
    print(f"Conformal (APS) α={alpha:.3f}  τ={tau:.6f}")
    print(f"  Example-wise coverage: {examplewise_coverage(cp_sets, Y_tst) * 100:.2f}%")
    print(f"  Avg set size:          {avg_set_size(cp_sets):.2f}")

    # 5) Plot a few cases (fixing the undefined vars)
    if RESULT_PLOTTING:
        N = len(P_tst)
        chosen = sorted(random.sample(range(N), k=min(5, N)))
        for i in chosen:
            plot_test_case_probs(
                probs=P_tst[i],
                true_labels=Y_tst[i],
                case_idx=i,
                topk_indices=(topk_idx[i].tolist() if hasattr(topk_idx[i], "tolist") else list(topk_idx[i])),
                conformal_set=np.flatnonzero(cp_mask[i]).tolist()
            )
            plot_test_case_clean(
                probs=P_tst[i],
                true_labels=Y_tst[i],
                case_idx=i,
                conformal_set=np.flatnonzero(cp_mask[i]).tolist()
            )
        i_bad_topk = pick_case_cp_cover_but_not_topk(P_tst, Y_tst, cp_mask, k=3, y_thr=0.1)
        if i_bad_topk is not None:
            print(f"Plotting CP-covered but top-k-missed case: {i_bad_topk}")
            plot_test_case_clean(
                probs=P_tst[i_bad_topk],
                true_labels=Y_tst[i_bad_topk],
                case_idx=i_bad_topk,
                conformal_set=np.flatnonzero(cp_mask[i_bad_topk]).tolist()
            )
        else:
            print("No case found where CP covers but top-k misses (try different k or y_thr).")

    if DEBUGGING:
        # X, Y = load_dataset(dataset_path)
        # for i, y in enumerate(Y):
        #     print(f"Sample {i}: sum = {np.sum(y)}, y = {y}")
        print("DEBUGGING: number of nodes:", G.number_of_nodes())
        print("DEBUGGING: number of edges:", G.number_of_edges())


if __name__ == "__main__":
    main()
