import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR



def make_scheduler(optimizer, cfg):
    name = cfg.scheduler.lower()
    if name == "steplr":
        return StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)
    elif name == "reduceonplateau":
        return ReduceLROnPlateau(optimizer,
                                 mode="min" if cfg.plateau_metric=="loss" else "max",
                                 factor=cfg.gamma,
                                 patience=cfg.step_size)
    elif name == "cosine":
        return CosineAnnealingLR(optimizer,
                                 T_max=cfg.num_epochs,
                                 eta_min=cfg.learning_rate * 0.01)
    else:
        raise ValueError(f"Unknown scheduler '{cfg.scheduler}'")



# device ~nadav
def evaluate_model(model, loader, loss_function, k=3):
    """
    Evaluate the model on the provided data loader.
    """
    model.eval()
    all_topk, all_targets, all_probs = [], [], []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(model.device)
            logits  = model(x_batch)

            if loss_function.lower() == "bce":
                probs = torch.sigmoid(logits)              # (B, num_lines)
            else:  # DKL
                probs = nn.functional.softmax(logits, dim=1)

            topk    = torch.topk(probs, k, dim=1).indices

            all_targets.append((y_batch > 0).int().cpu())
            all_probs.append(probs.cpu())
            all_topk.append(topk.cpu())

    # concatenate over batches
    y_true   = torch.cat(all_targets, 0).numpy()   # shape (N, L)
    y_probs  = torch.cat(all_probs,   0).numpy()   # shape (N, L)
    topk_idx = torch.cat(all_topk,    0).numpy()   # shape (N, k)

    # full-coverage accuracy: every true outage must be in the top-k
    correct = 0
    for true_vec, pred_idx in zip(y_true, topk_idx):
        true_indices = set(np.where(true_vec == 1)[0])
        if true_indices.issubset(set(pred_idx)):
            correct += 1

    total = len(y_true)
    return y_probs, y_true, topk_idx, correct, total



def get_pos_weights(train_loader):
    """
    Compute positive class weights for each output node (line) in the dataset.
    This is useful for imbalanced datasets in multi-label classification.
    Returns a tensor of weights to be used with BCEWithLogitsLoss.
    """
    all_targets = torch.cat([y for _, y in train_loader], 0)   # shape (N, num_lines)
    pos         = (all_targets == 1).sum(0).float()
    neg         = (all_targets == 0).sum(0).float()
    pos_weight  = neg / (pos + 1e-6)

    return pos_weight


def train_model(model, train_loader, config, params_path, test_loader=None):
    """
    Train the model using the provided data loader, optimizer, and loss function.
    """
    k = config.topk
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler  = make_scheduler(optimizer, config) 
    loss_name = config.loss_function.lower()
    print(f"Using loss function: {loss_name.upper()}")
    if loss_name == "bce":
        pos_w = get_pos_weights(train_loader).to(model.device)
        criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_w)  
    elif loss_name == "dkl":
        pass
        # criterion  = nn.KLDivLoss(reduction='batchmean')
    else:
        raise ValueError(f"Unknown loss function: {loss_name}. Use 'bce' and 'dkl'.")

    print("Training model...")
    for epoch in range(config.num_epochs):
        start_time = time.time()

        model.train()
        epoch_loss = 0.0

        for x_batch, y_batch in train_loader:
            # convert probabilistic targets -> binary multi-hot
            y_batch = y_batch.to(model.device)
            targets = (y_batch > 0).float()

            # Forward pass
            logits = model(x_batch)  # shape: (B, num_lines)

            ε = config.epsilon  # small smoothing factor
            # Compute loss
            if loss_name == "bce":
                # for each element: 1 → 1-ε,  0 → ε
                targets = targets * (1 - ε) + (1 - targets) * ε
                loss = criterion(logits, targets)  
            else:  # DKL
                loss = jeffreys_kl_loss(logits, targets, entropy_weight=config.entropy_weight, ε=ε)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        end_time = time.time()
        epoch_duration = end_time - start_time

        print(f"Epoch {epoch+1}/{config.num_epochs} - Loss: {epoch_loss:.4f} - Time: {epoch_duration:.2f}s")
        _, _, _, correct, total = evaluate_model(model, train_loader, config.loss_function, k)
        print(f"Train Top-{k} Accuracy: {(correct/total)*100:.2f}% ({correct}/{total})")

        # Step the learning rate scheduler
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            metric = epoch_loss if config.plateau_metric == "loss" else correct / total
            scheduler.step(metric)                       # needs a metric
        else:                                            # StepLR, Cosine, …
            scheduler.step()                             # plain step
        # print(f"LR now {optimizer.param_groups[0]['lr']:.2e}")

        # Evaluate on test set if provided
        if test_loader is not None:
            _, _, _, test_correct, test_total = evaluate_model(model, test_loader, config.loss_function, k)
            print(f"Test Top-{k} Accuracy: {(test_correct/test_total)*100:.2f}% ({test_correct}/{test_total})")

    torch.save({
        "model_state": model.state_dict(),
        "config_signature": {
            "output_features": model.gcn.G,
            "poly_order": model.gcn.H,
            "in_features": model.gcn.K
        }
    }, params_path)  # saving model parameters and configuration 
    print(f"Model trained and saved to {params_path}")
    

def save_model(model, path):
    print(f"saving model in {path}")
    torch.save(model.state_dict(), path)


def load_model(model, params_path):
    """
    Load the model and optionally train it if not using a pretrained model.
    """
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Weights file not found at {params_path}. Aborting.")

    checkpoint = torch.load(params_path, map_location=model.device, weights_only=True)
    model_config = {
        "output_features": model.gcn.G,
        "poly_order": model.gcn.H,
        "in_features": model.gcn.K
    }
    
    if checkpoint.get("config_signature") != model_config:
        raise FileNotFoundError("Saved weights don't match current model configuration. Aborting")
        
    else:
        print("Loading pretrained model...")
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        print(f"Model loaded.")


def print_parameter_count(model):
    """
    Print the number of parameters in each layer of the model and the total number of parameters.
    """
    total_params = 0
    print(f"\n{'Name':30} | {'Shape':20} | {'Parameters'}")
    print("-" * 60)
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count = param.numel()
            total_params += param_count
            print(f"{name:30} | {str(tuple(param.shape)):20} | {param_count:,}")

    print("-" * 60)
    print(f"{'Total Trainable Parameters':30} | {'':20} | {total_params:,}\n")


# ---------------------------------------------------------
# helper: build a valid target distribution for Jeffreys KL
# ---------------------------------------------------------
def build_target_distribution(y_batch: torch.Tensor, ε) -> torch.Tensor:
    """
    y_batch : (B, G)   multi-hot tensor (0/1) indicating which lines are out
    returns : (B, G)   probability rows that each sum to 1
                       • if ≥1 line is out  → uniform over those lines
                       • if no line is out → uniform over ALL G lines
    """
    B, G = y_batch.shape
    targets = torch.zeros_like(y_batch, dtype=torch.float)

    for i in range(B):
        positives = y_batch[i].nonzero(as_tuple=False).squeeze(1)  # indices of outaged lines

        if positives.numel() > 0:                # one or more outages
            prob = 1.0 / positives.numel()
            targets[i, positives] = prob         # uniform on positive lines
        else:                                    # no outage → uniform row
            targets[i].fill_(1.0 / G)

    # smoothing to avoid exact 0/1 probabilities
    uniform = torch.full_like(targets, 1.0 / G)
    targets = targets * (1.0 - ε) + uniform * ε

    return targets                                # (B, G)


def jeffreys_kl_loss(logits, y_batch, entropy_weight=0.0, ε=0.9):
    """
    Compute the Jeffreys KL divergence loss between model logits and target distribution.
    
    Args:
        logits: (B, G) tensor of model outputs (logits)
        y_batch: (B, G) multi-hot tensor indicating which lines are out
    """
    eps = 1e-8                        # numerical floor

    # model distribution p and log p
    log_p = F.log_softmax(logits, dim=1)      # log p_i
    p     = log_p.exp().clamp_min(eps)        # p_i  (avoid exact 0)

    # target distribution q
    q      = build_target_distribution(y_batch, ε)       # (B, G)
    log_q  = (q + eps).log()

    kl_pq  = F.kl_div(log_p, q,   reduction="batchmean")   # forward
    kl_qp  = F.kl_div(log_q, p,   reduction="batchmean")   # reverse
    loss   = kl_pq + kl_qp  # Jeffreys divergence

    # optional extra entropy penalty (set λ > 0 in config)
    entropy = -(p * log_p).sum(dim=1).mean()           # mean batch entropy
    loss   -= entropy_weight * entropy

    return loss 