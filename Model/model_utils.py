import os
# from sklearn.metrics import precision_score, recall_score, f1_score
import torch
import torch.nn as nn


def evaluate_model(model, loader, threshold=0.2, k=3):
    print("\nEvaluating on test set...")

    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            logits = model(x_batch)
            probs = torch.softmax(logits, dim=1)
            preds = (probs > threshold).int()

            all_preds.append(preds.cpu())
            all_targets.append(y_batch.cpu())
            all_probs.append(probs.cpu())

    y_true = torch.cat(all_targets, dim=0).numpy()
    y_pred = torch.cat(all_preds, dim=0).numpy()
    y_probs = torch.cat(all_probs, dim=0).numpy()

    # print(f"\nin evaulate: y_true: {y_true}") # ~nadav

    # make y_true binary for sklearn metrics
    y_true_bin = (y_true > 0).astype(int)

    # Macro (global) precision/recall/F1
    # precision = precision_score(y_true_bin, y_pred, average="macro", zero_division=0)
    # recall = recall_score(y_true_bin, y_pred, average="macro", zero_division=0)
    # f1 = f1_score(y_true_bin, y_pred, average="macro", zero_division=0)

    # print("\n🔢 Macro-Average (Overall):")
    # print(f"Precision = {precision:.3f}, Recall = {recall:.3f}, F1 = {f1:.3f}")

    # Simple case-level accuracy
    case_matches = (y_pred == y_true).all(axis=1)
    case_accuracy = case_matches.sum() / case_matches.shape[0]
    top_k_accuracy = (y_pred[:, :k] == y_true[:, :k]).all(axis=1).sum() / case_matches.shape[0]

    print(f"✅ Simple Case Accuracy: {case_accuracy*100:.2f}% ({case_matches.sum()}/{case_matches.shape[0]})")
    print(f"✅ Top-{k} Accuracy: {top_k_accuracy*100:.2f}% ({(y_pred[:, :k] == y_true[:, :k]).all(axis=1).sum()}/{case_matches.shape[0]})")
 
    return y_probs, y_true, y_pred



def train_model(model, train_loader, num_epochs, params_path):
    """
    Train the model using the provided data loader, optimizer, and loss function.
    """
    print("Training model...")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # need to take from config ~nadav
    criterion = nn.KLDivLoss(reduction="batchmean")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for x_batch, y_batch in train_loader:
            # Forward pass
            logits = model(x_batch)  # shape: (B, num_lines)
            # print("🔍 Logits stats — min:", logits.min().item(), "max:", logits.max().item())
            # print("🔍 Y_batch stats — min:", y_batch.min().item(), "max:", y_batch.max().item())

            # Compute loss
            loss = criterion(torch.log_softmax(logits, dim=1), y_batch)  # KLDivLoss expects log probabilities

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}")

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


def load_model(model, params_path, train_loader, num_epochs):
    """
    Load the model and optionally train it if not using a pretrained model.
    """
    if not os.path.exists(params_path):
        print("⚠️ Weights file not found!. Retraining...")
        train_model(model=model, train_loader=train_loader, num_epochs=num_epochs, params_path=params_path)
    
    checkpoint = torch.load(params_path)
    model_config = {
        "output_features": model.gcn.G,
        "poly_order": model.gcn.H,
        "in_features": model.gcn.K
    }
    
    if checkpoint.get("config_signature") != model_config:
        print("⚠️ Saved weights don't match current model configuration. Retraining...")
        train_model(model=model, train_loader=train_loader, num_epochs=num_epochs, params_path=params_path)
        
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