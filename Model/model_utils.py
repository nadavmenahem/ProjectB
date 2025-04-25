from sklearn.metrics import precision_score, recall_score, f1_score
import torch


def evaluate_model(model, loader, threshold=0.2):
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
            all_targets.append(y_batch.int().cpu())
            all_probs.append(probs.cpu())

    y_true = torch.cat(all_targets, dim=0).numpy()
    y_pred = torch.cat(all_preds, dim=0).numpy()
    y_probs = torch.cat(all_probs, dim=0).numpy()

    # Macro (global) precision/recall/F1
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print("\n🔢 Macro-Average (Overall):")
    print(f"Precision = {precision:.3f}, Recall = {recall:.3f}, F1 = {f1:.3f}")

    # Simple case-level accuracy
    case_matches = (y_pred == y_true).all(axis=1)
    case_accuracy = case_matches.sum() / case_matches.shape[0]
    print(f"✅ Simple Case Accuracy: {case_accuracy*100:.2f}% ({case_matches.sum()}/{case_matches.shape[0]})")
 
    return y_probs, y_true, y_pred



def train_model(model, train_loader, optimizer, criterion, num_epochs):
    print("Training model...")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for x_batch, y_batch in train_loader:
            # Forward pass
            logits = model(x_batch)  # shape: (B, num_lines)

            # Compute loss
            loss = criterion(torch.log_softmax(logits, dim=1), y_batch)  # KLDivLoss expects log probabilities

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}")

