import numpy as np
import torch


class ConformalPredictor:
    def __init__(self, model, score_fn=None):
        """
        Generic conformal predictor for classification.

        Args:
            model (torch.nn.Module): Trained PyTorch model.
            score_fn (callable): Function (probs, labels) -> nonconformity scores.
        """
        self.model = model
        self.score_fn = score_fn or self.default_score_fn
        self.calibration_scores = None


    def default_score_fn(self, probs, labels):
        """
        Default nonconformity score: 1 - predicted prob for true label(s).
        Supports multi-class or multi-label (one-hot).

        labels: (batch_size, num_classes) one-hot encoded labels
        probs: (batch_size, num_classes) softmax probabilities
        """
        return 1 - (probs * labels).sum(dim=1)


    def calibrate(self, cal_loader):
        """
        Run model on calibration set and compute nonconformity scores.

        Args:
            cal_loader (DataLoader): Loader for calibration data.
        """
        self.model.eval()
        all_scores = []

        with torch.no_grad():
            for x_batch, y_batch in cal_loader:
                logits = self.model(x_batch)
                probs = torch.softmax(logits, dim=1)
                scores = self.score_fn(probs, y_batch)
                all_scores.extend(scores.cpu().numpy())

        self.calibration_scores = np.array(all_scores)


    def get_p_values(self, x_batch):
        """
        Compute conformal p-values for each class.

        Returns:
            pvals: (batch_size, num_classes) numpy array of p-values.
        """
        if self.calibration_scores is None:
            raise RuntimeError("Must call `calibrate()` first.")

        self.model.eval()
        with torch.no_grad():
            logits = self.model(x_batch)
            probs = torch.softmax(logits, dim=1)

        num_classes = probs.shape[1]  # probs.shape: (batch_size, num_classes)       
        pvals = []
        for c in range(num_classes):
            fake_labels = torch.zeros_like(probs)
            fake_labels[:, c] = 1.0
            scores = self.score_fn(probs, fake_labels).cpu().numpy()

            p_c = (np.sum(self.calibration_scores[:, None] >= scores[None, :], axis=0) + 1) / (
                len(self.calibration_scores) + 1
            )
            pvals.append(p_c)

        return np.stack(pvals, axis=1)


    def predict(self, x_batch, alpha=0.1, return_probs=False, return_pvals=False):
        """
        Make conformal predictions for a batch.

        Args:
            x_batch (Tensor): Input batch.
            alpha (float): Error tolerance (1 - confidence).
            return_probs (bool): Also return raw softmax probabilities.
            return_pvals (bool): Also return calibrated p-values.

        Returns:
            prediction_sets: boolean array (batch_size, num_classes)
            probs: optional, softmax probs
            pvals: optional, p-values
        """
        pvals = self.get_p_values(x_batch)
        prediction_sets = pvals > alpha

        results = [prediction_sets]
        if return_probs or return_pvals:
            with torch.no_grad():
                logits = self.model(x_batch)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            if return_probs:
                results.append(probs)
            if return_pvals:
                results.append(pvals)
            return tuple(results)

        return prediction_sets


    def predict_top_k(self, x_batch, k=3):
        """
        Return top-k labels with highest conformal p-values.
        """
        pvals = self.get_p_values(x_batch)
        return np.argsort(-pvals, axis=1)[:, :k]
