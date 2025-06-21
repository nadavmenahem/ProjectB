import numpy as np
import torch


class ConformalPredictor:
    def __init__(self, model, loss_name, score_fn=None):
        """
        Generic conformal predictor for classification.

        Args:
            model (torch.nn.Module): Trained PyTorch model.
            score_fn (callable): Function (probs, labels) -> nonconformity scores.
        """
        self.model = model
        self.loss_name = loss_name.lower()
        if self.loss_name not in ["bce", "dkl"]:
            raise ValueError(f"Unsupported loss function: {self.loss_name}. Use 'bce' or 'dkl'.")
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
        Run model on the calibration set and store a matrix of
        non-conformity scores with shape (N_calibration, num_classes).
        """
        self.model.eval()
        batch_matrices = []                       # <- collect 2-D blocks

        with torch.no_grad():
            for x_batch, y_batch in cal_loader:
                logits = self.model(x_batch)
                probs  = (torch.sigmoid(logits) if self.loss_name == "bce"
                        else torch.softmax(logits, dim=1))

                # ---------- one matrix per batch ----------
                if self.loss_name == "bce":              # multi-label
                    # score = 1-p for positive labels, p otherwise
                    batch_scores = self._binary_score(probs, y_batch)  # (B, C)

                else:                                    # multi-class
                    # need a score FOR EVERY CLASS
                    B, C = probs.shape
                    eye  = torch.eye(C, device=probs.device)           # (C, C)
                    per_class = []
                    for c in range(C):
                        fake_y = eye[c].expand(B, C)                   # (B, C)
                        s      = self.score_fn(probs, fake_y)          # (B,)
                        per_class.append(s.unsqueeze(1))               # (B,1)
                    batch_scores = torch.cat(per_class, dim=1)         # (B, C)

                batch_matrices.append(batch_scores.cpu().numpy())
                # ------------------------------------------

        # final shape (N_calibration, num_classes)
        self.calibration_scores = np.concatenate(batch_matrices, axis=0).astype(np.float32)


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
            probs  = (torch.sigmoid(logits) if self.loss_name == "bce"
                   else torch.softmax(logits, dim=1))

        batch_size, num_classes = probs.shape
        pvals   = np.zeros((batch_size, num_classes), dtype=np.float32) # output
        N_cal   = self.calibration_scores.shape[0] # number of calibration samples
        
        for c in range(num_classes):
            if self.loss_name == "bce":
                # multilabel:  score = 1 - p_c
                scores = (1.0 - probs[:, c]).numpy()
            else:
                # multiclass: reuse the generic score_fn
                fake_labels = torch.zeros_like(probs)
                fake_labels[:, c] = 1.0
                scores = self.score_fn(probs, fake_labels).cpu().numpy()

            # Compare to class-specific calibration scores
            calib_scores_c = self.calibration_scores[:, c]
            p_c = ( (calib_scores_c[:, None] >= scores[None, :]).sum(axis=0) + 1
              ) / (N_cal + 1)

            # Store
            pvals[:, c] = p_c

        return pvals # pvals[i, c] is the p-value for class c on sample i in the batch


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
                if self.loss_name == "bce":
                    probs = torch.sigmoid(logits)
                else:  # DKL    
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


    # for binary classification
    def _binary_score(self, probs, labels):
        return torch.where(labels.bool(), 1 - probs, probs)