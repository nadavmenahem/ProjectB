import math, numpy as np, torch
import torch.nn as nn

# ---------------------------------------------------------------------
# 1) Get probabilities from your trained model
# ---------------------------------------------------------------------
@torch.no_grad()
def get_probs_and_targets(model, loader, loss_name="bce"):
    model.eval()
    all_p, all_y = [], []
    for xb, yb in loader:
        xb = xb.to(model.device)
        logits = model(xb)                               # (B, L)
        if loss_name.lower() == "bce":                   # multilabel
            p = torch.sigmoid(logits)
        else:                                            # multiclass (exactly one label)
            p = nn.functional.softmax(logits, dim=1)
        all_p.append(p.cpu())
        all_y.append((yb > 0).int().cpu())
    P = torch.cat(all_p, 0).numpy().astype(np.float64)   # (N, L)
    Y = torch.cat(all_y, 0).numpy().astype(np.int32)     # (N, L)
    return P, Y

# Finite-sample conformal quantile helper
def conformal_quantile(values, alpha, side="upper"):
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n == 0:
        return 1.0 if side == "upper" else 0.0
    # For upper-tail thresholds (like APS), use (1-α) quantile with "higher" interpolation
    q = math.ceil((n + 1) * (1 - alpha)) / n if side == "upper" else math.floor((n + 1) * alpha) / n
    q = min(max(q, 0.0), 1.0)
    return np.quantile(values, q, method="higher" if hasattr(np, "quantile") else "higher")

# ---------------------------------------------------------------------
# 2A) APS (example-wise) calibration & prediction
# ---------------------------------------------------------------------
def aps_calibrate(P_cal, Y_cal, alpha=0.1, randomized=True):
    """
    For each calibration sample, compute the cumulative probability mass needed
    to cover all true labels when sorting labels by descending prob.
    τ is set to the (1-α) conformal quantile of these masses.
    """
    masses = []
    for p, y in zip(P_cal, Y_cal):
        order = np.argsort(-p)
        p_sorted = p[order]
        y_sorted = y[order].astype(bool)
        if y_sorted.any():
            # worst-rank (lowest-prob) true label position
            r = np.where(y_sorted)[0].max()  # 0-based
            mass = p_sorted[: r + 1].sum()
            if randomized and r + 1 < len(p_sorted):
                mass += np.random.rand() * p_sorted[r + 1]  # standard randomized APS
            masses.append(mass)
        else:
            # no positives in calibration example -> contributes 0 mass
            masses.append(0.0)
    tau = conformal_quantile(masses, alpha, side="upper")
    return float(tau)

def aps_predict_sets(P, tau):
    """
    Build prediction set by taking top labels until cumulative prob >= tau.
    Also include ties at the boundary to avoid undercoverage.
    Returns: list of 1D int arrays (indices chosen per sample).
    """
    sets = []
    for p in P:
        order = np.argsort(-p)
        csum = 0.0
        k = 0
        while k < len(order) and csum < tau:
            csum += p[order[k]]
            k += 1
        if k == 0:
            sets.append(np.array([], dtype=int))
            continue
        boundary = p[order[k - 1]]
        S = set(order[:k])
        # include any ties at the boundary prob
        for j in order[k:]:
            if abs(p[j] - boundary) <= 1e-12:
                S.add(j)
            else:
                break  # since probs are sorted, once strictly smaller we can stop
        sets.append(np.fromiter(S, dtype=int))
    return sets


# Regularized APS
def raps_calibrate(P_cal, Y_cal, alpha=0.1, lam=0.05, k0=0, randomized=True):
    masses = []
    for p, y in zip(P_cal, Y_cal):
        order    = np.argsort(-p)
        p_sorted = p[order]
        y_sorted = y[order].astype(bool)
        if y_sorted.any():
            r = np.where(y_sorted)[0].max()         # worst-ranked true
            k = r + 1
            mass = p_sorted[:k].sum() + lam * max(0, k - k0)
            if randomized and k < len(p_sorted):
                mass += np.random.rand() * (p_sorted[k] + lam)
            masses.append(mass)
        else:
            masses.append(0.0)
    return conformal_quantile(np.asarray(masses, float), alpha)

# def raps_predict_sets(P, tau, lam=0.05, k0=0):
#     sets = []
#     for p in P:
#         order = np.argsort(-p)
#         csum, k = 0.0, 0
#         while k < len(order):
#             next_mass = csum + p[order[k]] + lam * max(0, (k+1) - k0)
#             if next_mass >= tau:
#                 k += 1
#                 break
#             csum = csum + p[order[k]]
#             k += 1
#         sets.append(order[:k])
#     return [np.array(S, dtype=int) for S in sets]

def raps_predict_sets(P, tau, lam=0.05, k0=1, rng=None):
    if rng is None: rng = np.random.default_rng()
    sets = []
    for p in P:
        order = np.argsort(-p); s = p[order]
        csum, L = 0.0, 0
        reg = 0.0
        while L < len(s):
            next_bar = s[L] + (lam if (L+1) > k0 else 0.0)
            if csum + reg + next_bar >= tau:
                csum += s[L]; reg += (lam if (L+1) > k0 else 0.0)
                L += 1
                break
            csum += s[L]; reg += (lam if (L+1) > k0 else 0.0)
            L += 1
        if L == 0:
            sets.append(np.array([], dtype=int)); continue
        # overshoot fraction wrt the effective last bar
        S_eff = csum + reg
        last_bar = s[L-1] + (lam if L > k0 else 0.0)
        V = (S_eff - tau) / max(last_bar, 1e-12)
        if rng.random() <= V:
            L -= 1
        sets.append(order[:L])
    return [np.array(S, dtype=int) for S in sets]


# ---------------------------------------------------------------------
# 2B) Subset-threshold (example-wise, simple) calibration & prediction
# ---------------------------------------------------------------------
def subset_calibrate(P_cal, Y_cal, alpha=0.1):
    """
    Compute t so that with prob ≥ 1-α, min predicted prob among the true labels ≥ t.
    """
    mins = []
    for p, y in zip(P_cal, Y_cal):
        pos = p[y.astype(bool)]
        mins.append(pos.min() if pos.size else 1.0)
    # Choose t as the α conformal quantile of the mins
    t = conformal_quantile(mins, alpha, side="lower")
    return float(t)

def subset_predict_sets(P, t):
    return [np.where(p >= t)[0] for p in P]

# ---------------------------------------------------------------------
# 2C) Label-wise calibration & prediction
# ---------------------------------------------------------------------
def labelwise_calibrate(P_cal, Y_cal, alpha=0.1):
    """
    For each label ℓ, set threshold tℓ as the α quantile of probs among positives (Y=1) for that label.
    """
    L = P_cal.shape[1]
    t = np.ones(L, dtype=np.float64)
    for l in range(L):
        pos_probs = P_cal[Y_cal[:, l] == 1, l]
        t[l] = conformal_quantile(pos_probs if len(pos_probs) else np.array([1.0]), alpha, side="lower")
    return t

def labelwise_predict_sets(P, t_vec):
    return [np.where(p >= t_vec)[0] for p in P]

# ---------------------------------------------------------------------
# 3) Metrics to check coverage & size
# ---------------------------------------------------------------------
def examplewise_coverage(sets, Y):
    """Fraction of examples where all true labels are included in the set."""
    """Counts cases where no outage is present as good"""
    good = 0
    for S, y in zip(sets, Y):
        S = set(S.tolist()) if hasattr(S, "tolist") else set(S)
        ok = True
        for l in np.where(y == 1)[0]:
            if l not in S:
                ok = False; break
        good += int(ok)
    return good / len(Y)

def avg_set_size(sets):
    return float(np.mean([len(S) for S in sets]))
