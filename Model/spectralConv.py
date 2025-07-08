import torch
import torch.nn as nn

class SpectralConvolution(nn.Module):
    def __init__(self, time_samples, in_features, out_features, Lambda, H):
        super().__init__()
        self.T, self.K, self.G, self.H = time_samples, in_features, out_features, H
        # Λ is a 1-D tensor of shape (N,)
        self.register_buffer('Lambda', Lambda)

        # precompute Λ^0…Λ^H → shape (H+1, N)
        h_idx = torch.arange(H+1, device=Lambda.device).unsqueeze(1)  # (H+1,1)

        # learnable coefficients w[t,k,g,h]
        self.weights = nn.Parameter(torch.randn(self.T,
                                                self.K,
                                                self.G,
                                                self.H+1))

    def forward(self, X_hat):
        """
        X_hat: (B, T, K, N)
        returns Y: (B, G, N)
        """
        B, T, K, N = X_hat.shape
        assert T == self.T and K == self.K and N == self.Lambda.size(0)

        # 1) Λ-powers  ------ shape:  (N, H+1)
        lambda_powers = self.Lambda.unsqueeze(1).pow(
            torch.arange(self.H + 1, device=X_hat.device))

        # 2) single contraction – multiply & add in one shot
        #
        #   btkn  = X_hat                       (B, T, K, N)
        #   nh    = Λ_powers                    (N, H+1)
        #   tkgh  = weights                     (T, K, G, H+1)
        #
        #   result indices kept:  b  g  n
        #
        Y = torch.einsum('btkn,nh,tkgh->bgn', X_hat, lambda_powers, self.weights)

        return Y
