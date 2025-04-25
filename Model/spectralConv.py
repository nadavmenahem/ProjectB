import torch
import torch.nn as nn

class SpectralConvolution(nn.Module):
    def __init__(self, in_features, out_features, Lambda, H):
        super(SpectralConvolution, self).__init__()
        self.in_features = in_features  # K
        self.out_features = out_features  # G
        self.H = H  # Polynomial order
        self.Lambda = Lambda  # (N,) — eigenvalues of normalized adjacency

        # Learnable weights: shape (K, G, H + 1)
        self.weights = nn.Parameter(torch.randn(in_features, out_features, H + 1))

    def forward(self, X_hat):  
        # X_hat: (B, T, K, N)
        B, T, K, N = X_hat.shape
        Y_hat = torch.zeros(B, T, self.out_features, N, device=X_hat.device)

        for g in range(self.out_features):
            for k in range(K):
                x_hat_k = X_hat[:, :, k, :]  # (B, T, N)
                conv_sum = torch.zeros_like(x_hat_k)

                for h in range(self.H + 1):
                    lambda_h = (self.Lambda ** h).view(1, 1, -1)  # shape (1, 1, N)
                    weight = self.weights[k, g, h]
                    conv_sum += weight * x_hat_k * lambda_h  # broadcasting
                Y_hat[:, :, g, :] += conv_sum

        return Y_hat
