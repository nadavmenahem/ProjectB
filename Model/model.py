#===================IMPORTS===================
import torch
import torch.nn as nn



#====================MODEL====================
class SpectralConvLayer(nn.Module):
    def __init__(self, in_features, out_features, Lambda, H=1):
        super().__init__()
        self.H = H
        self.in_features = in_features
        self.out_features = out_features

        # Lambda is the eigenvalue matrix (diagonal), shape: [N]
        self.register_buffer("Lambda", Lambda)

        # Learnable parameters: one set of weights per hop degree h
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(in_features, out_features)) for _ in range(H + 1)
        ])


    def forward(self, X_hat):
        # X_hat shape: [batch, N, in_features]
        batch_size, N, _ = X_hat.shape
        Y_hat = 0

        for h in range(self.H + 1):
            Λh = self.Lambda ** h  # shape: [N]
            Λh_diag = torch.diag(Λh)  # [N, N]

            # Apply Λ^h to signal
            filtered = torch.matmul(Λh_diag, X_hat)  # [batch, N, in_features]
            Y_hat += torch.matmul(filtered, self.weights[h])  # [batch, N, out_features]

        return Y_hat


class SpectralGCN(nn.Module):
    def __init__(self, V, Lambda, in_features, hidden_features, out_features, H=1):
        super().__init__()
        self.V = V           # Eigenvectors, shape [N, N]
        self.VT = V.T        # GFT matrix
        self.Lambda = Lambda # Eigenvalues, shape [N]

        self.conv1 = SpectralConvLayer(in_features, hidden_features, Lambda, H)
        self.fc = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        # x shape: [batch, time, N]
        batch_size, T, N = x.shape

        # Transpose to [batch, N, time]
        x = x.transpose(1, 2)

        # Apply GFT to each node’s signal
        x_hat = torch.matmul(self.VT, x)  # shape: [batch, N, time]

        # GCN expects last dim to be features → [batch, N, time]
        out_hat = self.conv1(x_hat)

        # Aggregate (e.g., mean over nodes or time), or flatten
        out = out_hat.mean(dim=1)  # [batch, hidden_features]
        return torch.sigmoid(self.fc(out))  # Binary classifier per line
