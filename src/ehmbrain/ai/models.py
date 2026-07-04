"""AI-suite models (phase F4). All torch models run on the best available
device: Apple-silicon MPS on the development machine (norm N1), CUDA or CPU
elsewhere.
"""

import numpy as np
import torch
import torch.nn as nn


def device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


class WindowAE(nn.Module):
    """Dense autoencoder over flattened deviation windows (anomaly detection).
    Trained on acute-free data; reconstruction error is the anomaly score."""

    def __init__(self, w=50, ch=4, latent=8):
        super().__init__()
        d = w * ch
        self.enc = nn.Sequential(nn.Linear(d, 96), nn.GELU(),
                                 nn.Linear(96, 32), nn.GELU(),
                                 nn.Linear(32, latent))
        self.dec = nn.Sequential(nn.Linear(latent, 32), nn.GELU(),
                                 nn.Linear(32, 96), nn.GELU(),
                                 nn.Linear(96, d))

    def forward(self, x):
        b = x.shape[0]
        flat = x.reshape(b, -1)
        return self.dec(self.enc(flat)).reshape(x.shape)


class RULNet(nn.Module):
    """GRU regressor: deviation sequence -> remaining useful life (kcycles)."""

    def __init__(self, ch=4, hidden=64, layers=2):
        super().__init__()
        self.gru = nn.GRU(ch, hidden, layers, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.GELU(),
                                  nn.Linear(32, 1))

    def forward(self, x):
        h, _ = self.gru(x)
        return self.head(h[:, -1]).squeeze(-1)


def train_torch(model, X, y=None, epochs=30, bs=256, lr=1e-3, dev=None,
                loss_fn=None, verbose=False, seed=0):
    """Minimal training loop. y=None trains an autoencoder (target = input)."""
    torch.manual_seed(seed)
    dev = dev or device()
    model = model.to(dev)
    X = torch.as_tensor(X, dtype=torch.float32)
    if y is None:
        Y = X
    else:
        y_arr = np.asarray(y)
        Y = torch.as_tensor(y_arr, dtype=torch.long
                            if np.issubdtype(y_arr.dtype, np.integer)
                            else torch.float32)
    loss_fn = loss_fn or nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    n = len(X)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb, yb = X[idx].to(dev), Y[idx].to(dev)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb if y is not None else xb)
            loss.backward()
            opt.step()
            tot += float(loss.detach().cpu()) * len(idx)
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f'  epoch {ep:3d}  loss {tot / n:.5f}', flush=True)
    return model


@torch.no_grad()
def predict_torch(model, X, bs=4096, dev=None):
    dev = dev or device()
    model = model.to(dev).eval()
    X = torch.as_tensor(X, dtype=torch.float32)
    outs = []
    for i in range(0, len(X), bs):
        outs.append(model(X[i:i + bs].to(dev)).cpu())
    return torch.cat(outs).numpy()


def ae_scores(model, X, dev=None):
    """Per-window reconstruction error (mean squared, over window and channels)."""
    rec = predict_torch(model, X, dev=dev)
    return ((rec - X) ** 2).mean(axis=(1, 2))
