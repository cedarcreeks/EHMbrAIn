"""F8/L5 (prereg-v8): is the AI RUL advantage architecture-robust?

Three sequence architectures under IDENTICAL, modest, equal effort (same
inputs/data/cap/epochs/seeds/fixed hyperparameters -- a fairness comparison,
not a per-architecture tuning campaign): GRU (recurrent), TCN (dilated causal
convolutions), Transformer (self-attention). Each vs the tuned traditional
baseline (90%-life RMSE 1981 cycles). If all beat it, the advantage is a
property of learning from sequences, not of one architecture.

Foreground (MPS). Output: data/processed/f8/arch_verdict.json
Usage: uv run python scripts/f8_l5_arch.py
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ehmbrain.ai.data import load_fleet_features, normalization
from ehmbrain.ai.models import predict_torch, train_torch

REPO_ROOT = Path(__file__).resolve().parents[1]
F5 = REPO_ROOT / 'data' / 'processed' / 'f5'
OUT = REPO_ROOT / 'data' / 'processed' / 'f8'
EVAL_FRACS = (0.5, 0.7, 0.9)
RUL_CAP = 12.0
SEQ, DS, HID = 64, 20, 64
TRAD_90 = 1981.0        # tuned traditional 90%-life RMSE (F5 verdict)


class TCN(nn.Module):
    """Dilated causal temporal convolutional network."""
    def __init__(self, ch=4, hid=HID, levels=4):
        super().__init__()
        layers, c_in = [], ch
        for i in range(levels):
            d = 2 ** i
            layers += [nn.Conv1d(c_in, hid, 3, padding=2 * d, dilation=d),
                       nn.GELU()]
            c_in = hid
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(hid, 1)

    def forward(self, x):                       # x: (N, T, C)
        h = self.net(x.transpose(1, 2))         # (N, hid, T')
        return self.head(h.mean(dim=2)).squeeze(-1)


class TransformerRUL(nn.Module):
    def __init__(self, ch=4, hid=HID, heads=4, layers=2):
        super().__init__()
        self.proj = nn.Linear(ch, hid)
        self.pos = nn.Parameter(torch.randn(1, SEQ, hid) * 0.02)
        enc = nn.TransformerEncoderLayer(hid, heads, hid * 2, batch_first=True,
                                         activation='gelu', dropout=0.0)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(hid, 1)

    def forward(self, x):
        h = self.proj(x) + self.pos[:, :x.shape[1]]
        return self.head(self.enc(h).mean(dim=1)).squeeze(-1)


class GRURUL(nn.Module):
    def __init__(self, ch=4, hid=HID, layers=2):
        super().__init__()
        self.gru = nn.GRU(ch, hid, layers, batch_first=True)
        self.head = nn.Linear(hid, 1)

    def forward(self, x):
        h, _ = self.gru(x)
        return self.head(h[:, -1]).squeeze(-1)


def build_sequences(fleet, norm, ids, rng, per=40):
    mu, sd = norm
    X, y = [], []
    for eid in ids:
        v = fleet[eid]
        Fn = ((v['F'] - mu) / sd)[::DS]
        for cut in rng.integers(SEQ * DS, v['life'], size=per):
            i = int(cut) // DS
            if i < SEQ:
                continue
            X.append(Fn[i - SEQ:i])
            y.append(min((v['life'] - int(cut)) / 1000.0, RUL_CAP))
    return np.array(X, np.float32), np.array(y, np.float32)


def evaluate(net, fleet, norm, test_ids):
    mu, sd = norm
    cpu = torch.device('cpu')
    errs = {f: [] for f in EVAL_FRACS}
    for eid in test_ids:
        v = fleet[eid]
        Fn = ((v['F'] - mu) / sd)[::DS]
        for f in EVAL_FRACS:
            cut = int(f * v['life']); i = cut // DS
            if i < SEQ:
                continue
            pred = float(predict_torch(net, Fn[i - SEQ:i][None], dev=cpu)[0])
            errs[f].append((min(pred, RUL_CAP) - min((v['life'] - cut) / 1000.0, RUL_CAP)) * 1000)
    return {f: float(np.sqrt(np.mean(np.square(errs[f])))) for f in EVAL_FRACS}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fleet = load_fleet_features()
    norm = normalization(fleet)
    train_ids = sorted(e for e, v in fleet.items() if v['split'] == 'train')
    test_ids = sorted(e for e, v in fleet.items() if v['split'] == 'test')

    arch = {'GRU': GRURUL, 'TCN': TCN, 'Transformer': TransformerRUL}
    results = {}
    for name, cls in arch.items():
        rmses = []
        for seed in (0, 1, 2):
            rng = np.random.default_rng(100 + seed)
            Xtr, ytr = build_sequences(fleet, norm, train_ids, rng)
            net = train_torch(cls(), Xtr, ytr, epochs=30, lr=1e-3, seed=seed)
            rmses.append(evaluate(net, fleet, norm, test_ids))
        mean = {f: float(np.mean([r[f] for r in rmses])) for f in EVAL_FRACS}
        results[name] = {'rmse_by_frac': mean}
        print(f"{name:12s} RMSE 50/70/90: "
              f"{mean[0.5]:.0f}/{mean[0.7]:.0f}/{mean[0.9]:.0f}", flush=True)

    beats = {n: bool(results[n]['rmse_by_frac'][0.9] < TRAD_90) for n in arch}
    verdict = {
        'traditional_90_rmse': TRAD_90,
        'architectures': {n: {'rmse_90': results[n]['rmse_by_frac'][0.9],
                              'beats_traditional': beats[n],
                              'rmse_by_frac': {str(k): v for k, v in
                                               results[n]['rmse_by_frac'].items()}}
                          for n in arch},
        'best_architecture': min(arch, key=lambda n: results[n]['rmse_by_frac'][0.9]),
        'H5L.1_architecture_robust': {'confirmed': bool(all(beats.values()))},
    }
    (OUT / 'arch_verdict.json').write_text(json.dumps(verdict, indent=2))
    print(f"all beat traditional (1981) @90%: {all(beats.values())} "
          f"-> H5L.1 {'CONFIRMED' if all(beats.values()) else 'refuted'}")
    print(f"best: {verdict['best_architecture']}")


if __name__ == '__main__':
    main()
