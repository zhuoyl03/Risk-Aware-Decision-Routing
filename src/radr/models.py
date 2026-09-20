"""Small cost-distribution models over frozen, observable image features."""

import copy

import numpy as np
import torch
from torch import nn


class LossMixture(nn.Module):
    """A mixture of cost predictors, not a sparse language-model MoE.

    Each component predicts P(loss | image, action); a learned soft gate mixes
    them. All action outcomes are supervised on router-fitting data. Labels and
    severity do not enter the forward pass.
    """
    def __init__(self, input_dim, actions, components=3, width=32):
        super().__init__()
        self.actions = actions
        self.components = nn.ModuleList([
            nn.Sequential(nn.Linear(input_dim, width), nn.Tanh(),
                          nn.Linear(width, actions*4)) for _ in range(components)])
        self.gate = nn.Linear(input_dim, components)

    def forward(self, x):
        probabilities = torch.stack([
            head(x).reshape(-1, self.actions, 4).softmax(-1) for head in self.components], dim=1)
        weights = self.gate(x).softmax(-1)
        return (probabilities*weights[:, :, None, None]).sum(1)


def fit_loss_model(x, targets, val_x, val_targets, seed, components=3, epochs=250):
    torch.manual_seed(seed)
    model = LossMixture(x.shape[1], targets.shape[1], components)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.005, weight_decay=.01)
    x, val_x = [torch.as_tensor(a, dtype=torch.float32) for a in (x, val_x)]
    targets, val_targets = [torch.as_tensor(a, dtype=torch.long) for a in (targets, val_targets)]
    history, best, stale, best_state = [], float("inf"), 0, None
    for epoch in range(epochs):
        model.train()
        p = model(x)
        loss = -p.gather(2, targets[..., None]).clamp_min(1e-9).log().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val = float(-model(val_x).gather(2, val_targets[..., None]).clamp_min(1e-9).log().mean())
        history.append(dict(epoch=epoch+1, train_nll=float(loss.detach()), calibration_nll=val))
        if val < best-1e-5:
            best, stale, best_state = val, 0, copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= 25:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, history


def predict_loss(model, x):
    with torch.no_grad():
        return model(torch.as_tensor(x, dtype=torch.float32)).numpy()


def loss_targets(predictions, labels, severity):
    """Cost-bin supervision, used only in fitting/calibration."""
    return np.where(predictions != labels[:, None], severity[:, None]+1, 0).astype(np.int64)
