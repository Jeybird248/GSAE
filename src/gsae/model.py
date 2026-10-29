from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class GraphSparseAutoencoder(nn.Module):
    def __init__(self, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.encoder = nn.Linear(self.hidden_dim, self.latent_dim, bias=False)
        self.decoder = nn.Linear(self.latent_dim, self.hidden_dim, bias=False)
        self.probe = nn.Linear(self.latent_dim, 2)
        nn.init.kaiming_uniform_(self.encoder.weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.decoder.weight, a=5**0.5)
        self.normalize_decoder_columns()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.encoder(x))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        recon = self.decoder(z)
        logits = self.probe(z)
        return z, recon, logits

    @torch.no_grad()
    def normalize_decoder_columns(self, eps: float = 1e-8) -> None:
        weight = self.decoder.weight.data
        weight.div_(weight.norm(dim=0, keepdim=True).clamp_min(eps))


def graph_dirichlet(decoder_weight: torch.Tensor, laplacian: torch.Tensor) -> torch.Tensor:
    """Return decoder-column Dirichlet energy trace(W^T L W), without averaging."""

    laplacian = laplacian.to(decoder_weight.device)
    if laplacian.is_sparse:
        lw = torch.sparse.mm(laplacian, decoder_weight)
    else:
        lw = laplacian @ decoder_weight
    return (decoder_weight * lw).sum()


def gsae_loss(
    model: GraphSparseAutoencoder,
    x: torch.Tensor,
    y: torch.Tensor,
    laplacian: torch.Tensor,
    lambda_sparse: float,
    lambda_sup: float,
    lambda_graph: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    z, recon, logits = model(x)
    # Average over examples and sum over features.
    rec = (recon - x).pow(2).sum(dim=1).mean()
    sparse = z.abs().sum(dim=1).mean()
    sup = F.cross_entropy(logits, y.long())
    graph = graph_dirichlet(model.decoder.weight, laplacian)
    total = rec + lambda_sparse * sparse + lambda_sup * sup + lambda_graph * graph
    return total, {
        "loss": float(total.detach().cpu()),
        "rec": float(rec.detach().cpu()),
        "sparse": float(sparse.detach().cpu()),
        "sup": float(sup.detach().cpu()),
        "graph": float(graph.detach().cpu()),
    }
