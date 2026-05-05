import torch

from gsae.graph import build_threshold_cosine_laplacian, graph_density


def test_threshold_cosine_laplacian_shape_and_energy():
    x = torch.randn(12, 8)
    x[:, 1] = x[:, 0] + 0.01 * torch.randn(12)
    lap = build_threshold_cosine_laplacian(x, threshold=0.5, chunk_size=4)
    assert lap.is_sparse
    assert lap.shape == (8, 8)
    assert graph_density(lap) > 0
    f = torch.randn(8, 3)
    energy = (f * torch.sparse.mm(lap, f)).sum()
    assert energy.item() >= -1e-5
