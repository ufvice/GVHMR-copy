from .knn import knn_points

__all__ = ["knn_points"]

*** Add File: pytorch3d/ops/knn.py
import torch


def knn_points(x: torch.Tensor, y: torch.Tensor, K: int = 1, return_nn: bool = False):
    """
    Lightweight CPU-only implementation of knn_points used in this project.

    Args:
        x: (B, N, D)
        y: (B, M, D)
        K: number of nearest neighbors, default 1
        return_nn: whether to return the neighbor coordinates

    Returns:
        sq_distances: (B, N, K)
        idx: (B, N, K) indices in y
        nn: (B, N, K, D) neighbor points (or None if return_nn=False)
    """
    assert x.ndim == 3 and y.ndim == 3, "x, y should be (B, N, D) and (B, M, D)"
    B, N, D = x.shape
    _, M, Dy = y.shape
    assert Dy == D, "x and y must have same feature dimension"

    # pairwise squared distance
    # x -> (B, N, 1, D), y -> (B, 1, M, D)
    diff = x.unsqueeze(2) - y.unsqueeze(1)  # (B, N, M, D)
    dist2 = (diff**2).sum(-1)  # (B, N, M)

    sq_distances, idx = torch.topk(dist2, k=K, dim=-1, largest=False, sorted=True)

    if return_nn:
        # gather neighbor points from y
        idx_expanded = idx.unsqueeze(-1).expand(-1, -1, -1, D)  # (B, N, K, D)
        y_expanded = y.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, M, D)
        nn = torch.gather(y_expanded, 2, idx_expanded)  # (B, N, K, D)
    else:
        nn = None

    return sq_distances, idx, nn

