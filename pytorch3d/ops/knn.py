import torch


def knn_points(x: torch.Tensor, y: torch.Tensor, K: int = 1, return_nn: bool = False):
    """
    Lightweight CPU-only implementation of pytorch3d.ops.knn_points.

    只实现本项目需要的功能，API 和返回
    形状尽量与原版 pytorch3d 对齐：
      - x: (B, N, D)
      - y: (B, M, D)
      - 返回:
          sq_distances: (B, N, K)
          idx: (B, N, K)
          nn: (B, N, K, D) 或 None
    """

    assert x.ndim == 3 and y.ndim == 3, "x, y should be (B, N, D) and (B, M, D)"
    B, N, D = x.shape
    _, M, Dy = y.shape
    assert Dy == D, "x and y must have same feature dimension"

    # pairwise squared distance: (B, N, M)
    diff = x.unsqueeze(2) - y.unsqueeze(1)  # (B, N, M, D)
    dist2 = (diff ** 2).sum(-1)  # (B, N, M)

    sq_distances, idx = torch.topk(dist2, k=K, dim=-1, largest=False, sorted=True)

    if return_nn:
        # gather neighbor points from y
        idx_expanded = idx.unsqueeze(-1).expand(-1, -1, -1, D)  # (B, N, K, D)
        y_expanded = y.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, M, D)
        nn = torch.gather(y_expanded, 2, idx_expanded)  # (B, N, K, D)
    else:
        nn = None

    return sq_distances, idx, nn
