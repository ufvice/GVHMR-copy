"""
Lightweight re-implementation of a small subset of `pytorch3d.transforms`
used in this repo, so that inference can run without installing the full
PyTorch3D package.

只实现本项目用到的少量函数，基于标准 SO(3)/四元数数学公式，
不会引入新的第三方依赖。
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def _safe_norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return torch.clamp(x.norm(dim=dim, keepdim=True), min=eps)


def axis_angle_to_matrix(angles: torch.Tensor) -> torch.Tensor:
    """
    Args:
        angles: (..., 3) axis-angle, where direction is axis and norm is angle (rad).
    Returns:
        R: (..., 3, 3)
    """
    orig_shape = angles.shape
    angles = angles.view(-1, 3)
    angle = _safe_norm(angles, dim=-1)  # (N,1)
    axis = angles / angle

    x, y, z = axis.unbind(-1)
    ca = torch.cos(angle).view(-1)
    sa = torch.sin(angle).view(-1)
    one_c = 1.0 - ca

    # Rodrigues' rotation formula
    r00 = ca + x * x * one_c
    r01 = x * y * one_c - z * sa
    r02 = x * z * one_c + y * sa

    r10 = y * x * one_c + z * sa
    r11 = ca + y * y * one_c
    r12 = y * z * one_c - x * sa

    r20 = z * x * one_c - y * sa
    r21 = z * y * one_c + x * sa
    r22 = ca + z * z * one_c

    R = torch.stack(
        [
            torch.stack([r00, r01, r02], dim=-1),
            torch.stack([r10, r11, r12], dim=-1),
            torch.stack([r20, r21, r22], dim=-1),
        ],
        dim=-2,
    )
    R = R.view(orig_shape[:-1] + (3, 3))

    # 对于非常小的角度，直接使用单位矩阵以避免数值不稳定
    small_angle = angle.view(-1) < 1e-6
    if small_angle.any():
        I = torch.eye(3, device=angles.device, dtype=angles.dtype)
        R[small_angle] = I

    return R


def matrix_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    """
    Args:
        R: (..., 3, 3)
    Returns:
        axis-angle: (..., 3)
    """
    orig_shape = R.shape
    R = R.view(-1, 3, 3)
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    theta = torch.acos(cos_theta)  # (N,)

    sin_theta = torch.sin(theta)
    axis = torch.zeros_like(R[:, 0])
    denom = 2.0 * _safe_norm(sin_theta.unsqueeze(-1))  # 只是保持形状一致

    axis[:, 0] = (R[:, 2, 1] - R[:, 1, 2]) / (2.0 * sin_theta + 1e-8)
    axis[:, 1] = (R[:, 0, 2] - R[:, 2, 0]) / (2.0 * sin_theta + 1e-8)
    axis[:, 2] = (R[:, 1, 0] - R[:, 0, 1]) / (2.0 * sin_theta + 1e-8)

    # 对于小角度，近似为零旋转
    small = theta < 1e-6
    if small.any():
        axis[small] = torch.tensor([1.0, 0.0, 0.0], device=R.device, dtype=R.dtype)

    aa = axis * theta.unsqueeze(-1)
    aa = aa.view(orig_shape[:-2] + (3,))
    return aa


def matrix_to_rotation_6d(R: torch.Tensor) -> torch.Tensor:
    """
    Args:
        R: (..., 3, 3)
    Returns:
        rot6d: (..., 6) using first two columns.
    """
    return R[..., :, :2].reshape(R.shape[:-2] + (6,))


def rotation_6d_to_matrix(rot_6d: torch.Tensor) -> torch.Tensor:
    """
    Args:
        rot_6d: (..., 6)
    Returns:
        R: (..., 3, 3)
    """
    a1 = rot_6d[..., 0:3]
    a2 = rot_6d[..., 3:6]

    b1 = F.normalize(a1, dim=-1)
    proj = (b1 * a2).sum(dim=-1, keepdim=True)
    b2 = F.normalize(a2 - proj * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)

    R = torch.stack([b1, b2, b3], dim=-2)
    return R


def matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix to quaternion (w, x, y, z).
    Args:
        R: (..., 3, 3)
    Returns:
        q: (..., 4)
    """
    R = R.float()
    orig_shape = R.shape[:-2]
    R_flat = R.reshape(-1, 3, 3)
    q = torch.zeros(R_flat.shape[0], 4, device=R.device, dtype=R.dtype)

    trace = R_flat[:, 0, 0] + R_flat[:, 1, 1] + R_flat[:, 2, 2]
    cond = trace > 0

    # trace > 0
    t = torch.sqrt(1.0 + trace[cond]) * 2.0
    q[cond, 0] = 0.25 * t
    q[cond, 1] = (R_flat[cond, 2, 1] - R_flat[cond, 1, 2]) / t
    q[cond, 2] = (R_flat[cond, 0, 2] - R_flat[cond, 2, 0]) / t
    q[cond, 3] = (R_flat[cond, 1, 0] - R_flat[cond, 0, 1]) / t

    # trace <= 0，分三种情况
    cond1 = ~cond & (R_flat[:, 0, 0] > R_flat[:, 1, 1]) & (R_flat[:, 0, 0] > R_flat[:, 2, 2])
    t = torch.sqrt(1.0 + R_flat[cond1, 0, 0] - R_flat[cond1, 1, 1] - R_flat[cond1, 2, 2]) * 2.0
    q[cond1, 0] = (R_flat[cond1, 2, 1] - R_flat[cond1, 1, 2]) / t
    q[cond1, 1] = 0.25 * t
    q[cond1, 2] = (R_flat[cond1, 0, 1] + R_flat[cond1, 1, 0]) / t
    q[cond1, 3] = (R_flat[cond1, 0, 2] + R_flat[cond1, 2, 0]) / t

    cond2 = ~cond & ~cond1 & (R_flat[:, 1, 1] > R_flat[:, 2, 2])
    t = torch.sqrt(1.0 - R_flat[cond2, 0, 0] + R_flat[cond2, 1, 1] - R_flat[cond2, 2, 2]) * 2.0
    q[cond2, 0] = (R_flat[cond2, 0, 2] - R_flat[cond2, 2, 0]) / t
    q[cond2, 1] = (R_flat[cond2, 0, 1] + R_flat[cond2, 1, 0]) / t
    q[cond2, 2] = 0.25 * t
    q[cond2, 3] = (R_flat[cond2, 1, 2] + R_flat[cond2, 2, 1]) / t

    cond3 = ~cond & ~cond1 & ~cond2
    t = torch.sqrt(1.0 - R_flat[cond3, 0, 0] - R_flat[cond3, 1, 1] + R_flat[cond3, 2, 2]) * 2.0
    q[cond3, 0] = (R_flat[cond3, 1, 0] - R_flat[cond3, 0, 1]) / t
    q[cond3, 1] = (R_flat[cond3, 0, 2] + R_flat[cond3, 2, 0]) / t
    q[cond3, 2] = (R_flat[cond3, 1, 2] + R_flat[cond3, 2, 1]) / t
    q[cond3, 3] = 0.25 * t

    q = q.view(orig_shape + (4,))
    return q


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Args:
        quaternions: (..., 4) in (w, x, y, z) format.
    Returns:
        R: (..., 3, 3)
    """
    q = F.normalize(quaternions, dim=-1)
    w, x, y, z = q.unbind(-1)

    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z

    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    m00 = ww + xx - yy - zz
    m01 = 2 * (xy - wz)
    m02 = 2 * (xz + wy)

    m10 = 2 * (xy + wz)
    m11 = ww - xx + yy - zz
    m12 = 2 * (yz - wx)

    m20 = 2 * (xz - wy)
    m21 = 2 * (yz + wx)
    m22 = ww - xx - yy + zz

    R = torch.stack(
        [
            torch.stack([m00, m01, m02], dim=-1),
            torch.stack([m10, m11, m12], dim=-1),
            torch.stack([m20, m21, m22], dim=-1),
        ],
        dim=-2,
    )
    return R


def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Args:
        quaternions: (..., 4) in (w, x, y, z) format.
    Returns:
        axis-angle: (..., 3)
    """
    q = F.normalize(quaternions, dim=-1)
    w, v = q[..., 0], q[..., 1:]
    angle = 2.0 * torch.acos(torch.clamp(w, -1.0, 1.0))
    sin_half = _safe_norm(v, dim=-1).squeeze(-1)

    axis = torch.zeros_like(v)
    mask = sin_half > 1e-6
    axis[mask] = v[mask] / sin_half[mask].unsqueeze(-1)
    axis[~mask] = torch.tensor([1.0, 0.0, 0.0], device=q.device, dtype=q.dtype)

    return axis * angle.unsqueeze(-1)


def so3_exp_map(log_rot: torch.Tensor) -> torch.Tensor:
    """
    SO(3) exponential map. Alias of axis_angle_to_matrix.
    """
    return axis_angle_to_matrix(log_rot)


def so3_log_map(R: torch.Tensor) -> torch.Tensor:
    """
    SO(3) log map. Alias of matrix_to_axis_angle.
    """
    return matrix_to_axis_angle(R)


def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Args:
        euler_angles: (..., 3) in radians.
        convention: string of 3 chars from {X,Y,Z}, e.g. "XYZ", "YXZ".
    Returns:
        R: (..., 3, 3)
    """
    assert len(convention) == 3
    euler_angles = euler_angles.view(-1, 3)

    def _single_axis_matrix(angle: torch.Tensor, axis: str) -> torch.Tensor:
        zero = torch.zeros_like(angle)
        one = torch.ones_like(angle)
        c = torch.cos(angle)
        s = torch.sin(angle)

        if axis == "X":
            R = torch.stack(
                [
                    torch.stack([one, zero, zero], dim=-1),
                    torch.stack([zero, c, -s], dim=-1),
                    torch.stack([zero, s, c], dim=-1),
                ],
                dim=-2,
            )
        elif axis == "Y":
            R = torch.stack(
                [
                    torch.stack([c, zero, s], dim=-1),
                    torch.stack([zero, one, zero], dim=-1),
                    torch.stack([-s, zero, c], dim=-1),
                ],
                dim=-2,
            )
        elif axis == "Z":
            R = torch.stack(
                [
                    torch.stack([c, -s, zero], dim=-1),
                    torch.stack([s, c, zero], dim=-1),
                    torch.stack([zero, zero, one], dim=-1),
                ],
                dim=-2,
            )
        else:
            raise ValueError(f"Invalid axis: {axis}")
        return R

    R = torch.eye(3, device=euler_angles.device, dtype=euler_angles.dtype).unsqueeze(0).repeat(
        euler_angles.shape[0], 1, 1
    )
    for idx, axis in enumerate(convention):
        R_axis = _single_axis_matrix(euler_angles[:, idx], axis)
        R = R @ R_axis

    R = R.view(euler_angles.shape[0], 3, 3)
    return R


__all__ = [
    "axis_angle_to_matrix",
    "matrix_to_axis_angle",
    "matrix_to_rotation_6d",
    "rotation_6d_to_matrix",
    "matrix_to_quaternion",
    "quaternion_to_matrix",
    "quaternion_to_axis_angle",
    "so3_exp_map",
    "so3_log_map",
    "euler_angles_to_matrix",
]

