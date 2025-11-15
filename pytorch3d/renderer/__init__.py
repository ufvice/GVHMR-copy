"""
Lightweight stub of `pytorch3d.renderer` so that code which imports
rendering utilities can run without installing the real PyTorch3D.

这里只提供项目中用到的一些类/函数的空实现，主要目的是
避免 ImportError；如果真正调用渲染，则只返回占位结果。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple

import torch


@dataclass
class PerspectiveCameras:
    """
    Minimal placeholder for pytorch3d.renderer.PerspectiveCameras.

    只保存传入的参数，不做真正相机建模。
    """

    device: Any
    R: Optional[torch.Tensor] = None
    T: Optional[torch.Tensor] = None
    K: Optional[torch.Tensor] = None
    image_size: Optional[Sequence[Tuple[int, int]]] = None
    in_ndc: bool = False


@dataclass
class PointLights:
    """Placeholder for pytorch3d.renderer.PointLights."""

    device: Any
    location: Any = None


@dataclass
class Materials:
    """Placeholder for pytorch3d.renderer.Materials."""

    device: Any
    specular_color: Any = None
    shininess: Any = None


@dataclass
class RasterizationSettings:
    """Placeholder for pytorch3d.renderer.RasterizationSettings."""

    image_size: Any
    blur_radius: float = 0.0
    bin_size: Optional[int] = None


class MeshRasterizer:
    """Placeholder for pytorch3d.renderer.MeshRasterizer."""

    def __init__(self, raster_settings: RasterizationSettings):
        self.raster_settings = raster_settings

    def __call__(self, meshes, **kwargs):  # pragma: no cover - placeholder
        # 返回一个空的“图像”张量占位，避免后续代码崩溃
        H, W = 256, 256
        if isinstance(self.raster_settings.image_size, (list, tuple)):
            H, W = self.raster_settings.image_size[0]
        return torch.zeros(1, H, W, 4, device=getattr(meshes, "device", "cpu"))


class SoftPhongShader:
    """Placeholder for pytorch3d.renderer.SoftPhongShader."""

    def __init__(self, device=None, cameras=None, lights=None, materials=None, **kwargs):
        self.device = device
        self.cameras = cameras
        self.lights = lights
        self.materials = materials

    def __call__(self, fragments, meshes, **kwargs):  # pragma: no cover - placeholder
        # 直接忽略输入，返回一个占位张量
        return torch.zeros_like(fragments)


class MeshRenderer:
    """
    Placeholder for pytorch3d.renderer.MeshRenderer.

    被调用时返回一个零图像张量，形状 (1, H, W, 4)。
    """

    def __init__(self, rasterizer: MeshRasterizer, shader: SoftPhongShader):
        self.rasterizer = rasterizer
        self.shader = shader

    def __call__(self, meshes, materials=None, cameras=None, lights=None, **kwargs):  # pragma: no cover
        fragments = self.rasterizer(meshes)
        return fragments


def look_at_rotation(eye, at=None, up=None, device=None):
    """
    Simplified placeholder for pytorch3d.renderer.cameras.look_at_rotation.

    返回单位旋转矩阵，主要用于避免导入错误。
    """
    eye = torch.as_tensor(eye, device=device, dtype=torch.float32)
    batch = eye.shape[0] if eye.ndim > 1 else 1
    return torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0).repeat(batch, 1, 1)


__all__ = [
    "PerspectiveCameras",
    "TexturesVertex",
    "PointLights",
    "Materials",
    "RasterizationSettings",
    "MeshRenderer",
    "MeshRasterizer",
    "SoftPhongShader",
    "look_at_rotation",
]


class TexturesVertex:
    """
    Very small stub for pytorch3d.renderer.TexturesVertex.

    只保存顶点颜色张量，方便后续代码访问属性。
    """

    def __init__(self, verts_features: torch.Tensor):
        self.verts_features = verts_features


