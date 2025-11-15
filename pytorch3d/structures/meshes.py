from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import torch


@dataclass
class Meshes:
    """
    Minimal placeholder for pytorch3d.structures.Meshes.

    只存储 verts / faces / textures，供上层可视化代码访问。
    """

    verts: torch.Tensor
    faces: torch.Tensor
    textures: Optional[Any] = None

    @property
    def device(self):
        return self.verts.device


def join_meshes_as_scene(meshes_list: List[Meshes]) -> Meshes:
    """
    Simple utility that concatenates multiple Meshes into one.
    仅用于兼容接口，不追求与原版完全一致。
    """

    if len(meshes_list) == 0:
        raise ValueError("meshes_list must not be empty")

    verts = torch.cat([m.verts for m in meshes_list], dim=1)
    faces = torch.cat([m.faces for m in meshes_list], dim=1)
    textures = meshes_list[0].textures
    return Meshes(verts=verts, faces=faces, textures=textures)

