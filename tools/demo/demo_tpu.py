import argparse
from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader

import hydra
from hydra import initialize_config_module, compose

from hmr4d.configs import register_store_gvhmr
from hmr4d.utils.pylogger import Log

from hmr4d.dataset.phase1_demo_tpu import Phase1DemoDatasetTPU
from hmr4d.model.gvhmr.gvhmr_pl_demo_tpu import DemoPLTPU


def parse_args_to_cfg_tpu():
    """
    TPU demo 参数解析 + Hydra cfg 组装。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_root", type=str, required=True)
    parser.add_argument("--bbox_pt", type=str, required=True)
    parser.add_argument("--output_labels_pt", type=str, required=True)
    parser.add_argument("--static_cam", action="store_true")
    parser.add_argument("--use_dpvo", action="store_true")
    parser.add_argument("--f_mm", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--ckpt_path", type=str, default=None)
    args = parser.parse_args()

    video_root = Path(args.video_root)
    bbox_pt = Path(args.bbox_pt)
    output_labels_pt = Path(args.output_labels_pt)

    assert video_root.exists(), f"video_root not found: {video_root}"
    assert bbox_pt.exists(), f"bbox_pt not found: {bbox_pt}"
    output_labels_pt.parent.mkdir(parents=True, exist_ok=True)

    # Hydra cfg：沿用 demo.yaml，只改 video_name/static_cam/use_dpvo/f_mm/model 等设置
    with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
        overrides = [
            "model/gvhmr=gvhmr_pl_demo_tpu",
            f"video_name=phase1_tpu_{bbox_pt.stem}",
            f"static_cam={args.static_cam}",
            f"use_dpvo={args.use_dpvo}",
        ]
        if args.f_mm is not None:
            overrides.append(f"f_mm={args.f_mm}")
        if args.ckpt_path is not None:
            overrides.append(f"ckpt_path={args.ckpt_path}")

        register_store_gvhmr()
        cfg = compose(config_name="demo", overrides=overrides)

    # 额外记录 TPU demo 特有路径
    cfg.video_root = str(video_root)
    cfg.bbox_pt = str(bbox_pt)
    cfg.output_labels_pt = str(output_labels_pt)
    cfg.num_workers = args.num_workers

    Log.info(f"[TPU Demo] video_root = {cfg.video_root}")
    Log.info(f"[TPU Demo] bbox_pt = {cfg.bbox_pt}")
    Log.info(f"[TPU Demo] output_labels_pt = {cfg.output_labels_pt}")
    Log.info(f"[TPU Demo] static_cam = {cfg.static_cam}, use_dpvo = {cfg.use_dpvo}, f_mm = {cfg.f_mm}")

    return cfg


def build_dataloader(cfg) -> DataLoader:
    dataset = Phase1DemoDatasetTPU(
        video_root=cfg.video_root,
        bbox_pt_path=cfg.bbox_pt,
        static_cam=cfg.static_cam,
        use_dpvo=cfg.use_dpvo,
        f_mm=cfg.f_mm,
    )

    def collate_fn(batch):
        # batch_size 固定为 1：每个 step 处理一个视频，避免复杂 padding
        assert len(batch) == 1
        return batch[0]

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=False,
        collate_fn=collate_fn,
    )
    return loader


def map_joints_to_22(joints: torch.Tensor) -> torch.Tensor:
    """
    关节点映射到 22 维骨架。
    目前 EnDecoder.fk_v2 已直接输出 22 关节，这里作为占位映射（identity）。
    """
    # joints: (T, J, C)
    if joints.size(1) == 22:
        return joints
    if joints.size(1) < 22:
        pad = torch.zeros(joints.size(0), 22 - joints.size(1), joints.size(2), device=joints.device, dtype=joints.dtype)
        return torch.cat([joints, pad], dim=1)
    return joints[:, :22]


def build_entry_from_outputs(
    sample: Dict[str, Any],
    outputs: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    """
    将模型输出整理为 val_labels_phase1.pt 风格的 entry。
    """
    K_fullimg = sample["K_fullimg"]  # (T, 3, 3)
    R_w2c = sample["R_w2c"]  # (T, 3, 3)
    t_w2c = sample["t_w2c"]  # (T, 3)

    joints_c = outputs["joints_c"]  # (T, J, 3)
    joints_w = outputs["joints_w"]  # (T, J, 3)
    joints_2d = outputs["joints_2d"]  # (T, J, 2)

    joints_c_22 = map_joints_to_22(joints_c)
    joints_w_22 = map_joints_to_22(joints_w)
    joints_2d_22 = map_joints_to_22(joints_2d)

    T = joints_c_22.shape[0]

    # 简单的帧级 mask（全 1）
    mask_raw = torch.ones(T, dtype=torch.bool)

    # intrinsic：取首帧
    intrinsic = K_fullimg[0]

    # extrinsic: [T, 4, 4]，world -> camera
    extrinsic = torch.eye(4).repeat(T, 1, 1)
    extrinsic[:, :3, :3] = R_w2c
    extrinsic[:, :3, 3] = t_w2c

    entry = {
        "labels": {
            "joints_2d": joints_2d_22,  # (T, 22, 2)
            "joints_c": joints_c_22,  # (T, 22, 3)
            "joints_w": joints_w_22,  # (T, 22, 3)
            "mask_raw": mask_raw,  # (T,)
        },
        "cameras": {
            "intrinsic": intrinsic,  # (3, 3)
            "extrinsic": extrinsic,  # (T, 4, 4)
        },
    }
    return entry


def main():
    cfg = parse_args_to_cfg_tpu()

    # DataLoader（CPU 预处理 + 多 worker）
    loader = build_dataloader(cfg)

    # HMR4D 模型（TPU 前向）
    Log.info("[HMR4D-TPU] Building model")
    model: DemoPLTPU = hydra.utils.instantiate(cfg.model, _recursive_=False)
    model.load_pretrained_model(cfg.ckpt_path)
    model.eval()

    labels_dict: Dict[str, Any] = {}

    from tqdm import tqdm

    for sample in tqdm(loader, desc="TPU Inference"):
        video_id = sample["video_id"]
        Log.info(f"[TPU Demo] processing video_id = {video_id}")

        # DemoPLTPU.predict 内部完成 XLA 前向 + xm._fetch 回 CPU
        outputs = model.predict(sample, static_cam=cfg.static_cam)
        entry = build_entry_from_outputs(sample, outputs)
        labels_dict[video_id] = entry

    Log.info(f"[TPU Demo] Saving labels to {cfg.output_labels_pt}")
    torch.save(labels_dict, cfg.output_labels_pt)


if __name__ == "__main__":
    main()

