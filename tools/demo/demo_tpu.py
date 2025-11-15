import argparse
from pathlib import Path
from typing import Dict, Any, List
import time

import torch
import hydra
from hydra import initialize_config_module, compose

from hmr4d.configs import register_store_gvhmr
from hmr4d.utils.pylogger import Log

from hmr4d.dataset.phase1_demo_tpu import Phase1DemoDatasetTPU
from hmr4d.model.gvhmr.gvhmr_pl_demo_tpu import DemoPLTPU
from hmr4d.utils.video_io_utils import get_video_lwh
from hmr4d.utils.geo.hmr_cam import (
    get_bbx_xys_from_xyxy,
    estimate_K,
    create_camera_sensor,
)
from hmr4d.utils.geo_transform import compute_cam_angvel

try:
    import wandb
except ImportError:
    wandb = None
    Log.warn("[TPU Demo] wandb 未安装，跳过 W&B 日志记录")


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
            "model=gvhmr/gvhmr_pl_demo_tpu",
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


def build_dataset(cfg) -> Phase1DemoDatasetTPU:
    """
    构建 TPU Demo 专用 Dataset，但不再通过 DataLoader 按视频即时做所有预处理，
    而是后续在本文件中按「阶段」对所有视频批处理：
        1) VO / 相机外参估计
        2) ViTPose 姿态估计
        3) HMR2.0 特征提取
        4) GVHMR 模型推理

    这样可以在同一阶段内对大量视频重复调用同一套 PyTorch/XLA 计算图，
    有利于 XLA 复用编译结果，减少每个视频单独构图/编译的额外开销。
    """
    dataset = Phase1DemoDatasetTPU(
        video_root=cfg.video_root,
        bbox_pt_path=cfg.bbox_pt,
        static_cam=cfg.static_cam,
        use_dpvo=cfg.use_dpvo,
        f_mm=cfg.f_mm,
    )
    return dataset


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

    wandb_run = None
    if wandb is not None:
        # 初始化 wandb：记录基本配置与硬件信息（TPU 环境由 torch_xla 管理）
        run_name = f"tpu_demo_{Path(cfg.bbox_pt).stem}"
        wandb_run = wandb.init(
            project="gvhmr_tpu_demo",
            name=run_name,
            config={
                "video_root": cfg.video_root,
                "bbox_pt": cfg.bbox_pt,
                "output_labels_pt": cfg.output_labels_pt,
                "static_cam": cfg.static_cam,
                "use_dpvo": cfg.use_dpvo,
                "f_mm": cfg.f_mm,
                "num_workers": cfg.num_workers,
            },
        )

    # Dataset（仅负责索引、bbox 字典和 VO/VitPose/HMR2 工具，不再用 DataLoader 即时做所有预处理）
    dataset = build_dataset(cfg)
    video_ids: List[str] = dataset.video_ids
    num_videos = len(video_ids)

    if wandb_run is not None:
        # 记录一份数据量级信息，方便在 dashboard 上查看整体任务规模
        wandb.log({"data/num_videos": num_videos}, step=0)

    ########################################
    # 阶段 0：基础元信息预计算（长度 / 相机内参）
    ########################################
    Log.info("[TPU Demo] Phase 0: prepare meta info for all videos")
    meta_dict: Dict[str, Dict[str, Any]] = {}
    for video_id in video_ids:
        bbox_xyxy = dataset.bbox_dict[video_id].float()  # (T, 4)
        video_path = Path(cfg.video_root) / f"{video_id}.mp4"
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found for id={video_id}: {video_path}")

        length_bbox = bbox_xyxy.shape[0]
        length_vid, width, height = get_video_lwh(video_path)
        if length_vid != length_bbox:
            Log.warn(
                f"[TPU Demo] length mismatch for {video_id}: "
                f"video={length_vid}, bbox={length_bbox}. Using bbox length."
            )
        T = length_bbox

        # bbox: [T,4] -> [T,3]
        bbx_xys = get_bbx_xys_from_xyxy(bbox_xyxy, base_enlarge=1.2).float()

        # 相机内参（全图）
        if cfg.f_mm is not None:
            _, _, K_fullimg_single = create_camera_sensor(width, height, cfg.f_mm)
        else:
            K_fullimg_single = estimate_K(width, height)
        K_fullimg = K_fullimg_single.repeat(T, 1, 1)  # (T, 3, 3)

        meta_dict[video_id] = {
            "T": T,
            "video_path": video_path,
            "width": width,
            "height": height,
            "bbox_xyxy": bbox_xyxy,
            "bbx_xys": bbx_xys,
            "K_fullimg": K_fullimg,
        }

    ########################################
    # 阶段 1：视觉里程计 VO / 相机外参
    ########################################
    Log.info("[TPU Demo] Phase 1: VO / camera extrinsics for all videos")
    vo_dict: Dict[str, Dict[str, torch.Tensor]] = {}
    for idx, video_id in enumerate(video_ids):
        info = meta_dict[video_id]
        video_path: Path = info["video_path"]
        width: int = info["width"]
        height: int = info["height"]
        T: int = info["T"]

        vo = dataset._compute_vo(video_path, T, width, height)
        R_w2c = vo["R_w2c"][:T]
        t_w2c = vo["t_w2c"][:T]
        cam_angvel = compute_cam_angvel(R_w2c)

        vo_dict[video_id] = {
            "R_w2c": R_w2c,
            "t_w2c": t_w2c,
            "cam_angvel": cam_angvel,
        }

        if wandb_run is not None:
            wandb.log(
                {
                    "progress_vo/video_index": idx + 1,
                    "progress_vo/num_videos": num_videos,
                },
                step=idx + 1,
            )

    ########################################
    # 阶段 2：ViTPose 姿态估计（TPU/XLA 上的纯网络前向）
    ########################################
    Log.info("[TPU Demo] Phase 2: ViTPose for all videos")
    vitpose_extractor = dataset._get_vitpose_extractor()
    kp2d_dict: Dict[str, torch.Tensor] = {}
    for idx, video_id in enumerate(video_ids):
        info = meta_dict[video_id]
        video_path: Path = info["video_path"]
        bbx_xys: torch.Tensor = info["bbx_xys"]

        kp2d = vitpose_extractor.extract(str(video_path), bbx_xys, img_ds=dataset.img_ds)
        kp2d_dict[video_id] = kp2d

        if wandb_run is not None:
            wandb.log(
                {
                    "progress_vitpose/video_index": idx + 1,
                    "progress_vitpose/num_videos": num_videos,
                },
                step=idx + 1,
            )

    ########################################
    # 阶段 3：HMR2.0 特征提取（TPU/XLA 上的 ViT 主干前向）
    ########################################
    Log.info("[TPU Demo] Phase 3: HMR2 Feature for all videos")
    feat_extractor = dataset._get_feature_extractor()
    feat_dict: Dict[str, torch.Tensor] = {}
    for idx, video_id in enumerate(video_ids):
        info = meta_dict[video_id]
        video_path: Path = info["video_path"]
        bbx_xys: torch.Tensor = info["bbx_xys"]

        vit_features = feat_extractor.extract_video_features(str(video_path), bbx_xys, img_ds=dataset.img_ds)
        feat_dict[video_id] = vit_features

        if wandb_run is not None:
            wandb.log(
                {
                    "progress_hmr2/video_index": idx + 1,
                    "progress_hmr2/num_videos": num_videos,
                },
                step=idx + 1,
            )

    ########################################
    # 阶段 4：GVHMR 模型（TPU 前向，只消费已缓存的特征）
    ########################################
    Log.info("[HMR4D-TPU] Building model")
    model: DemoPLTPU = hydra.utils.instantiate(cfg.model, _recursive_=False)
    model.load_pretrained_model(cfg.ckpt_path)
    model.eval()

    labels_dict: Dict[str, Any] = {}

    from tqdm import tqdm

    for idx, video_id in enumerate(tqdm(video_ids, desc="TPU Inference")):
        info = meta_dict[video_id]
        vo_info = vo_dict[video_id]

        T = info["T"]
        bbx_xys = info["bbx_xys"]
        K_fullimg = info["K_fullimg"]
        kp2d = kp2d_dict[video_id]
        vit_features = feat_dict[video_id]
        R_w2c = vo_info["R_w2c"]
        t_w2c = vo_info["t_w2c"]
        cam_angvel = vo_info["cam_angvel"]

        sample: Dict[str, Any] = {
            "video_id": video_id,
            "length": torch.tensor(T, dtype=torch.long),
            "bbx_xys": bbx_xys,
            "kp2d": kp2d,
            "K_fullimg": K_fullimg,
            "cam_angvel": cam_angvel,
            "f_imgseq": vit_features,
            "R_w2c": R_w2c,
            "t_w2c": t_w2c,
        }

        Log.info(f"[TPU Demo] processing video_id = {video_id}")

        t0 = time.time()

        outputs = model.predict(sample, static_cam=cfg.static_cam)
        entry = build_entry_from_outputs(sample, outputs)
        labels_dict[video_id] = entry

        if wandb_run is not None:
            elapsed = time.time() - t0
            wandb.log(
                {
                    "progress/video_index": idx + 1,
                    "progress/num_videos": num_videos,
                    "timing/per_video_seconds": elapsed,
                },
                step=idx + 1,
            )

    Log.info(f"[TPU Demo] Saving labels to {cfg.output_labels_pt}")
    torch.save(labels_dict, cfg.output_labels_pt)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
