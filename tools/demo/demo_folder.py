# tools/demo/demo_folder.py
import os
import cv2
import sys
import glob
import shutil
import torch
import numpy as np
import argparse
import pytorch_lightning as pl
from datetime import datetime
from pathlib import Path
from typing import Dict

from tqdm import tqdm
from einops import einsum, rearrange
from pytorch3d.transforms import quaternion_to_matrix

from hmr4d.utils.pylogger import Log
import hydra
from hydra import initialize_config_module, compose

from hmr4d.configs import register_store_gvhmr
from hmr4d.utils.video_io_utils import (
    get_video_lwh,
    read_video_np,
    save_video,
    merge_videos_horizontal,
    get_writer,
    get_video_reader,
)
from hmr4d.utils.vis.cv2_utils import (
    draw_bbx_xyxy_on_image_batch,
    draw_coco17_skeleton_batch,
)

from hmr4d.utils.preproc import Tracker, Extractor, VitPoseExtractor, SimpleVO
from hmr4d.utils.geo.hmr_cam import (
    get_bbx_xys_from_xyxy,
    estimate_K,
    convert_K_to_K4,
    create_camera_sensor,
)
from hmr4d.utils.geo_transform import (
    compute_cam_angvel,
    apply_T_on_points,
    compute_T_ayfz2ay,
)
from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL
from hmr4d.utils.net_utils import detach_to_cpu, to_cuda
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.vis.renderer import (
    Renderer,
    get_global_cameras_static,
    get_ground_params_from_points,
)

CRF = 23  # 17 基本无损；+6 大约减半码率

# -----------------------------
# 公共：单视频 cfg 构建 + 预处理/数据载入/渲染
# -----------------------------

def build_cfg_for_video(video_path: Path, output_root: Path, static_cam: bool, use_dpvo: bool, f_mm: int, verbose: bool):
    assert video_path.exists(), f"Video not found at {video_path}"
    length, width, height = get_video_lwh(video_path)
    Log.info(f"[Input]: {video_path}")
    Log.info(f"(L, W, H) = ({length}, {width}, {height})")

    with initialize_config_module(version_base="1.3", config_module=f"hmr4d.configs"):
        overrides = [
            f"video_name={video_path.stem}",
            f"static_cam={static_cam}",
            f"verbose={verbose}",
            f"use_dpvo={use_dpvo}",
        ]
        if f_mm is not None:
            overrides.append(f"f_mm={f_mm}")
        # 将单个视频的中间产物与可视化，写到每视频独立的 work 目录中（避免互相覆盖）
        if output_root is not None:
            overrides.append(f"output_root={str(output_root)}")
        register_store_gvhmr()
        cfg = compose(config_name="demo", overrides=overrides)

    # 输出目录/预处理目录
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.preprocess_dir).mkdir(parents=True, exist_ok=True)

    # 将原视频“规范化”拷贝到 cfg.video_path（帧率/编码保持一致性，简化后续流程）
    Log.info(f"[Copy Video] {video_path} -> {cfg.video_path}")
    if not Path(cfg.video_path).exists() or get_video_lwh(video_path)[0] != get_video_lwh(cfg.video_path)[0]:
        reader = get_video_reader(video_path)
        writer = get_writer(cfg.video_path, fps=30, crf=CRF)
        for img in tqdm(reader, total=get_video_lwh(video_path)[0], desc=f"Copy"):
            writer.write_frame(img)
        writer.close()
        reader.close()

    return cfg


@torch.no_grad()
def run_preprocess(cfg):
    Log.info(f"[Preprocess] Start!")
    tic = Log.time()
    video_path = cfg.video_path
    paths = cfg.paths
    static_cam = cfg.static_cam
    verbose = cfg.verbose

    # 1) 跟踪 + 提取 bbox
    if not Path(paths.bbx).exists():
        tracker = Tracker()
        bbx_xyxy = tracker.get_one_track(video_path).float()  # (L, 4)
        bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()  # (L, 3)
        torch.save({"bbx_xyxy": bbx_xyxy, "bbx_xys": bbx_xys}, paths.bbx)
        del tracker
    else:
        bbx_xys = torch.load(paths.bbx)["bbx_xys"]
        Log.info(f"[Preprocess] bbx (xyxy, xys) from {paths.bbx}")
    if verbose:
        video = read_video_np(video_path)
        bbx_xyxy = torch.load(paths.bbx)["bbx_xyxy"]
        video_overlay = draw_bbx_xyxy_on_image_batch(bbx_xyxy, video)
        save_video(video_overlay, cfg.paths.bbx_xyxy_video_overlay)

    # 2) VitPose 关键点
    if not Path(paths.vitpose).exists():
        vitpose_extractor = VitPoseExtractor()
        vitpose = vitpose_extractor.extract(video_path, bbx_xys)
        torch.save(vitpose, paths.vitpose)
        del vitpose_extractor
    else:
        vitpose = torch.load(paths.vitpose)
        Log.info(f"[Preprocess] vitpose from {paths.vitpose}")
    if verbose:
        video = read_video_np(video_path)
        video_overlay = draw_coco17_skeleton_batch(video, vitpose, 0.5)
        save_video(video_overlay, paths.vitpose_video_overlay)

    # 3) ViT 图像特征
    if not Path(paths.vit_features).exists():
        extractor = Extractor()
        vit_features = extractor.extract_video_features(video_path, bbx_xys)
        torch.save(vit_features, paths.vit_features)
        del extractor
    else:
        Log.info(f"[Preprocess] vit_features from {paths.vit_features}")

    # 4) 视觉里程计（相机旋转）
    if not static_cam:
        if not Path(paths.slam).exists():
            if not cfg.use_dpvo:
                simple_vo = SimpleVO(cfg.video_path, scale=0.5, step=8, method="sift", f_mm=cfg.f_mm)
                vo_results = simple_vo.compute()  # (L, 4, 4), numpy
                torch.save(vo_results, paths.slam)
            else:
                from hmr4d.utils.preproc.slam import SLAMModel
                length, width, height = get_video_lwh(cfg.video_path)
                K_fullimg = estimate_K(width, height)
                intrinsics = convert_K_to_K4(K_fullimg)
                slam = SLAMModel(video_path, width, height, intrinsics, buffer=4000, resize=0.5)
                bar = tqdm(total=length, desc="DPVO")
                while True:
                    ret = slam.track()
                    if ret:
                        bar.update()
                    else:
                        break
                slam_results = slam.process()  # (L, 7), numpy
                torch.save(slam_results, paths.slam)
        else:
            Log.info(f"[Preprocess] slam results from {paths.slam}")

    Log.info(f"[Preprocess] End. Time elapsed: {Log.time()-tic:.2f}s")


def load_data_dict(cfg):
    paths = cfg.paths
    length, width, height = get_video_lwh(cfg.video_path)
    if cfg.static_cam:
        R_w2c = torch.eye(3).repeat(length, 1, 1)
    else:
        traj = torch.load(cfg.paths.slam)
        if cfg.use_dpvo:  # DPVO
            traj_quat = torch.from_numpy(traj[:, [6, 3, 4, 5]])
            R_w2c = quaternion_to_matrix(traj_quat).mT
        else:  # SimpleVO
            R_w2c = torch.from_numpy(traj[:, :3, :3])

    if cfg.f_mm is not None:
        K_fullimg = create_camera_sensor(width, height, cfg.f_mm)[2].repeat(length, 1, 1)
    else:
        K_fullimg = estimate_K(width, height).repeat(length, 1, 1)

    data = {
        "length": torch.tensor(length),
        "bbx_xys": torch.load(paths.bbx)["bbx_xys"],
        "kp2d": torch.load(paths.vitpose),
        "K_fullimg": K_fullimg,
        "cam_angvel": compute_cam_angvel(R_w2c),
        "f_imgseq": torch.load(paths.vit_features),
    }
    return data


def render_incam(cfg):
    incam_video_path = Path(cfg.paths.incam_video)
    if incam_video_path.exists():
        Log.info(f"[Render Incam] Exists: {incam_video_path}")
        return

    pred = torch.load(cfg.paths.hmr4d_results)
    smplx = make_smplx("supermotion").cuda()
    smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").cuda()
    faces_smpl = make_smplx("smpl").faces

    smplx_out = smplx(**to_cuda(pred["smpl_params_incam"]))
    pred_c_verts = torch.stack([torch.matmul(smplx2smpl, v_) for v_ in smplx_out.vertices])

    video_path = cfg.video_path
    length, width, height = get_video_lwh(video_path)
    K = pred["K_fullimg"][0]

    renderer = Renderer(width, height, device="cuda", faces=faces_smpl, K=K)
    reader = get_video_reader(video_path)
    # bbx 叠加可按需恢复
    writer = get_writer(incam_video_path, fps=30, crf=CRF)
    for i, img_raw in tqdm(enumerate(reader), total=get_video_lwh(video_path)[0], desc=f"Rendering Incam"):
        img = renderer.render_mesh(pred_c_verts[i].cuda(), img_raw, [0.8, 0.8, 0.8])
        writer.write_frame(img)
    writer.close()
    reader.close()


def render_global(cfg):
    global_video_path = Path(cfg.paths.global_video)
    if global_video_path.exists():
        Log.info(f"[Render Global] Exists: {global_video_path}")
        return

    pred = torch.load(cfg.paths.hmr4d_results)
    smplx = make_smplx("supermotion").cuda()
    smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").cuda()
    faces_smpl = make_smplx("smpl").faces
    J_regressor = torch.load("hmr4d/utils/body_model/smpl_neutral_J_regressor.pt").cuda()

    smplx_out = smplx(**to_cuda(pred["smpl_params_global"]))
    pred_ay_verts = torch.stack([torch.matmul(smplx2smpl, v_) for v_ in smplx_out.vertices])

    def move_to_start_point_face_z(verts):
        "XZ to origin, Start from the ground, Face-Z"
        verts = verts.clone()  # (L, V, 3)
        offset = einsum(J_regressor, verts[0], "j v, v i -> j i")[0]  # (3)
        offset[1] = verts[:, :, [1]].min()
        verts = verts - offset
        T_ay2ayfz = compute_T_ayfz2ay(einsum(J_regressor, verts[[0]], "j v, l v i -> l j i"), inverse=True)
        verts = apply_T_on_points(verts, T_ay2ayfz)
        return verts

    verts_glob = move_to_start_point_face_z(pred_ay_verts)
    joints_glob = einsum(J_regressor, verts_glob, "j v, l v i -> l j i")
    global_R, global_T, global_lights = get_global_cameras_static(
        verts_glob.cpu(), beta=2.0, cam_height_degree=20, target_center_height=1.0
    )

    video_path = cfg.video_path
    length, width, height = get_video_lwh(video_path)
    _, _, K = create_camera_sensor(width, height, 24)  # 以 24mm 渲染

    renderer = Renderer(width, height, device="cuda", faces=faces_smpl, K=K)

    scale, cx, cz = get_ground_params_from_points(joints_glob[:, 0], verts_glob)
    renderer.set_ground(scale * 1.5, cx, cz)
    color = torch.ones(3).float().cuda() * 0.8

    writer = get_writer(global_video_path, fps=30, crf=CRF)
    for i in tqdm(range(length), desc=f"Rendering Global"):
        cameras = renderer.create_camera(global_R[i], global_T[i])
        img = renderer.render_with_ground(verts_glob[[i]], color[None], cameras, global_lights)
        writer.write_frame(img)
    writer.close()


# -----------------------------
# 批处理主流程
# -----------------------------

def parse_folder_args():
    parser = argparse.ArgumentParser("GVHMR Folder Demo (Batch Inference)")
    parser.add_argument("--input_dir", type=str, required=True, help="扁平视频目录（mp4/mov/avi/mkv）")
    parser.add_argument("--output_dir", type=str, required=True, help="输出根目录")
    parser.add_argument("-s", "--static_cam", action="store_true", help="相机静止则跳过 VO")
    parser.add_argument("--use_dpvo", action="store_true", help="使用 DPVO（默认不使用，用 SimpleVO）")
    parser.add_argument(
        "--f_mm",
        type=int,
        default=None,
        help="全画幅焦距（mm）。iPhone 15p: [0.5x,1x,2x,3x]≈[13,24,48,77]；也可设 135、200。",
    )
    parser.add_argument("--verbose", action="store_true", help="写入中间可视化（bbox/pose）")
    return parser.parse_args()


def collect_result_from_pred(sequence_name: str, pred: Dict) -> Dict:
    """
    将 Demo 预测结构规整为目标字典：
    {
      'sequence_name': {
        'smlpX_data_c': {...},
        'smlpX_data_w': {...},
      }
    }
    """
    incam = pred["smpl_params_incam"]
    globl = pred["smpl_params_global"]
    # 确保在 CPU 上（如果 pred 来自 torch.load，一般已在 CPU）
    def cpu(t): return t.detach().cpu() if torch.is_tensor(t) else t

    return {
        sequence_name: {
            "smlpX_data_c": {
                "global_orient": cpu(incam["global_orient"]),
                "body_pose":     cpu(incam["body_pose"]),
                "betas":         cpu(incam["betas"]),
                "transl":        cpu(incam["transl"]),
            },
            "smlpX_data_w": {
                "global_orient": cpu(globl["global_orient"]),
                "body_pose":     cpu(globl["body_pose"]),
                "betas":         cpu(globl["betas"]),
                "transl":        cpu(globl["transl"]),
            },
        }
    }


def main():
    args = parse_folder_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    assert input_dir.exists() and input_dir.is_dir(), f"input_dir not found: {input_dir}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 统一时间戳（推理启动时刻）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pt_path = output_dir / f"{timestamp}.pt"
    viz_dir = output_dir / f"{timestamp}"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # 收集所有视频
    exts = ("*.mp4", "*.MP4", "*.mov", "*.MOV", "*.avi", "*.AVI", "*.mkv", "*.MKV")
    video_list = []
    for ext in exts:
        video_list.extend(sorted(input_dir.glob(ext)))
    assert len(video_list) > 0, f"No videos found in {input_dir}"

    Log.info(f"[Batch] {len(video_list)} videos found.")
    Log.info(f"[Batch] Output: {pt_path}")
    Log.info(f"[Batch] Viz Dir: {viz_dir}")

    # 聚合结果
    all_results = {}

    # 为每个视频创建独立的 work 目录，避免相互覆盖
    work_root = output_dir / f"_work_{timestamp}"
    work_root.mkdir(parents=True, exist_ok=True)

    for vid_path in video_list:
        sequence_name = vid_path.stem
        Log.info("=" * 80)
        Log.info(f"[Video] {sequence_name}")
        # 每视频一个 cfg（输出到 work_root/sequence_name 下）
        per_video_root = work_root / sequence_name
        per_video_root.mkdir(parents=True, exist_ok=True)
        cfg = build_cfg_for_video(
            video_path=vid_path,
            output_root=per_video_root,
            static_cam=args.static_cam,
            use_dpvo=args.use_dpvo,
            f_mm=args.f_mm,
            verbose=args.verbose,
        )
        paths = cfg.paths

        # 预处理
        run_preprocess(cfg)
        data = load_data_dict(cfg)

        # 推理（缓存复用）
        if not Path(paths.hmr4d_results).exists():
            Log.info("[HMR4D] Predicting")
            model: DemoPL = hydra.utils.instantiate(cfg.model, _recursive_=False)
            model.load_pretrained_model(cfg.ckpt_path)
            model = model.eval().cuda()
            tic = Log.sync_time()
            pred = model.predict(data, static_cam=cfg.static_cam)
            pred = detach_to_cpu(pred)
            data_time = data["length"] / 30
            Log.info(f"[HMR4D] Elapsed: {Log.sync_time() - tic:.2f}s for data-length={data_time:.1f}s")
            torch.save(pred, paths.hmr4d_results)
        else:
            Log.info(f"[HMR4D] Load cached results: {paths.hmr4d_results}")
            pred = torch.load(paths.hmr4d_results)

        # 渲染与合成左右画面
        render_incam(cfg)
        render_global(cfg)
        if not Path(paths.incam_global_horiz_video).exists():
            Log.info("[Merge Videos]")
            merge_videos_horizontal([paths.incam_video, paths.global_video], paths.incam_global_horiz_video)

        # 复制到最终可视化目录，命名为 viz_<name>.mp4
        final_viz = viz_dir / f"viz_{sequence_name}.mp4"
        shutil.copyfile(paths.incam_global_horiz_video, final_viz)
        Log.info(f"[Viz Saved] {final_viz}")

        # 收集需要的四个字段，写入聚合字典
        one_result = collect_result_from_pred(sequence_name, pred)
        all_results.update(one_result)

    # 写出聚合 pt
    torch.save(all_results, pt_path)
    Log.info(f"[All Done] Results saved to {pt_path}")
    Log.info(f"[All Done] Visualizations in {viz_dir}")

    # 可选：清理中间 work 目录（默认保留便于复现/排错）
    # shutil.rmtree(work_root, ignore_errors=True)


if __name__ == "__main__":
    # 打印 GPU 信息（可选）
    if torch.cuda.is_available():
        Log.info(f"[GPU]: {torch.cuda.get_device_name()}")
    else:
        Log.warn("[GPU]: CUDA not available")

    main()
