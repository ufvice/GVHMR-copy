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
import inspect
from datetime import datetime
from pathlib import Path
from typing import Dict

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
from hmr4d.utils.geo_transform import compute_cam_angvel, apply_T_on_points, compute_T_ayfz2ay
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
    """为单个视频构建 Hydra cfg，并把原视频规范化复制到 cfg.video_path。
    去掉 tqdm，改为 Log 进度。
    参考原实现：复制流程与目录创建。"""
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

    # 将原视频“规范化”拷贝到 cfg.video_path（统一帧率/编码以简化后续流程）
    Log.info(f"[Copy Video] {video_path} -> {cfg.video_path}")
    need_copy = (not Path(cfg.video_path).exists()) or (get_video_lwh(video_path)[0] != get_video_lwh(cfg.video_path)[0])
    if need_copy:
        reader = get_video_reader(video_path)
        writer = get_writer(cfg.video_path, fps=30, crf=CRF)
        total = get_video_lwh(video_path)[0]
        step = max(total // 10, 1)
        for idx, img in enumerate(reader):
            writer.write_frame(img)
            if (idx % step == 0) or (idx + 1 == total):
                Log.info(f"[Copy] {idx + 1}/{total}")
        writer.close()
        reader.close()

    return cfg


@torch.no_grad()
def run_preprocess(cfg, batch_size: int = None):
    """按原逻辑进行单视频预处理：跟踪→关键点→ViT特征→VO；仅替换进度输出。
    在 Extractor 支持的情况下传入 batch_size。"""
    Log.info(f"[Preprocess] Start!")
    tic = Log.time()
    video_path = cfg.video_path
    paths = cfg.paths
    static_cam = cfg.static_cam
    verbose = cfg.verbose

    # 1) 跟踪得到 bbox，并派生 bbx_xys（按原规则扩大比例）
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

    # 2) ViTPose 关键点
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

    # 3) ViT 特征（尽量使用传入 batch_size）
    if not Path(paths.vit_features).exists():
        extractor = Extractor()
        sig = inspect.signature(extractor.extract_video_features)
        if "batch_size" in sig.parameters:
            vit_features = extractor.extract_video_features(video_path, bbx_xys, batch_size=batch_size)
        else:
            vit_features = extractor.extract_video_features(video_path, bbx_xys)
        torch.save(vit_features, paths.vit_features)
        del extractor
    else:
        Log.info(f"[Preprocess] vit_features from {paths.vit_features}")

    # 4) 视觉里程计/相机旋转（DPVO 或 SimpleVO）
    if not static_cam:  # use slam to get cam rotation
        if not Path(paths.slam).exists():
            if not cfg.use_dpvo:
                # SimpleVO
                simple_vo = SimpleVO(cfg.video_path, scale=0.5, step=8, method="sift", f_mm=cfg.f_mm)
                vo_results = simple_vo.compute()  # (L, 4, 4), numpy
                torch.save(vo_results, paths.slam)
            else:
                # DPVO：把原来的 tqdm 进度改为 Log
                from hmr4d.utils.preproc.slam import SLAMModel
                length, width, height = get_video_lwh(cfg.video_path)
                K_fullimg = estimate_K(width, height)
                intrinsics = convert_K_to_K4(K_fullimg)
                slam = SLAMModel(video_path, width, height, intrinsics, buffer=4000, resize=0.5)
                processed = 0
                step = max(length // 10, 1)
                while True:
                    ret = slam.track()
                    if ret:
                        processed += 1
                        if (processed % step == 0) or (processed == length):
                            Log.info(f"[DPVO] {processed}/{length}")
                    else:
                        break
                slam_results = slam.process()  # (L, 7), numpy
                torch.save(slam_results, paths.slam)
        else:
            Log.info(f"[Preprocess] slam results from {paths.slam}")

    Log.info(f"[Preprocess] End. Time elapsed: {Log.time()-tic:.2f}s")


def load_data_dict(cfg):
    """装载推理所需字段：length、bbx_xys、kp2d、K_fullimg、cam_angvel、f_imgseq。"""
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


def render_incam(cfg, pred):
    """渲染相机坐标系视角视频。仅用 Log 做简易进度提示。"""
    incam_video_path = Path(cfg.paths.incam_video)
    if incam_video_path.exists():
        Log.info(f"[Render Incam] Video already exists at {incam_video_path}")
        return

    smplx = make_smplx("supermotion").cuda()
    smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").cuda()
    faces_smpl = make_smplx("smpl").faces

    # smpl
    smplx_out = smplx(**to_cuda(pred["smpl_params_incam"]))
    pred_c_verts = torch.stack([torch.matmul(smplx2smpl, v_) for v_ in smplx_out.vertices])

    # -- rendering code -- #
    video_path = cfg.video_path
    length, width, height = get_video_lwh(video_path)
    K = pred["K_fullimg"][0]

    # renderer
    renderer = Renderer(width, height, device="cuda", faces=faces_smpl, K=K)
    reader = get_video_reader(video_path)  # (F, H, W, 3), uint8, numpy
    bbx_xys_render = torch.load(cfg.paths.bbx)["bbx_xys"]

    writer = get_writer(str(incam_video_path), fps=30, crf=CRF)
    step = max(length // 10, 1)
    for i, img in enumerate(reader):
        verts = pred_c_verts[[i]].contiguous()
        cam = renderer.create_camera(torch.eye(3).cuda(), torch.zeros(3).cuda())
        lights = renderer.create_lights()
        rendered = renderer.render_with_ground(verts, None, cam, lights, background=img)
        writer.write_frame(rendered)
        if (i % step == 0) or (i + 1 == length):
            Log.info(f"[Render Incam] {i + 1}/{length}")
    writer.close()


def render_global(cfg, pred):
    """渲染世界坐标系（重力对齐）视角。去掉 tqdm；保持缓存输出路径不变。"""
    global_video_path = Path(cfg.paths.global_video)
    if global_video_path.exists():
        Log.info(f"[Render Global] Video already exists at {global_video_path}")
        return

    smplx = make_smplx("supermotion").cuda()
    smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").cuda()
    faces_smpl = make_smplx("smpl").faces

    smplx_out = smplx(**to_cuda(pred["smpl_params_global"]))
    verts_glob = torch.stack([torch.matmul(smplx2smpl, v_) for v_ in smplx_out.vertices])  # (F, V, 3)

    length = verts_glob.shape[0]
    width, height = pred["image_wh"]
    width, height = int(width), int(height)

    # renderer (静态全局视角)
    renderer = Renderer(width, height, device="cuda", faces=faces_smpl, K=torch.eye(3).float().cuda())
    global_R, global_T, global_lights = get_global_cameras_static(verts_glob)

    # 地面
    joints_glob = smplx_out.joints.detach().cpu()
    scale, cx, cz = get_ground_params_from_points(joints_glob[:, 0], verts_glob)
    renderer.set_ground(scale * 1.5, cx, cz)
    color = torch.ones(3).float().cuda() * 0.8

    writer = get_writer(str(global_video_path), fps=30, crf=CRF)
    step = max(length // 10, 1)
    for i in range(length):
        cameras = renderer.create_camera(global_R[i], global_T[i])
        img = renderer.render_with_ground(verts_glob[[i]], color[None], cameras, global_lights)
        writer.write_frame(img)
        if (i % step == 0) or (i + 1 == length):
            Log.info(f"[Render Global] {i + 1}/{length}")
    writer.close()


# -----------------------------
# 批处理主流程（本次改造重点）
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
    # 新增：batch_size 暴露到命令行（用于可用的特征提取批量）
    parser.add_argument("--batch_size", type=int, default=8, help="特征抽取的批大小（Extractor支持则生效）")
    return parser.parse_args()


def collect_result_from_pred(sequence_name: str, pred: Dict) -> Dict:
    """
    将 Demo 预测结构规整为目标字典，CPU 张量，便于 torch.save 增量更新。
    """
    incam = pred["smpl_params_incam"]
    globl = pred["smpl_params_global"]

    def cpu(t):  # 确保落盘前都在 CPU
        return t.detach().cpu() if torch.is_tensor(t) else t

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

    # 收集所有视频（保持与原实现一致的后缀集合）
    exts = ("*.mp4", "*.MP4", "*.mov", "*.MOV", "*.avi", "*.AVI", "*.mkv", "*.MKV")
    video_list = []
    for ext in exts:
        video_list.extend(sorted(input_dir.glob(ext)))
    assert len(video_list) > 0, f"No videos found in {input_dir}"

    Log.info(f"[Batch] {len(video_list)} videos found.")
    Log.info(f"[Batch] Output: {pt_path}")
    Log.info(f"[Batch] Viz Dir: {viz_dir}")

    # 聚合结果
    all_results: Dict[str, Dict] = {}

    # 为每个视频创建独立的 work 目录，避免相互覆盖
    work_root = output_dir / f"_work_{timestamp}"
    work_root.mkdir(parents=True, exist_ok=True)

    # -------- Stage-0：构建 per-video cfg --------
    cfg_list = []
    for vid_path in video_list:
        sequence_name = vid_path.stem
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
        cfg_list.append((sequence_name, cfg))

    # -------- Stage-1：批量预处理（仅生成/复用中间缓存） --------
    Log.info("=" * 80)
    Log.info("[Stage-1] 批量预处理开始")
    for sequence_name, cfg in cfg_list:
        Log.info("-" * 60)
        Log.info(f"[Preprocess] {sequence_name}")
        run_preprocess(cfg, batch_size=args.batch_size)
    Log.info("[Stage-1] 批量预处理完成")

    # -------- Stage-2：批量推理 + 渲染 + 增量写出结果 --------
    Log.info("=" * 80)
    Log.info("[Stage-2] 批量推理/渲染开始")

    # 准备模型
    pl.seed_everything(42)
    with torch.inference_mode():
        model: DemoPL = hydra.utils.instantiate(
            {"_target_": "hmr4d.model.gvhmr.gvhmr_pl_demo.DemoPL"}, _recursive_=False
        )
        # 默认权重路径由 DemoPL 内部处理；如需固定权重也可使用 build_gvhmr_demo
        model = model.eval().cuda() if torch.cuda.is_available() else model.eval()

        for sequence_name, cfg in cfg_list:
            Log.info("-" * 60)
            Log.info(f"[Infer] {sequence_name}")

            # 装载数据
            data = load_data_dict(cfg)

            # 前向（与原单视频逻辑一致）
            inputs = {
                "length": data["length"][None],  # (1,)
                "kp2d": data["kp2d"][None],
                "bbx_xys": data["bbx_xys"][None],
                "K_fullimg": data["K_fullimg"][None],
                "cam_angvel": data["cam_angvel"][None],
                "f_imgseq": data["f_imgseq"][None],
            }
            inputs_cuda = to_cuda(inputs) if torch.cuda.is_available() else inputs
            pred = model.predict(inputs_cuda)  # 与 DemoPL 接口保持一致
            pred = detach_to_cpu(pred)  # 便于后续渲染/落盘

            # 渲染与合成左右画面
            paths = cfg.paths
            render_incam(cfg, pred)
            render_global(cfg, pred)
            if not Path(paths.incam_global_horiz_video).exists():
                Log.info("[Merge Videos]")
                merge_videos_horizontal([paths.incam_video, paths.global_video], paths.incam_global_horiz_video)

            # 复制到最终可视化目录，命名为 viz_<name>.mp4
            final_viz = viz_dir / f"viz_{sequence_name}.mp4"
            shutil.copyfile(paths.incam_global_horiz_video, final_viz)
            Log.info(f"[Viz Saved] {final_viz}")

            # 收集需要的四个字段，写入聚合字典 —— 每个视频完成后立即增量保存（追加式）
            one_result = collect_result_from_pred(sequence_name, pred)
            all_results.update(one_result)
            torch.save(all_results, pt_path)
            Log.info(f"[Saved] Append result of '{sequence_name}' -> {pt_path}")

    Log.info("[Stage-2] 全部推理/渲染完成")
    Log.info(f"[All Done] Results saved to {pt_path}")
    Log.info(f"[All Done] Visualizations in {viz_dir}")

    # （可选）清理中间 work 目录；按需求启用
    # shutil.rmtree(work_root, ignore_errors=True)


if __name__ == "__main__":
    if torch.cuda.is_available():
        Log.info(f"[GPU]: {torch.cuda.get_device_name()}")
    else:
        Log.warn("[GPU]: CUDA not available")
    main()
