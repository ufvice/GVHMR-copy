import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional, Any

from hmr4d.utils.video_io_utils import get_video_lwh
from hmr4d.utils.preproc import VitPoseExtractor, Extractor, SimpleVO
from hmr4d.utils.geo.hmr_cam import (
    get_bbx_xys_from_xyxy,
    estimate_K,
    convert_K_to_K4,
    create_camera_sensor,
)
from hmr4d.utils.geo_transform import compute_cam_angvel
from hmr4d.utils.pylogger import Log
from pytorch3d.transforms import quaternion_to_matrix

try:
    import torch_xla.core.xla_model as xm

    _HAS_XLA = True
except Exception:  # pragma: no cover - 仅在无 XLA 环境下触发
    _HAS_XLA = False


class Phase1DemoDatasetTPU(Dataset):
    """
    视频 + bbox 字典 -> 推理输入 Dataset（TPU 版）。

    - 输入：
      - video_root: 所有视频所在目录，文件名通常为 `{video_id}.mp4`
      - bbox_pt_path: `val_bbox_phase1.pt` 路径，结构为 {video_id: [T, 4]}
    - 输出（单个 sample 字典）：
      - video_id: str
      - length: torch.Tensor[T]
      - bbx_xys: [T, 3]
      - kp2d: [T, 17, 3]（VitPose）
      - K_fullimg: [T, 3, 3]
      - cam_angvel: [T, 6]
      - f_imgseq: [T, C]
      - R_w2c: [T, 3, 3]
      - t_w2c: [T, 3]
    """

    def __init__(
        self,
        video_root: str,
        bbox_pt_path: str,
        static_cam: bool = False,
        use_dpvo: bool = False,
        f_mm: Optional[int] = None,
        img_ds: float = 0.5,
    ) -> None:
        super().__init__()
        self.video_root = Path(video_root)
        self.bbox_pt_path = Path(bbox_pt_path)
        self.static_cam = static_cam
        self.use_dpvo = use_dpvo
        self.f_mm = f_mm
        self.img_ds = img_ds

        assert self.video_root.exists(), f"video_root not found: {self.video_root}"
        assert self.bbox_pt_path.exists(), f"bbox_pt not found: {self.bbox_pt_path}"

        self.bbox_dict: Dict[str, torch.Tensor] = torch.load(self.bbox_pt_path)
        self.video_ids: List[str] = sorted(self.bbox_dict.keys())

        # 运行时按需懒加载的预处理器（在 DataLoader worker 内部初始化）
        self._vitpose_extractor: Optional[VitPoseExtractor] = None
        self._feature_extractor: Optional[Extractor] = None

    def __len__(self) -> int:
        return len(self.video_ids)

    def _get_vitpose_extractor(self) -> VitPoseExtractor:
        if self._vitpose_extractor is None:
            # tqdm_leave=False 以避免多 worker 下的多重进度条
            # 若检测到 torch_xla，可尝试将 ViTPose 模型放到 XLA 设备上。
            # 注意：在 DataLoader 使用多进程 + XLA 时可能存在不稳定因素，建议在这种模式下将
            # num_workers 设为 0（单进程）再使用。
            if _HAS_XLA:
                device = xm.xla_device()
                self._vitpose_extractor = VitPoseExtractor(tqdm_leave=False, device=device)
            else:
                self._vitpose_extractor = VitPoseExtractor(tqdm_leave=False)
        return self._vitpose_extractor

    def _get_feature_extractor(self) -> Extractor:
        if self._feature_extractor is None:
            # 若检测到 torch_xla，可尝试将 HMR2 特征提取器放到 XLA 设备上。
            # 同样建议在这种模式下将 DataLoader 的 num_workers 设为 0。
            if _HAS_XLA:
                device = xm.xla_device()
                self._feature_extractor = Extractor(tqdm_leave=False, device=device)
            else:
                self._feature_extractor = Extractor(tqdm_leave=False)
        return self._feature_extractor

    def _compute_vo(
        self,
        video_path: Path,
        length: int,
        width: int,
        height: int,
    ) -> Dict[str, torch.Tensor]:
        """
        视觉里程计（VO）计算相机外参。
        - static_cam=True: 直接返回单位外参。
        - use_dpvo=True: 调用 DPVO（slam.py），输出 (L, 7) -> R_w2c, t_w2c。
        - 否则：SimpleVO -> (L, 4, 4)。
        """
        if self.static_cam:
            R_w2c = torch.eye(3).repeat(length, 1, 1)
            t_w2c = torch.zeros(length, 3)
            return {"R_w2c": R_w2c, "t_w2c": t_w2c}

        if self.use_dpvo:
            from hmr4d.utils.preproc.slam import SLAMModel

            K_fullimg_est = estimate_K(width, height)
            intrinsics = convert_K_to_K4(K_fullimg_est)
            slam = SLAMModel(str(video_path), width, height, intrinsics, buffer=4000, resize=0.5)

            from tqdm import tqdm

            bar = tqdm(total=length, desc="DPVO", leave=False)
            while True:
                ret = slam.track()
                if ret:
                    bar.update()
                else:
                    break
            bar.close()
            traj = slam.process()  # (L, 7), numpy

            traj = torch.from_numpy(traj).float()
            # DPVO 约定: [tx, ty, tz, qx, qy, qz, qw]
            t_w2c = traj[:, 0:3]
            traj_quat = traj[:, [6, 3, 4, 5]]  # (L, 4) -> [qw, qx, qy, qz]
            R_w2c = quaternion_to_matrix(traj_quat).mT  # (L, 3, 3)
        else:
            # SimpleVO: 直接输出 (L, 4, 4) 齐次变换矩阵
            simple_vo = SimpleVO(str(video_path), scale=0.5, step=8, method="sift", f_mm=self.f_mm)
            vo_results = simple_vo.compute()  # (L, 4, 4), numpy
            vo_results = torch.from_numpy(vo_results).float()
            R_w2c = vo_results[:, :3, :3]
            t_w2c = vo_results[:, :3, 3]

        return {"R_w2c": R_w2c, "t_w2c": t_w2c}

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        video_id = self.video_ids[idx]
        bbox_xyxy = self.bbox_dict[video_id].float()  # (T, 4)

        video_path = self.video_root / f"{video_id}.mp4"
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found for id={video_id}: {video_path}")

        length_bbox = bbox_xyxy.shape[0]
        length_vid, width, height = get_video_lwh(video_path)
        if length_vid != length_bbox:
            Log.warn(
                f"[Phase1DemoDatasetTPU] length mismatch for {video_id}: "
                f"video={length_vid}, bbox={length_bbox}. Using bbox length."
            )
        T = length_bbox

        # bbox: [T,4] -> [T,3]
        bbx_xys = get_bbx_xys_from_xyxy(bbox_xyxy, base_enlarge=1.2).float()

        # 相机内参（全图）
        if self.f_mm is not None:
            _, _, K_fullimg_single = create_camera_sensor(width, height, self.f_mm)
        else:
            K_fullimg_single = estimate_K(width, height)
        K_fullimg = K_fullimg_single.repeat(T, 1, 1)  # (T, 3, 3)

        # VO / 相机外参
        vo_dict = self._compute_vo(video_path, length_vid, width, height)
        R_w2c = vo_dict["R_w2c"][:T]  # (T, 3, 3)
        t_w2c = vo_dict["t_w2c"][:T]  # (T, 3)
        cam_angvel = compute_cam_angvel(R_w2c)  # (T, 6)

        # VitPose 关键点
        vitpose_extractor = self._get_vitpose_extractor()
        kp2d = vitpose_extractor.extract(str(video_path), bbx_xys, img_ds=self.img_ds)  # (T, 17, 3)

        # ViT/HMR2 特征
        feat_extractor = self._get_feature_extractor()
        vit_features = feat_extractor.extract_video_features(str(video_path), bbx_xys, img_ds=self.img_ds)  # (T, C)

        sample: Dict[str, Any] = {
            "video_id": video_id,
            "length": torch.tensor(T, dtype=torch.long),
            "bbx_xys": bbx_xys,  # (T, 3)
            "kp2d": kp2d,  # (T, 17, 3)
            "K_fullimg": K_fullimg,  # (T, 3, 3)
            "cam_angvel": cam_angvel,  # (T, 6)
            "f_imgseq": vit_features,  # (T, C)
            "R_w2c": R_w2c,  # (T, 3, 3)
            "t_w2c": t_w2c,  # (T, 3)
        }
        return sample


__all__ = ["Phase1DemoDatasetTPU"]
