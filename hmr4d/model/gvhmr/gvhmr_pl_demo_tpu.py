import torch
import pytorch_lightning as pl
from hydra.utils import instantiate

from hmr4d.utils.pylogger import Log
from hmr4d.configs import MainStore, builds

from hmr4d.utils.geo.hmr_cam import (
    normalize_kp2d,
    compute_bbox_info_bedlam,
    compute_transl_full_cam,
    perspective_projection,
)
from hmr4d.model.gvhmr.pipeline.gvhmr_pipeline import get_smpl_params_w_Rt_v2
from hmr4d.model.gvhmr.utils.postprocess import pp_static_joint, pp_static_joint_cam, process_ik

import torch_xla.core.xla_model as xm


class DemoPLTPU(pl.LightningModule):
    """
    TPU 推理用的 Demo 封装，只做前向推理与简单后处理。
    """

    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = instantiate(pipeline, _recursive_=False)
        self.device_xla = None

    def _init_xla(self) -> None:
        if self.device_xla is None:
            self.device_xla = xm.xla_device()
            # 只将核心的 denoiser3d 放到 XLA 上，后处理（SMPLX、pytorch3d 等）
            # 仍然在 CPU 上执行，避免 XLA 对复杂几何算子编译过慢或崩溃。
            self.pipeline.denoiser3d.to(self.device_xla)

    @torch.no_grad()
    def predict(self, sample, static_cam: bool = False):
        """
        sample:
            {
                "length": (T,),
                "kp2d": (T, J, 3),
                "bbx_xys": (T, 3),
                "K_fullimg": (T, 3, 3),
                "cam_angvel": (T, 6),
                "f_imgseq": (T, C),
            }
        返回:
            {
                "joints_c": (T, 22, 3),
                "joints_w": (T, 22, 3),
                "joints_2d": (T, 22, 2),
            }
        """
        self._init_xla()

        # ========= 准备 CPU 上的输入 ========= #
        # Demo 中 batch_size=1，这里统一扩展 B 维到 1，和训练/评估 pipeline 保持一致。
        device_cpu = torch.device("cpu")

        length = sample["length"].to(device_cpu)  # 标量 T
        T = int(length.item())

        kp2d = sample["kp2d"].to(device_cpu)  # (T, J, 3)
        bbx_xys = sample["bbx_xys"].to(device_cpu)  # (T, 3)
        K_fullimg = sample["K_fullimg"].to(device_cpu)  # (T, 3, 3)
        cam_angvel = sample["cam_angvel"].to(device_cpu)  # (T, 6)
        f_imgseq = sample["f_imgseq"].to(device_cpu)  # (T, C)

        # 组装成与训练一致的 (B, L, *) 结构
        length_b = length.view(1)  # (1,)
        bbx_xys_b = bbx_xys.unsqueeze(0)  # (1, T, 3)
        K_fullimg_b = K_fullimg.unsqueeze(0)  # (1, T, 3, 3)
        cam_angvel_b = cam_angvel.unsqueeze(0)  # (1, T, 6)
        f_imgseq_b = f_imgseq.unsqueeze(0)  # (1, T, C)

        # 关键点归一化仍在 CPU 上做（纯张量算子）
        obs_b = normalize_kp2d(kp2d.unsqueeze(0), bbx_xys_b)  # (1, T, J, 3)

        # ========= XLA 上只跑 denoiser3d ========= #
        # 将网络关键信息搬到 XLA：obs/cliff_cam/cam_angvel/f_imgseq
        length_x = length_b.to(self.device_xla)
        bbx_xys_x = bbx_xys_b.to(self.device_xla)
        K_fullimg_x = K_fullimg_b.to(self.device_xla)
        cam_angvel_x = cam_angvel_b.to(self.device_xla)
        obs_x = obs_b.to(self.device_xla)
        f_imgseq_x = f_imgseq_b.to(self.device_xla)

        # cliff_cam 条件（BEDLAM 风格），完全是基础张量运算，适合放在 XLA
        cliff_cam_x = compute_bbox_info_bedlam(bbx_xys_x, K_fullimg_x)  # (1, T, 3)

        f_cam_angvel_x = cam_angvel_x
        if getattr(self.pipeline.args, "normalize_cam_angvel", False):
            # cam_angvel 归一化使用 pipeline 中缓存的统计量
            mean = self.pipeline.cam_angvel_mean.to(self.device_xla)
            std = self.pipeline.cam_angvel_std.to(self.device_xla)
            f_cam_angvel_x = (f_cam_angvel_x - mean) / std

        f_condition_x = {
            "obs": obs_x,
            "f_cliffcam": cliff_cam_x,
            "f_cam_angvel": f_cam_angvel_x,
            "f_imgseq": f_imgseq_x,
        }

        # 仅核心网络（denoiser3d）在 XLA 上前向
        model_output_x = self.pipeline.denoiser3d(length=length_x, **f_condition_x)

        # 标记一个 step，触发 XLA 图执行，然后将网络输出显式搬回 CPU；
        # 后续 decode + SMPLX 等都在 CPU 上完成，避免在 XLA 上编译复杂几何算子。
        xm.mark_step()

        pred_x = model_output_x["pred_x"].detach().cpu()  # (1, T, C)
        pred_cam = model_output_x["pred_cam"].detach().cpu()  # (1, T, 3)
        static_conf_logits = model_output_x["static_conf_logits"].detach().cpu()  # (1, T, *)

        # ========= CPU 上复用训练时的后处理逻辑 ========= #
        # 1) decode 到 SMPL 参数空间
        decode_dict = self.pipeline.endecoder.decode(pred_x)  # (1, T, C) -> dict

        # 2) 相机坐标系下的 SMPLX 参数（incam）
        pred_smpl_params_incam = {
            "body_pose": decode_dict["body_pose"],  # (1, T, 63)
            "betas": decode_dict["betas"],  # (1, T, 10)
            "global_orient": decode_dict["global_orient"],  # (1, T, 3)
            "transl": compute_transl_full_cam(pred_cam, bbx_xys_b, K_fullimg_b),  # (1, T, 3)
        }

        # 3) 全局坐标系下的 SMPLX 参数（world）
        pred_smpl_params_global = get_smpl_params_w_Rt_v2(
            global_orient_gv=decode_dict["global_orient_gv"],
            local_transl_vel=decode_dict["local_transl_vel"],
            global_orient_c=decode_dict["global_orient"],
            cam_angvel=cam_angvel_b,
        )
        pred_smpl_params_global = {
            "body_pose": decode_dict["body_pose"],
            "betas": decode_dict["betas"],
            **pred_smpl_params_global,
        }

        outputs = {
            "model_output": {
                "pred_x": pred_x,
                "pred_cam": pred_cam,
                "static_conf_logits": static_conf_logits,
            },
            # 为了与训练/评估阶段的后处理接口保持兼容，同时在顶层提供 static_conf_logits
            # 供 pp_static_joint / process_ik 直接访问。
            "static_conf_logits": static_conf_logits,
            "decode_dict": decode_dict,
            "pred_smpl_params_incam": pred_smpl_params_incam,
            "pred_smpl_params_global": pred_smpl_params_global,
        }

        # 4) 使用静态相机 / IK 等先验做后处理（完全在 CPU 上）
        if static_cam:
            outputs["pred_smpl_params_global"]["transl"] = pp_static_joint_cam(outputs, self.pipeline.endecoder)
        else:
            outputs["pred_smpl_params_global"]["transl"] = pp_static_joint(outputs, self.pipeline.endecoder)

        body_pose = process_ik(outputs, self.pipeline.endecoder)
        decode_dict["body_pose"] = body_pose
        outputs["pred_smpl_params_global"]["body_pose"] = body_pose
        outputs["pred_smpl_params_incam"]["body_pose"] = body_pose

        # 5) 最终 3D/2D joints（CPU 上 FK + 投影）
        pred_joints_c = self.pipeline.endecoder.fk_v2(**outputs["pred_smpl_params_incam"])  # (1, T, 22, 3)
        pred_joints_w = self.pipeline.endecoder.fk_v2(**outputs["pred_smpl_params_global"])  # (1, T, 22, 3)
        pred_kp2d_fullimg = perspective_projection(pred_joints_c, K_fullimg_b)  # (1, T, 22, 2)

        return {
            "joints_c": pred_joints_c[0].cpu(),  # (T, 22, 3)
            "joints_w": pred_joints_w[0].cpu(),  # (T, 22, 3)
            "joints_2d": pred_kp2d_fullimg[0].cpu(),  # (T, 22, 2)
        }

    def load_pretrained_model(self, ckpt_path):
        """Load pretrained checkpoint, and assign each weight to the corresponding part."""
        Log.info(f"[PL-Trainer TPU] Loading ckpt: {ckpt_path}")

        state_dict = torch.load(ckpt_path, "cpu")["state_dict"]
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if len(missing) > 0:
            Log.warn(f"Missing keys: {missing}")
        if len(unexpected) > 0:
            Log.warn(f"Unexpected keys: {unexpected}")


MainStore.store(
    name="gvhmr_pl_demo_tpu",
    node=builds(DemoPLTPU, pipeline="${pipeline}"),
    group="model/gvhmr",
)
