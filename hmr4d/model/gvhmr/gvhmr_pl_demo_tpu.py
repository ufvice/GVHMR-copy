import torch
import pytorch_lightning as pl
from hydra.utils import instantiate

from hmr4d.utils.pylogger import Log
from hmr4d.configs import MainStore, builds

from hmr4d.utils.geo.hmr_cam import normalize_kp2d

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
            self.to(self.device_xla)

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

        length = sample["length"].to(self.device_xla)
        kp2d = sample["kp2d"].to(self.device_xla)
        bbx_xys = sample["bbx_xys"].to(self.device_xla)
        K_fullimg = sample["K_fullimg"].to(self.device_xla)
        cam_angvel = sample["cam_angvel"].to(self.device_xla)
        f_imgseq = sample["f_imgseq"].to(self.device_xla)

        batch = {
            "length": length[None],
            "obs": normalize_kp2d(kp2d, bbx_xys)[None],
            "bbx_xys": bbx_xys[None],
            "K_fullimg": K_fullimg[None],
            "cam_angvel": cam_angvel[None],
            "f_imgseq": f_imgseq[None],
        }

        outputs = self.pipeline.forward(batch, train=False, postproc=True, static_cam=static_cam)

        # joints in camera / world coords and 2D projections are expected
        joints_c = outputs["pred_joints_c"][0]  # (T, 22, 3)
        joints_w = outputs["pred_joints_w"][0]  # (T, 22, 3)
        joints_2d = outputs["pred_kp2d_fullimg"][0]  # (T, 22, 2)

        # 将结果拉回 CPU，方便后续组织 label
        out_cpu = xm._fetch(
            {
                "joints_c": joints_c,
                "joints_w": joints_w,
                "joints_2d": joints_2d,
            }
        )
        xm.mark_step()
        return out_cpu

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

