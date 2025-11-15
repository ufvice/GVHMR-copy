import torch
from hmr4d.network.hmr2 import load_hmr2, HMR2


from hmr4d.utils.video_io_utils import read_video_np
import cv2
import numpy as np

from hmr4d.network.hmr2.utils.preproc import crop_and_resize, IMAGE_MEAN, IMAGE_STD
from tqdm import tqdm


def get_batch(input_path, bbx_xys, img_ds=0.5, img_dst_size=256, path_type="video"):
    if path_type == "video":
        imgs = read_video_np(input_path, scale=img_ds)
    elif path_type == "image":
        imgs = cv2.imread(str(input_path))[..., ::-1]
        imgs = cv2.resize(imgs, (0, 0), fx=img_ds, fy=img_ds)
        imgs = imgs[None]
    elif path_type == "np":
        assert isinstance(input_path, np.ndarray)
        assert img_ds == 1.0  # this is safe
        imgs = input_path

    gt_center = bbx_xys[:, :2]
    gt_bbx_size = bbx_xys[:, 2]

    # Blur image to avoid aliasing artifacts
    if True:
        gt_bbx_size_ds = gt_bbx_size * img_ds
        ds_factors = ((gt_bbx_size_ds * 1.0) / img_dst_size / 2.0).numpy()
        imgs = np.stack(
            [
                # gaussian(v, sigma=(d - 1) / 2, channel_axis=2, preserve_range=True) if d > 1.1 else v
                cv2.GaussianBlur(v, (5, 5), (d - 1) / 2) if d > 1.1 else v
                for v, d in zip(imgs, ds_factors)
            ]
        )

    # Output
    imgs_list = []
    bbx_xys_ds_list = []
    for i in range(len(imgs)):
        img, bbx_xys_ds = crop_and_resize(
            imgs[i],
            gt_center[i] * img_ds,
            gt_bbx_size[i] * img_ds,
            img_dst_size,
            enlarge_ratio=1.0,
        )
        imgs_list.append(img)
        bbx_xys_ds_list.append(bbx_xys_ds)
    imgs = torch.from_numpy(np.stack(imgs_list))  # (F, 256, 256, 3), RGB
    bbx_xys = torch.from_numpy(np.stack(bbx_xys_ds_list)) / img_ds  # (F, 3)

    imgs = ((imgs / 255.0 - IMAGE_MEAN) / IMAGE_STD).permute(0, 3, 1, 2)  # (F, 3, 256, 256
    return imgs, bbx_xys


class Extractor:
    def __init__(self, tqdm_leave: bool = True, device: torch.device | None = None):
        # 允许外部传入 device（例如 XLA 设备）；若不指定则按照 CUDA→CPU 的优先级选择。
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.extractor: HMR2 = load_hmr2().to(self.device).eval()
        self.tqdm_leave = tqdm_leave

    def extract_video_features(self, video_path, bbx_xys, img_ds=0.5):
        """
        img_ds makes the image smaller, which is useful for faster processing
        """
        # Get the batch
        if isinstance(video_path, str):
            imgs, bbx_xys = get_batch(video_path, bbx_xys, img_ds=img_ds)
        else:
            assert isinstance(video_path, torch.Tensor)
            imgs = video_path

        # Inference
        F, _, H, W = imgs.shape  # (F, 3, H, W)

        # 只把纯卷积/Transformer 前向放到目标 device（包含 XLA）；
        # 视频解码与裁剪仍在 CPU 完成，避免不必要的 host<->device 往返。
        imgs = imgs.to(self.device)
        batch_size = 16  # 对于 GPU/TPU 约 5GB 显存；在 CPU 环境可根据需要适当调小

        is_xla = hasattr(self.device, "type") and str(self.device.type) == "xla"
        features = []
        for j in tqdm(range(0, F, batch_size), desc="HMR2 Feature", leave=self.tqdm_leave):
            imgs_batch = imgs[j : j + batch_size]
            B = imgs_batch.shape[0]
            if B == 0:
                continue

            pad_len = 0
            # 在 XLA 上，为避免最后一个 batch 因 batch_size 变化触发额外编译，
            # 将最后一批补齐到固定 batch_size，再在输出阶段裁掉 padding。
            if is_xla and B < batch_size:
                pad_len = batch_size - B
                imgs_batch = torch.cat(
                    [imgs_batch, imgs_batch[-1:].expand(pad_len, -1, -1, -1)],
                    dim=0,
                )

            with torch.no_grad():
                feature = self.extractor({"img": imgs_batch})

                if pad_len > 0:
                    feature = feature[:B]

                if is_xla:
                    # 在 XLA 上先保持为 XLA Tensor，统一在函数末尾一次性搬回 CPU，
                    # 减少多次 host<->device 同步带来的开销。
                    features.append(feature)
                else:
                    features.append(feature.detach().cpu())

        features = torch.cat(features, dim=0)  # (F, 1024)
        if is_xla:
            # 对于 XLA：此处触发图执行并拷贝回 CPU。
            features = features.detach().cpu()
        else:
            features = features.clone()

        return features
