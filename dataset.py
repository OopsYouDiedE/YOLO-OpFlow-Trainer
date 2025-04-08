import os
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
import cv2
import numpy as np
import requests
from tqdm import tqdm
import zipfile


class FlowDataset(Dataset):
    """
    光流数据集加载器，支持Sintel和FlyingChairs数据集，支持自动下载
    """

    def __init__(self, data_root, dataset_type="sintel", split="train", augment=True):
        if dataset_type == "sintel":
            self.data_root = os.path.join(data_root, "Sintel")
        elif dataset_type == "flyingchairs":
            self.data_root = os.path.join(data_root, "FlyingChairs")
        else:
            raise ValueError(f"不支持的数据集类型: {dataset_type}")
        self.split = split
        self.augment = augment
        self.samples = []

        # 创建数据根目录（如果不存在）
        os.makedirs(self.data_root, exist_ok=True)

        # 自动下载缺失的数据集
        if dataset_type == "sintel":
            if not self._check_sintel_exists():
                print("检测到Sintel数据集缺失，开始自动下载...")
                self._download_sintel()
        elif dataset_type == "flyingchairs":
            if not self._check_flyingchairs_exists():
                print("检测到FlyingChairs数据集缺失，开始自动下载...")
                self._download_flyingchairs()

        # 加载数据集
        if dataset_type == "sintel":
            self._load_sintel_data()
        elif dataset_type == "flyingchairs":
            self._load_flyingchairs_data()

        print(f"已加载 {dataset_type} 数据集，共 {len(self.samples)} 对样本")

    def _check_sintel_exists(self):
        """检查Sintel数据集完整性"""
        required = ["training/clean", "training/final", "training/flow"]
        return all(os.path.exists(os.path.join(self.data_root, p)) for p in required)

    def _check_flyingchairs_exists(self):
        """检查FlyingChairs数据集完整性"""
        data_dir = os.path.join(self.data_root, "data")
        if not os.path.exists(data_dir):
            return False
        # 随机检查10个样本
        img1_files = [f for f in os.listdir(data_dir) if f.endswith("_img1.ppm")][:10]
        if not img1_files:
            return False
        for f in img1_files:
            base = f[:-9]
            if not all(
                os.path.exists(os.path.join(data_dir, f"{base}{suffix}"))
                for suffix in ["_img2.ppm", "_flow.flo"]
            ):
                return False
        return True

    def _download_sintel(self):
        """下载并解压Sintel数据集"""
        url = "http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip"
        zip_path = os.path.join(self.data_root, "Sintel.zip")
        self._download_file(url, zip_path)

        # 解压文件
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(self.data_root)

        os.remove(zip_path)

    def _download_flyingchairs(self):
        """下载并解压FlyingChairs数据集"""
        url = (
            "https://lmb.informatik.uni-freiburg.de/data/FlyingChairs/FlyingChairs.zip"
        )
        zip_path = os.path.join(self.data_root, "FlyingChairs.zip")
        self._download_file(url, zip_path)

        # 解压文件
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(self.data_root)

        os.remove(zip_path)

    def _download_file(self, url, save_path):
        """带进度条的文件下载函数"""
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        progress_bar = tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc=f"下载 {os.path.basename(save_path)}",
        )

        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))
        progress_bar.close()

    def _load_sintel_data(self):
        """加载Sintel数据集"""
        # Sintel数据集目录结构：
        # - training
        #   - clean/final (两种渲染风格)
        #   - flow
        base_dir = os.path.join(self.data_root, "training")

        # 选择clean或final渲染风格
        styles = ["clean", "final"]

        for style in styles:
            img_dir = os.path.join(base_dir, style)
            flow_dir = os.path.join(base_dir, "flow")

            # 遍历所有场景
            for scene in sorted(os.listdir(img_dir)):
                scene_img_dir = os.path.join(img_dir, scene)
                scene_flow_dir = os.path.join(flow_dir, scene)

                if not os.path.isdir(scene_img_dir):
                    continue

                # 获取所有图像并排序
                images = sorted(
                    [f for f in os.listdir(scene_img_dir) if f.endswith(".png")]
                )

                # 对于每对连续帧
                for i in range(len(images) - 1):
                    img1_path = os.path.join(scene_img_dir, images[i])
                    img2_path = os.path.join(scene_img_dir, images[i + 1])
                    flow_path = os.path.join(scene_flow_dir, f"frame_{i:04d}.flo")
                    if not os.path.exists(flow_path):
                        print(f"警告: 光流文件 {flow_path} 不存在，跳过该样本")
                        continue
                    if os.path.exists(flow_path):
                        self.samples.append(
                            {
                                "img1_path": img1_path,
                                "img2_path": img2_path,
                                "flow_path": flow_path,
                                "style": style,
                            }
                        )

    def _load_flyingchairs_data(self):
        """加载FlyingChairs数据集（修复文件存在性检查）"""
        img_dir = os.path.join(self.data_root, "data")
        img1_files = sorted([f for f in os.listdir(img_dir) if f.endswith("_img1.ppm")])

        for img1_file in img1_files:
            base_name = img1_file[:-9]
            img2_file = f"{base_name}_img2.ppm"
            flow_file = f"{base_name}_flow.flo"

            img1_path = os.path.join(img_dir, img1_file)
            img2_path = os.path.join(img_dir, img2_file)
            flow_path = os.path.join(img_dir, flow_file)

            # 修复逻辑：两个文件必须同时存在
            if not (os.path.exists(img2_path) and os.path.exists(flow_path)):
                print(f"警告: 缺失文件 {img2_path} 或 {flow_path}，跳过该样本")
                continue

            self.samples.append(
                {"img1_path": img1_path, "img2_path": img2_path, "flow_path": flow_path}
            )

    def _read_flow(self, flow_path):
        """读取光流文件(.flo格式)"""
        with open(flow_path, "rb") as f:
            header = np.fromfile(f, np.float32, count=1)
            if header != 202021.25:
                raise Exception("无效的光流文件格式")

            width = np.fromfile(f, np.int32, count=1)[0]
            height = np.fromfile(f, np.int32, count=1)[0]

            flow = np.fromfile(f, np.float32, count=width * height * 2)
            flow = flow.reshape((height, width, 2))

        return flow

    def _preprocess_image(self, img_path):
        """预处理图像"""
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 中央裁剪到最近的32的倍数长宽
        h, w = img.shape[:2]
        crop_h = (h // 32) * 32  # 计算最近的32的倍数
        crop_w = (w // 32) * 32

        # 计算裁剪区域
        y0 = (h - crop_h) // 2
        x0 = (w - crop_w) // 2
        img = img[y0 : y0 + crop_h, x0 : x0 + crop_w]

        # 转换为CHW格式并标准化
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0

        return img

    def _augment_data(self, img1, img2, flow):
        """数据增强"""
        # 随机裁剪，先禁用，感觉意义不明
        """h, w = flow.shape[1:3]
        crop_h, crop_w = 448, 448  # 裁剪尺寸

        if h > crop_h and w > crop_w:
            y0 = np.random.randint(0, h - crop_h)
            x0 = np.random.randint(0, w - crop_w)

            img1 = img1[:, y0:y0+crop_h, x0:x0+crop_w]
            img2 = img2[:, y0:y0+crop_h, x0:x0+crop_w]
            flow = flow[:, y0:y0+crop_h, x0:x0+crop_w]"""

        # 随机水平翻转
        if np.random.rand() > 0.5:
            img1 = np.flip(img1, axis=2).copy()
            img2 = np.flip(img2, axis=2).copy()
            flow = np.flip(flow, axis=2).copy()
            flow[0, :, :] *= -1  # 翻转x方向光流

        # 随机垂直翻转
        if np.random.rand() > 0.5:
            img1 = np.flip(img1, axis=1).copy()
            img2 = np.flip(img2, axis=1).copy()
            flow = np.flip(flow, axis=1).copy()
            flow[1, :, :] *= -1  # 翻转y方向光流

        return img1, img2, flow

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 读取图像
        img1 = self._preprocess_image(sample["img1_path"])
        img2 = self._preprocess_image(sample["img2_path"])

        # 读取光流
        flow = self._read_flow(sample["flow_path"])
        # 转换为CHW格式
        flow = flow.transpose(2, 0, 1).astype(np.float32)

        # 中央裁剪到最近的32的倍数长宽（与图像裁剪方式一致）
        h, w = flow.shape[1:3]
        crop_h = (h // 32) * 32
        crop_w = (w // 32) * 32

        y0 = (h - crop_h) // 2
        x0 = (w - crop_w) // 2
        flow = flow[:, y0 : y0 + crop_h, x0 : x0 + crop_w]

        return {"img1": img1, "img2": img2, "flow": flow}
