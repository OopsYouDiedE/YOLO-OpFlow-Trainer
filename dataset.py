import os
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from dataset import FlowDataset
from net import FlowLoss, FlowModel, YoloBackend

#光流可视化函数
def flow_to_image(flow):
    """
    将光流张量转换为RGB图像
    :param flow: [B, 2, H, W]
    :return: [B, 3, H, W]
    """
    u = flow[:, 0, :, :]
    v = flow[:, 1, :, :]
    max_flow = torch.sqrt(u**2 + v**2).max()
    angle = torch.atan2(v, u)
    angle = (angle + np.pi) / (2 * np.pi)  # 归一化到[0,1]
    magnitude = torch.sqrt(u**2 + v**2) / max_flow
    hsv = torch.stack([angle, torch.ones_like(angle), magnitude], dim=1)
    rgb = plt.cm.hsv(hsv.cpu().numpy())
    rgb = torch.from_numpy(rgb).permute(0, 3, 1, 2).float()
    return rgb


# 训练函数
def train(flow_model, feature_extractor, train_loader, val_loader, args):
    """
    训练光流模型

    Args:
        flow_model (FlowModel): 光流模型
        feature_extractor (YoloBackend): 特征提取模型
        train_loader (DataLoader): 训练数据加载器
        val_loader (DataLoader): 验证数据加载器
        args: 训练参数
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flow_model.to(device)
    feature_extractor.to(device)

    # 将特征提取器设置为评估模式，我们不训练它
    feature_extractor.eval()

    # 优化器，使用较大的初始学习率
    optimizer = Adam(
        filter(lambda p: p.requires_grad, flow_model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step, gamma=args.lr_gamma
    )

    # TensorBoard
    writer = SummaryWriter(args.log_dir)

    # 初始化损失函数
    criterion = FlowLoss().to(device)

    # 训练循环
    best_val_loss = float("inf")
    start_epoch = 0

    # 加载预训练模型（如果提供）
    if os.path.exists(args.pretrain):
        flow_model.load_state_dict(torch.load(args.pretrain))
        print("Loaded pretrain model")
    else:
        print("预训练模型不存在，使用随机初始化")

    # 加载checkpoint（如果存在）
    checkpoint_path = os.path.join(args.save_dir, 'checkpoint.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        flow_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        print(f"Resuming training from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        # 训练
        flow_model.train()
        train_loss = 0
        train_epe = 0

        for i, batch in enumerate(train_loader):
            # 获取图像
            img1 = batch["img1"].to(device)  # 形状应为 [B,3,640,640]
            img2 = batch["img2"].to(device)  # 形状应为 [B,3,640,640]
            gt_flow = batch["flow"].to(device)  # 形状应为 [B,2,640,640]

            # 使用特征提取器提取特征
            with torch.no_grad():
                features1 = feature_extractor._predict_backend(img1)
                features2 = feature_extractor._predict_backend(img2)

            # 前向传播 - 光流模型
            pred_flows = flow_model(features1, features2)

            # 计算损失
            loss, loss_details = criterion(pred_flows, gt_flow, img1)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_epe += loss_details["epe"].item()

            # 打印进度
            if i % args.print_freq == 0:
                print(
                    f"Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(train_loader)}], "
                    f"Loss: {loss.item():.4f}, EPE: {loss_details['epe'].item():.4f}"
                )

        avg_train_loss = train_loss / len(train_loader)
        avg_train_epe = train_epe / len(train_loader)
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("EPE/train", avg_train_epe, epoch)

        # 验证
        flow_model.eval()
        val_loss = 0
        val_epe = 0

        with torch.no_grad():
            for batch in val_loader:
                img1 = batch["img1"].to(device)
                img2 = batch["img2"].to(device)
                gt_flow = batch["flow"].to(device)
                features1 = feature_extractor._predict_backend(img1)
                features2 = feature_extractor._predict_backend(img2)
                pred_flows = flow_model(features1, features2)
                loss, loss_details = criterion(pred_flows, gt_flow, img1)
                val_loss += loss.item()
                val_epe += loss_details["epe"].item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_epe = val_epe / len(val_loader)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        writer.add_scalar("EPE/val", avg_val_epe, epoch)

        print(
            f"Epoch [{epoch+1}/{args.epochs}], Train Loss: {avg_train_loss:.4f}, Train EPE: {avg_train_epe:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, Val EPE: {avg_val_epe:.4f}"
        )

        # 更新学习率
        scheduler.step()

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                flow_model.state_dict(), os.path.join(args.save_dir, "best_model.pth")
            )

        # 定期保存检查点
        if (epoch + 1) % args.save_freq == 0:
            torch.save(
                flow_model.state_dict(),
                os.path.join(args.save_dir, f"model_epoch_{epoch+1}.pth"),
            )

        # 保存checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': flow_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
        }
        torch.save(checkpoint, checkpoint_path)

        # 可视化光流
        if epoch % args.vis_freq == 0:
            with torch.no_grad():
                # 训练集样本
                batch = next(iter(train_loader))
                img1 = batch["img1"].to(device)
                img2 = batch["img2"].to(device)
                gt_flow = batch["flow"].to(device)
                features1 = feature_extractor._predict_backend(img1)
                features2 = feature_extractor._predict_backend(img2)
                pred_flows = flow_model(features1, features2)
                pred_flow = pred_flows[0]  # 假设pred_flows是多尺度列表，取最大尺度
                pred_flow_img = flow_to_image(pred_flow)
                gt_flow_img = flow_to_image(gt_flow)
                writer.add_image("Train/Pred_Flow", make_grid(pred_flow_img), epoch)
                writer.add_image("Train/GT_Flow", make_grid(gt_flow_img), epoch)

                # 验证集样本
                batch = next(iter(val_loader))
                img1 = batch["img1"].to(device)
                img2 = batch["img2"].to(device)
                gt_flow = batch["flow"].to(device)
                features1 = feature_extractor._predict_backend(img1)
                features2 = feature_extractor._predict_backend(img2)
                pred_flows = flow_model(features1, features2)
                pred_flow = pred_flows[0]
                pred_flow_img = flow_to_image(pred_flow)
                gt_flow_img = flow_to_image(gt_flow)
                writer.add_image("Val/Pred_Flow", make_grid(pred_flow_img), epoch)
                writer.add_image("Val/GT_Flow", make_grid(gt_flow_img), epoch)

    # 保存最终模型
    torch.save(flow_model.state_dict(), os.path.join(args.save_dir, "final_model.pth"))
    writer.close()


# 划分训练集和验证集 (8:2)
def split_dataset(dataset, val_ratio=0.2, random_seed=42):
    """
    将数据集分为训练集和验证集

    Args:
        dataset: 完整数据集
        val_ratio: 验证集比例
        random_seed: 随机种子

    Returns:
        train_loader, val_loader: 训练和验证数据加载器
    """
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    split = int(np.floor(val_ratio * dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


if __name__ == "__main__":
    # 参数设置
    class Args:
        def __init__(self):
            self.lr = 1e-3  # 较大的初始学习率
            self.weight_decay = 1e-4
            self.lr_step = 10
            self.lr_gamma = 0.5
            self.epochs = 50
            self.print_freq = 20
            self.save_freq = 5
            self.vis_freq = 1  # 每隔多少个epoch可视化一次
            self.log_dir = "logs/flow"
            self.save_dir = "checkpoints/flow"
            self.data_root ="./"
            self.dataset_type='sintel'
            self.batch_size = 8
            self.num_workers = 4
            self.yolo_model_path = "FastSAM-s.pt"  # YoloBackend 模型路径
            self.pretrain = ""  # 预训练模型路径（可选）

    args = Args()

    # 创建目录
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    # 初始化模型
    flow_model = FlowModel()

    # 初始化特征提取器 - 使用 YoloBackend
    feature_extractor = YoloBackend(
        model_path=args.yolo_model_path, out_features=[4, 6, 9]
    )

    # 加载数据集并分割
    dataset = FlowDataset()  # 假设数据集已经编写好
    train_loader, val_loader = split_dataset(dataset)

    # 训练模型
    train(flow_model, feature_extractor, train_loader, val_loader, args)
