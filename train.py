import os
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import argparse
import torch.cuda.amp as amp  # 引入混合精度训练模块

from dataset import FlowDataset
from net import FlowLoss, FlowModel, YoloBackend

# 光流可视化函数（保持不变）
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

# 优化后的训练函数
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
    feature_extractor.eval()

    optimizer = Adam(
        filter(lambda p: p.requires_grad, flow_model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step, gamma=args.lr_gamma
    )
    writer = SummaryWriter(args.log_dir)
    criterion = FlowLoss().to(device)

    best_val_loss = float("inf")
    start_epoch = 0

    # 加载预训练模型或检查点
    if args.pretrain and os.path.exists(args.pretrain):
        flow_model.load_state_dict(torch.load(args.pretrain))
        print("已加载预训练模型")
    else:
        print("未指定预训练模型或路径不存在，使用随机初始化")

    checkpoint_path = os.path.join(args.save_dir, 'checkpoint.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        flow_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        print(f"从第 {start_epoch} 个 epoch 恢复训练")

    # 初始化混合精度训练的 GradScaler
    scaler = amp.GradScaler()

    for epoch in range(start_epoch, args.epochs):
        flow_model.train()
        Bradshaw_loss = 0
        train_epe = 0

        for i, batch in enumerate(train_loader):
            img1 = batch["img1"].to(device)
            img2 = batch["img2"].to(device)
            gt_flow = batch["flow"].to(device)

            with torch.no_grad():
                features1 = feature_extractor._predict_backend(img1)
                features2 = feature_extractor._predict_backend(img2)

            # 使用混合精度训练
            with amp.autocast():
                pred_flows = flow_model(features1, features2, img1)
                loss, loss_details = criterion(pred_flows, gt_flow, img1)

            optimizer.zero_grad()
            scaler.scale(loss).backward()  # 缩放损失并反向传播
            scaler.step(optimizer)         # 更新优化器
            scaler.update()                # 更新 scaler

            train_loss += loss.item()
            train_epe += loss_details["epe"].item()

            if i % args.print_freq == 0:
                print(
                    f"Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(train_loader)}], "
                    f"Loss: {loss.item():.4f}, EPE: {loss_details['epe'].item():.4f}"
                )

        avg_train_loss = train_loss / len(train_loader)
        avg_train_epe = train_epe / len(train_loader)
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("EPE/train", avg_train_epe, epoch)

        # 验证阶段（减少不必要梯度计算）
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
                with amp.autocast():  # 验证阶段也使用混合精度
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

        scheduler.step()

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                flow_model.state_dict(), os.path.join(args.save_dir, "best_model.pth")
            )

        # 定期保存模型
        if (epoch + 1) % args.save_freq == 0:
            torch.save(
                flow_model.state_dict(),
                os.path.join(args.save_dir, f"model_epoch_{epoch+1}.pth"),
            )

        # 保存检查点
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': flow_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
        }
        torch.save(checkpoint, checkpoint_path)

        # 可视化（保持不变，但可减少频率）
        if epoch % args.vis_freq == 0:
            with torch.no_grad():
                batch = next(iter(train_loader))
                img1 = batch["img1"].to(device)
                img2 = batch["img2"].to(device)
                gt_flow = batch["flow"].to(device)
                features1 = feature_extractor._predict_backend(img1)
                features2 = feature_extractor._predict_backend(img2)
                with amp.autocast():
                    pred_flows = flow_model(features1, features2)
                pred_flow = pred_flows[0]
                pred_flow_img = flow_to_image(pred_flow)
                gt_flow_img = flow_to_image(gt_flow)
                writer.add_image("Train/Pred_Flow", make_grid(pred_flow_img), epoch)
                writer.add_image("Train/GT_Flow", make_grid(gt_flow_img), epoch)

                batch = next(iter(val_loader))
                img1 = batch["img1"].to(device)
                img2 = batch["img2"].to(device)
                gt_flow = batch["flow"].to(device)
                features1 = feature_extractor._predict_backend(img1)
                features2 = feature_extractor._predict_backend(img2)
                with amp.autocast():
                    pred_flows = flow_model(features1, features2)
                pred_flow = pred_flows[0]
                pred_flow_img = flow_to_image(pred_flow)
                gt_flow_img = flow_to_image(gt_flow)
                writer.add_image("Val/Pred_Flow", make_grid(pred_flow_img), epoch)
                writer.add_image("Val/GT_Flow", make_grid(gt_flow_img), epoch)

    torch.save(flow_model.state_dict(), os.path.join(args.save_dir, "final_model.pth"))
    writer.close()

# 数据集划分函数（保持不变）
def split_dataset(dataset, batch_size, num_workers, val_ratio=0.2, random_seed=42):
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
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练光流模型')
    parser.add_argument('--lr', type=float, default=1e-3, help='初始学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减系数')
    parser.add_argument('--lr_step', type=int, default=10, help='学习率衰减步长')
    parser.add_argument('--lr_gamma', type=float, default=0.5, help='学习率衰减因子')
    parser.add_argument('--epochs', type=int, default=50, help='训练的总 epoch 数')
    parser.add_argument('--print_freq', type=int, default=20, help='打印训练进度的频率')
    parser.add_argument('--save_freq', type=int, default=5, help='保存模型的频率')
    parser.add_argument('--vis_freq', type=int, default=1, help='可视化光流的频率')
    parser.add_argument('--log_dir', type=str, default='logs/flow', help='TensorBoard 日志目录')
    parser.add_argument('--save_dir', type=str, default='checkpoints/flow', help='模型保存目录')
    parser.add_argument('--data_root', type=str, default='./', help='数据集根目录')
    parser.add_argument('--dataset_type', type=str, default='sintel', help='数据集类型')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载的 worker 数量')
    parser.add_argument('--yolo_model_path', type=str, default='FastSAM-s.pt', help='Yolo 模型路径')
    parser.add_argument('--pretrain', type=str, default='', help='预训练模型路径（可选）')

    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    flow_model = FlowModel()
    feature_extractor = YoloBackend(
        model_path=args.yolo_model_path, out_features=[4, 6, 9]
    )

    dataset = FlowDataset(args.data_root, args.dataset_type)
    train_loader, val_loader = split_dataset(dataset, args.batch_size, args.num_workers)

    train(flow_model, feature_extractor, train_loader, val_loader, args)
