
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import FastSAM
import time

class Timer:
    """简单的计时器类，用于测量代码块的执行时间"""
    def __init__(self):
        self.time_cost_dict ={}
        self.last_start = None
    def start(self):
        """开始计时"""
        self.last_start = time.time()
    def end(self,name):
        self.time_cost_dict[name]=time.time()-self.last_start
        self.last_start=None
    def next(self,name):
        self.end(name)
        self.start()
class YoloBackend(nn.Module):
    def __init__(self, model_path="FastSAM-s.pt", out_features=[4, 6, 9]):
        """
        用于为光流模型提供图像编码

        Args:

        """
        super(YoloBackend, self).__init__()

        # 加载FastSAM模型
        model = FastSAM(model_path)
        self.yolo_backend = model.model.model
        self.out_features = out_features
        self.save = model.model.save

    def _predict_backend(self, x):
        """
        执行网络前向传播提取特征

        Args:
            x (torch.Tensor): 输入张量

        Returns:
            list: 各层特征输出的列表
        """
        y = []
        out = []
        # 遍历YOLO后端各层
        for m in self.yolo_backend:
            if m.f != -1:  # 如果不是来自上一层
                x = (
                    y[m.f]
                    if isinstance(m.f, int)
                    else [x if j == -1 else y[j] for j in m.f]
                )
            x = m(x)  # 执行当前层
            y.append(x if m.i in self.save else None)  # 保存输出
            if m.i in self.out_features:
                out.append(x)
            if m.i == max(self.out_features):
                return out
        return y

class EdgeAwareConv(nn.Module):
    """增强的边缘感知卷积模块"""
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(EdgeAwareConv, self).__init__()
        self.conv_spatial = nn.Conv2d(
            in_channels, 
            out_channels,
            kernel_size=kernel_size,
            padding=dilation * (kernel_size // 2),
            dilation=dilation
        )
        self.conv_edge = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        # 增强的边缘检测器
        self.edge_detector = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        edge_map = self.edge_detector(x)
        spatial_feat = self.conv_spatial(x)
        edge_feat = self.conv_edge(x)
        output = edge_map * edge_feat + (1 - edge_map) * spatial_feat
        return output, edge_map


class ImprovedFlowEstimatorBlock(nn.Module):
    """改进的光流估计块，考虑边缘感知"""
    def __init__(self, in_channels, hidden_channels=128):
        super(ImprovedFlowEstimatorBlock, self).__init__()
        self.conv1 = EdgeAwareConv(in_channels, hidden_channels)
        self.conv2 = EdgeAwareConv(hidden_channels, hidden_channels)
        self.conv3 = nn.Conv2d(hidden_channels, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.skip_conn = nn.Conv2d(in_channels, hidden_channels, 1) if in_channels != hidden_channels else nn.Identity()

    def forward(self, x):
        identity = self.skip_conn(x)
        x, edge_map1 = self.conv1(x)
        x = self.relu(x)
        x = x + identity
        identity = x
        x, edge_map2 = self.conv2(x)
        x = self.relu(x)
        x = x + identity
        flow = self.conv3(x)
        return flow, edge_map1 * edge_map2


class EfficientCostVolumeLayer(nn.Module):
    """高效代价体积计算层"""
    def __init__(self, max_displacement=4):
        super(EfficientCostVolumeLayer, self).__init__()
        self.max_displacement = max_displacement
        self.range = 2 * max_displacement + 1

    def forward(self, feat1, feat2):
        B, C, H, W = feat1.shape
        pad_size = self.max_displacement
        feat2_padded = F.pad(feat2, [pad_size] * 4)
        cost_volume = []
        for i in range(self.range):
            for j in range(self.range):
                if i == pad_size and j == pad_size:
                    continue
                feat2_slice = feat2_padded[:, :, i:i+H, j:j+W]
                correlation = torch.mean(feat1 * feat2_slice, dim=1, keepdim=True)
                cost_volume.append(correlation)
        cost_volume = torch.cat(cost_volume, dim=1)
        return cost_volume


class ChannelAttention(nn.Module):
    """通道注意力机制"""
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, 1)
        self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, 1)
        
    def forward(self, x):
        avg_out = self.fc2(F.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(F.relu(self.fc1(self.max_pool(x))))
        return F.sigmoid(avg_out + max_out)

class ContextNetwork(nn.Module):
    """增强的上下文感知网络"""
    def __init__(self, in_channels):
        super(ContextNetwork, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1, dilation=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 128, 3, padding=2, dilation=2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 128, 3, padding=4, dilation=4),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 96, 3, padding=8, dilation=8),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(96, 64, 3, padding=16, dilation=16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 32, 3, padding=1, dilation=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 2, 3, padding=1, dilation=1)
        )
        self.ca = ChannelAttention(128)
        
    def forward(self, x):
        x = self.convs[0](x)
        x = self.convs[1](x)
        x = x * self.ca(x)  # 应用通道注意力
        for layer in self.convs[2:]:
            x = layer(x)
        return x



class FlowModel(nn.Module):
    """改进的多尺度光流模型"""
    def __init__(self):
        super(FlowModel, self).__init__()
        self.timer=Timer()
        self.feat_channels = [512, 256, 128]  # 假设 YoloBackend 输出通道数
        
        # 特征处理
        self.conv_s1 = nn.Conv2d(self.feat_channels[0], 256, 1)  # 20x20
        self.conv_s2 = nn.Conv2d(self.feat_channels[1], 128, 1)  # 40x40
        self.conv_s3 = nn.Conv2d(self.feat_channels[2], 64, 1)   # 80x80

        # 代价体积计算
        self.cost_volume_s1 = EfficientCostVolumeLayer(max_displacement=4)
        self.cost_volume_s2 = EfficientCostVolumeLayer(max_displacement=4)
        self.cost_volume_s3 = EfficientCostVolumeLayer(max_displacement=4)

        # 光流估计块
        cv_channels = (4*2+1)**2 - 1
        self.flow_s1 = ImprovedFlowEstimatorBlock(256*2 + cv_channels, 128)
        self.flow_s2 = ImprovedFlowEstimatorBlock(128*2 + cv_channels + 2, 96)
        self.flow_s3 = ImprovedFlowEstimatorBlock(64*2 + cv_channels + 2, 64)

        # 上下文网络
        self.context_s3 = ContextNetwork(64*2 + 2)
        
        # 光流细化，结合图像特征
        self.refine_flow = nn.Sequential(
            EdgeAwareConv(2 + 3, 32),  # 光流+RGB图像
            nn.ReLU(inplace=True),
            EdgeAwareConv(32, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 3, padding=1)
        )
        
        # 边缘检测器
        self.edge_detector = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def _warp(self, x, flow):
        B, C, H, W = x.size()
        grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
        grid = torch.stack((grid_x, grid_y), dim=0).float().to(x.device)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)
        flow_grid = grid + flow
        flow_grid[:, 0, :, :] = 2.0 * flow_grid[:, 0, :, :] / (W - 1) - 1.0
        flow_grid[:, 1, :, :] = 2.0 * flow_grid[:, 1, :, :] / (H - 1) - 1.0
        flow_grid = flow_grid.permute(0, 2, 3, 1)
        return F.grid_sample(x, flow_grid, mode="bilinear", padding_mode="border", align_corners=True)

    def forward(self, img1, img2):
        t=self.timer
        """输入为两帧图像 [B, 3, H, W]"""
        t.start()
        features1 = self.feature_extractor._predict_backend(img1)
        features2 = self.feature_extractor._predict_backend(img2)
        f1_s3, f1_s2, f1_s1 = features1  # 80x80, 40x40, 20x20
        f2_s3, f2_s2, f2_s1 = features2
        t.next("特征提取")
        edge_map = self.edge_detector(img1)
        t.next("边缘检测")
        f1_s1 = self.conv_s1(f1_s1)
        f2_s1 = self.conv_s1(f2_s1)
        f1_s2 = self.conv_s2(f1_s2)
        f2_s2 = self.conv_s2(f2_s2)
        f1_s3 = self.conv_s3(f1_s3)
        f2_s3 = self.conv_s3(f2_s3)
        t.next("特征处理")
        # 粗尺度光流估计
        cost_s1 = self.cost_volume_s1(f1_s1, f2_s1)
        flow_s1_input = torch.cat([f1_s1, f2_s1, cost_s1], dim=1)
        flow_s1, edge_s1 = self.flow_s1(flow_s1_input)
        
        flow_s1_up = F.interpolate(flow_s1, scale_factor=2, mode="bilinear", align_corners=True) * 2.0
        edge_s1_up = F.interpolate(edge_s1, scale_factor=2, mode="bilinear", align_corners=True)
        warped_f2_s2 = self._warp(f2_s2, flow_s1_up)
        t.next("40x40光流")
        cost_s2 = self.cost_volume_s2(f1_s2, warped_f2_s2)
        flow_s2_input = torch.cat([f1_s2, warped_f2_s2, cost_s2, flow_s1_up], dim=1)
        flow_s2_delta, edge_s2 = self.flow_s2(flow_s2_input)
        flow_s2 = flow_s1_up + flow_s2_delta
        edge_combined = edge_s1_up * edge_s2

        flow_s2_up = F.interpolate(flow_s2, scale_factor=2, mode="bilinear", align_corners=True) * 2.0
        edge_s2_up = F.interpolate(edge_combined, scale_factor=2, mode="bilinear", align_corners=True)
        warped_f2_s3 = self._warp(f2_s3, flow_s2_up)
        t.next("80*80光流")
        cost_s3 = self.cost_volume_s3(f1_s3, warped_f2_s3)
        flow_s3_input = torch.cat([f1_s3, warped_f2_s3, cost_s3, flow_s2_up], dim=1)
        flow_s3_delta, edge_s3 = self.flow_s3(flow_s3_input)
        flow_s3 = flow_s2_up + flow_s3_delta

        context_input = torch.cat([f1_s3, warped_f2_s3, flow_s3], dim=1)
        context_flow = self.context_s3(context_input)
        flow_s3_refined = flow_s3 + context_flow
        edge_combined = edge_s2_up * edge_s3
        t.next("160*160光流")
        flow_160 = F.interpolate(flow_s3_refined, scale_factor=2, mode="bilinear", align_corners=True) * 2.0
        edge_160 = F.interpolate(edge_combined, scale_factor=2, mode="bilinear", align_corners=True)

        img1_160 = F.interpolate(img1, size=flow_160.shape[2:], mode="bilinear", align_corners=True)
        refine_input = torch.cat([flow_160, img1_160], dim=1)
        refined_flow_delta_160, refine_edge = self.refine_flow[0](refine_input)
        refined_flow_delta_160 = self.refine_flow[1](refined_flow_delta_160)
        refined_flow_delta_160, _ = self.refine_flow[2](refined_flow_delta_160)
        refined_flow_delta_160 = self.refine_flow[3](refined_flow_delta_160)
        refined_flow_delta_160 = self.refine_flow[4](refined_flow_delta_160)
        
        final_edge = edge_160 * refine_edge
        refined_flow_160 = flow_160 + refined_flow_delta_160
        
        flow_640 = F.interpolate(refined_flow_160, scale_factor=4, mode="bilinear", align_corners=True) * 4.0
        edge_640 = F.interpolate(final_edge, scale_factor=4, mode="bilinear", align_corners=True)
        t.end("640*640光流")
        return {"flow_160": refined_flow_160, "flow_640": flow_640, "edge_map": edge_640}


class FlowLoss(nn.Module):
    """改进的光流损失，关注内部平滑和边缘锐利度"""

    def __init__(
        self, 
        weights=(0.2, 0.4, 0.8, 1.0), 
        census_weight=0.3, 
        smooth_weight=0.5,
        edge_aware_weight=0.8
    ):
        super(FlowLoss, self).__init__()
        self.weights = weights
        self.census_weight = census_weight
        self.smooth_weight = smooth_weight
        self.edge_aware_weight = edge_aware_weight

    def forward(self, pred_flows, gt_flow, img1=None):
        """
        Args:
            pred_flows: 字典，包含不同尺度的预测光流和边缘图
            gt_flow: 真实光流 [B,2,H,W]
            img1: 第一帧图像，用于计算census loss和平滑损失（可选）
        Returns:
            loss: 总损失
        """
        total_loss = 0.0
        losses = {}

        # 计算160x160尺度的损失
        flow_160 = pred_flows["flow_160"]
        gt_flow_160 = F.interpolate(gt_flow, scale_factor=1/4, mode="bilinear", align_corners=True) / 4.0

        # L1 损失
        l1_loss_160 = F.l1_loss(flow_160, gt_flow_160)
        losses["l1_loss_160"] = l1_loss_160
        total_loss += self.weights[2] * l1_loss_160

        # 计算640x640尺度的损失
        flow_640 = pred_flows["flow_640"]
        l1_loss_640 = F.l1_loss(flow_640, gt_flow)
        losses["l1_loss_640"] = l1_loss_640
        total_loss += self.weights[3] * l1_loss_640
        
        # 获取预测的边缘图
        edge_map = pred_flows.get("edge_map", None)

        if img1 is not None:
            # Census损失（对光照变化鲁棒）
            if self.census_weight > 0:
                census_loss = self._census_loss(flow_640, gt_flow, img1)
                losses["census_loss"] = census_loss
                total_loss += self.census_weight * census_loss

            # 改进的边缘感知平滑损失
            if self.smooth_weight > 0:
                if edge_map is not None:
                    # 使用预测的边缘图进行平滑损失计算
                    smooth_loss = self._edge_aware_smoothness_loss(flow_640, edge_map)
                else:
                    # 使用图像梯度作为边缘指示器
                    smooth_loss = self._improved_smoothness_loss(flow_640, img1)
                
                losses["smooth_loss"] = smooth_loss
                total_loss += self.smooth_weight * smooth_loss
                
            # 边缘保持损失 - 鼓励在边缘处有更大的光流变化
            if self.edge_aware_weight > 0 and edge_map is not None:
                edge_aware_loss = self._edge_preservation_loss(flow_640, gt_flow, edge_map)
                losses["edge_aware_loss"] = edge_aware_loss
                total_loss += self.edge_aware_weight * edge_aware_loss

        # 计算EPE (End Point Error)
        with torch.no_grad():
            epe = torch.sqrt(torch.sum((flow_640 - gt_flow) ** 2, dim=1)).mean()
            losses["epe"] = epe

        return total_loss, losses

    def _census_loss(self, pred_flow, gt_flow, img1, patch_size=3):
        """Census损失 - 对光照变化有鲁棒性"""
        B, _, H, W = img1.shape
        patch_half = patch_size // 2
        
        # 更高效的Census变换实现
        def census_transform(x, patch_size=3):
            padded = F.pad(x, [patch_half, patch_half, patch_half, patch_half])
            patches = []
            center = padded[:, :, patch_half:-patch_half, patch_half:-patch_half]
            
            for i in range(patch_size):
                for j in range(patch_size):
                    if i == patch_half and j == patch_half:
                        continue
                    neighbor = padded[:, :, i:i+H, j:j+W]
                    patches.append((center > neighbor).float())
                    
            return torch.cat(patches, dim=1)
            
        census_img1 = census_transform(img1, patch_size)
        
        # 使用光流翘曲census特征
        warped_census_pred = self._warp(census_img1, pred_flow)
        warped_census_gt = self._warp(census_img1, gt_flow)
        
        # Hamming距离作为Census损失
        census_loss = torch.abs(warped_census_pred - warped_census_gt).mean()
        
        return census_loss
        
    def _improved_smoothness_loss(self, flow, img, alpha=10.0):
        """改进的平滑损失，考虑区域内部的一致性和边缘的锐利度"""
        # 计算图像梯度
        img_dx = torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])
        img_dy = torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])
        
        # 对梯度进行指数映射，增强对比度
        weight_x = torch.exp(-alpha * torch.mean(img_dx, dim=1, keepdim=True))
        weight_y = torch.exp(-alpha * torch.mean(img_dy, dim=1, keepdim=True))
        
        # 一阶梯度
        flow_dx = torch.abs(flow[:, :, :, :-1] - flow[:, :, :, 1:])
        flow_dy = torch.abs(flow[:, :, :-1, :] - flow[:, :, 1:, :])
        
        # 二阶梯度 - 鼓励区域内部的光滑性
        flow_dxx = torch.abs(flow[:, :, :, :-2] - 2 * flow[:, :, :, 1:-1] + flow[:, :, :, 2:])
        flow_dyy = torch.abs(flow[:, :, :-2, :] - 2 * flow[:, :, 1:-1, :] + flow[:, :, 2:, :])
        
        # 一阶梯度权重（边缘处小权重，内部大权重）
        smooth_1st = (weight_x * flow_dx).mean() + (weight_y * flow_dy).mean()
        
        # 二阶梯度权重（内部区域平滑性）
        # 注意：二阶梯度不在边缘处应用 
        weight_xx = weight_x[:, :, :, :-1] * weight_x[:, :, :, 1:]
        weight_yy = weight_y[:, :, :-1, :] * weight_y[:, :, 1:, :]
        
        smooth_2nd = (weight_xx * flow_dxx).mean() + (weight_yy * flow_dyy).mean()
        
        # 综合损失
        return smooth_1st + 0.5 * smooth_2nd
        
    def _edge_aware_smoothness_loss(self, flow, edge_map, lambda_smooth=1.0):
        """使用预测的边缘图的平滑损失"""
        # 反转边缘图，使其在边缘处为低值，内部为高值
        smoothness_weight = 1.0 - edge_map
        
        # 计算光流梯度
        flow_dx = torch.abs(flow[:, :, :, :-1] - flow[:, :, :, 1:])
        flow_dy = torch.abs(flow[:, :, :-1, :] - flow[:, :, 1:, :])
        
        # 在非边缘区域应用更强的平滑约束
        weight_x = smoothness_weight[:, :, :, :-1]
        weight_y = smoothness_weight[:, :, :-1, :]
        
        # 加权平滑损失
        smooth_loss = (weight_x * flow_dx).mean() + (weight_y * flow_dy).mean()
        
        return lambda_smooth * smooth_loss
        
    def _edge_preservation_loss(self, pred_flow, gt_flow, edge_map, lambda_edge=1.0):
        """鼓励在边缘处保留真实的光流变化"""
        # 只关注边缘区域的损失
        edge_mask = (edge_map > 0.5).float()
        
        # 计算边缘处的L1损失
        edge_loss = torch.abs(pred_flow - gt_flow) * edge_mask
        
        # 平均损失
        num_edge_pixels = edge_mask.sum() + 1e-6  # 避免除零
        edge_loss = edge_loss.sum() / num_edge_pixels
        
        return lambda_edge * edge_loss

    def _warp(self, x, flow):
        """使用光流对特征进行翘曲"""
        B, C, H, W = x.size()

        # 归一化网格坐标
        grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
        grid = torch.stack((grid_x, grid_y), dim=0).float().to(x.device)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)
        
        # 添加光流偏移
        flow_grid = grid + flow

        # 归一化到[-1,1]范围
        flow_grid[:, 0, :, :] = 2.0 * flow_grid[:, 0, :, :] / (W - 1) - 1.0
        flow_grid[:, 1, :, :] = 2.0 * flow_grid[:, 1, :, :] / (H - 1) - 1.0

        # 重排网格维度为B,H,W,2
        flow_grid = flow_grid.permute(0, 2, 3, 1)

        # 使用网格采样进行翘曲
        warped = F.grid_sample(
            x, flow_grid, mode="bilinear", padding_mode="border", align_corners=True
        )

        return warped
 