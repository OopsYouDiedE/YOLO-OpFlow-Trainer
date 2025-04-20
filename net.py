import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from hmac import new
from tkinter import NO
from ultralytics import YOLO, FastSAM
from ultralytics.nn.modules import Segment
from ultralytics.utils import ops
from ultralytics.engine.results import Results
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_
from dataset import FlowDataset
from ultralytics.utils.tal import TORCH_1_10, dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import fuse_conv_and_bn, smart_inference_mode

from ultralytics.nn.modules.block import DFL, SAVPE, BNContrastiveHead, ContrastiveHead, Proto, Residual, SwiGLUFFN
from ultralytics.nn.modules.conv import Conv, DWConv
from ultralytics.nn.modules.transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer
from ultralytics.nn.modules.utils import bias_init_with_prob, linear_init
from ultralytics.nn.modules.head import Segment,Detect

class YoloBasedDetFlowUnionModel(FastSAM):
    def __init__(self,model_path,):
        super().__init__()
        self.retina_masks = True  # 是否使用 RetinaNet 的掩码处理方式

        # 2. 获取原始 Detect head
        old_head = self.model.model[-1]
        assert isinstance(old_head, Segment), "最后一个模块不是 Segment，请确认模型是 YOLO-Seg 类型。"


        new_head = DetectFlowUnionHead(1, (320,640,640))
        old_head=self.model.model[-1]
        new_head.f=old_head.f
        new_head.i=old_head.i
        new_head.stride=old_head.stride
        print("old_head.stride",old_head.stride)
        self.model.model[-1] = new_head  # 替换为新 head

        # 4. 加载原始 Detect head 中的可兼容权重
        self.load_matching_weights(new_head, old_head)

        # 5. 冻结非光流分支参数
        self.freeze_non_flow_parts(new_head)
        self.head=new_head
        self.model.head = new_head         # ——> 同步到底层 FastSAM.model
        
    def load_matching_weights(self, new_head, old_head):
        """尝试从旧的 head 中加载匹配的参数到新 head 中"""
        old_sd = old_head.state_dict()
        new_sd = new_head.state_dict()

        matched_sd = {
            k: v for k, v in old_sd.items() if k in new_sd and new_sd[k].shape == v.shape
        }
        print("不匹配的键")
        print([k for k in old_sd if k not in new_sd])
        print(f"加载了 {len(matched_sd)}/{len(new_sd)} 个参数到 DetectFlowUnionHead。")
        
        new_head.load_state_dict(matched_sd, strict=False)

    def freeze_non_flow_parts(self, model):
        """只训练光流分支参数"""
        for name, param in model.named_parameters():
            if not ("cv_flow" in name or "cv_prob" in name):
                param.requires_grad = False



class DetectFlowUnionHead(Segment):
    """YOLO Detect head for detection models with optical flow branches."""


    def __init__(self, nc=1, ch=()):
        self.nl=len(ch)
        self.vxy_pram=1/4
        self.legacy=True #因为是使用FastSam的v8模型
        self.result_cache=[None] * self.nl
        super().__init__(nc,ch=ch)
        # base channels for detection and flow branches
        c2= max((16, ch[0] // 4, self.reg_max * 4))

        # Flow branch (x, y, vx, vy)
        self.cv_flow = nn.ModuleList(
            nn.Sequential(Conv(x*2, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 2 * self.reg_max, 1)) for x in ch
        )



        self.flow_cache = [None] * self.nl

        
    def forward(self, x):
        if self.flow_cache[0] ==None:
            self.set_cache(x)
            if self.training:
                return None
        p = self.proto(x[0])  # mask protos
        bs = p.shape[0]  # batch size
        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients

        """Concatenates and returns predicted bounding boxes and class probabilities."""
        cache=self.flow_cache
        self.set_cache(x)

        for i in range(self.nl):
            det_feat = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
            
            merge=torch.cat((x[i],cache[i]),1)
            flow_feat = self.cv_flow[i](merge)
            self.result_cache[i]=flow_feat
        
        return super().forward(x)


    def set_cache(self, x):
        self.flow_cache = [f.detach() for f in x]

    def reset_cache(self):
        self.flow_cache = [None] * self.nl





def compute_flow_loss(dets: torch.Tensor,
                      pred_flows: torch.Tensor,
                      gt_flow: torch.Tensor) -> torch.Tensor:
    """
    基于模型输出的检测坐标和预测光流，与真实光流图计算 L1 损失；

    dets:       Tensor, shape=[B, N, 4]，前两维为 (x, y) 像素坐标
    pred_flows: Tensor, shape=[B, N, 2]，模型预测的 (vx, vy)
    gt_flow:    Tensor, shape=[B, 2, H, W]，真实光流图
    """

    B, N, _ = dets.shape
    print
    _, C, H, W = gt_flow.shape

    # 提取 (x, y) 像素坐标
    coords = dets[..., :2]  # [B, N, 2]

    # 归一化到 [-1,1]
    norm_x = (coords[..., 0] / (W - 1)) * 2 - 1
    norm_y = (coords[..., 1] / (H - 1)) * 2 - 1
    grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(2)  # [B, N, 1, 2]

    # 从真实光流图中采样
    sampled_gt = F.grid_sample(gt_flow, grid, align_corners=True)  # [B, 2, N, 1]
    sampled_gt = sampled_gt.squeeze(-1).permute(0, 2, 1)             # [B, N, 2]

    # 计算 L1 损失
    loss = F.l1_loss(pred_flows, sampled_gt, reduction='mean')
    return loss


def train(model: YoloBasedDetFlowUnionModel,
          dataloader: DataLoader,
          optimizer: optim.Optimizer,
          device: torch.device,
          epochs: int = 10):
    model.to(device)
    model.val()
    #model.model.train()  # 冻结非光流分支，训练光流分支

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch in dataloader:
            img1 = batch['img1'].to(device)
            img2 = batch['img2'].to(device)
            gt_flow = batch['flow'].to(device)
            model.head.reset_cache()  # 重置缓存
            optimizer.zero_grad()
            # 前向第二帧用于缓存特征
            model.model(img2)
            # 前向第一帧得到输出
            res = model.model(img1)
            dets=res[0][0]
            def o(x):
                if isinstance(x,torch.Tensor):
                    return x.shape
                return [o(xi) for xi in x]
            print(o(res))
            pred_flows = model.head.flow_cache  # 获取光流分支的输出
            # 假设 outputs 返回 (dets, pred_flows)

            loss = compute_flow_loss(dets, pred_flows, gt_flow)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * img1.size(0)

        avg_loss = total_loss / len(dataloader.dataset)
        print(f"Epoch {epoch}/{epochs} - Flow Loss: {avg_loss:.6f}")


if __name__ == "__main__":
    # 配置区
    data_root = "./"
    batch_size = 4
    lr = 1e-4
    epochs = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = YoloBasedDetFlowUnionModel("FastSAM-s.pt")
    dataset = FlowDataset(data_root, dataset_type="sintel", split="train")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    train(model, loader, optimizer, device, epochs)
