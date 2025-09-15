#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------
#   Project:  CLIP Demo
#   File:     clip_infer.py
#   Author:   DeZhen Wang, from QUT
#   Date:     2025-08
#   Implementation: Based on open_clip and PyTorch
#   Description: Simple example for extracting image features
# -----------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from Models.SFGNet.some_functions import LayerNorm2d, ConvMlp


class InteractiveStructureEnhancementBlock(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        # 特征归一化
        self.norm_main = LayerNorm2d(dim)
        self.norm_aux = LayerNorm2d(dim)

        # 特征投影
        self.main_proj = nn.Conv2d(dim, dim, 1)
        self.aux_proj = nn.Conv2d(dim, dim, 1)

        # 注意力缩放
        self.scale = dim ** -0.5

        # 动态权重生成
        self.weight_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.ReLU(),
            nn.Conv2d(dim // 4, 1, 1),
            nn.Sigmoid()
        )

        # 输出投影
        self.out_proj = nn.Conv2d(dim, dim, 1)

        # 前馈网络
        self.ffn = ConvMlp(dim, dim)

    def forward(self, main_feat, aux_feats):
        """
        main_feat: 主特征 [B, C, H, W]
        aux_feats: 辅助特征列表 [B, C, H, W] * N
        """
        B, C, H, W = main_feat.shape

        # 归一化
        main_norm = self.norm_main(main_feat)
        aux_norms = [self.norm_aux(feat) for feat in aux_feats]

        # 投影
        main_q = self.main_proj(main_norm)
        aux_kvs = [self.aux_proj(aux) for aux in aux_norms]

        # 计算注意力和融合
        enhanced_feats = []
        for aux_kv in aux_kvs:
            # 空间注意力
            q_flat = main_q.flatten(2).transpose(1, 2)  # [B, HW, C]
            k_flat = aux_kv.flatten(2)  # [B, C, HW]

            attn = (q_flat @ k_flat) * self.scale  # [B, HW, HW]
            attn = F.softmax(attn, dim=-1)

            v_flat = aux_kv.flatten(2).transpose(1, 2)  # [B, HW, C]
            out_flat = attn @ v_flat  # [B, HW, C]
            out = out_flat.transpose(1, 2).reshape(B, C, H, W)

            enhanced_feats.append(out)

        # 动态加权融合
        if len(enhanced_feats) > 1:
            weights = [self.weight_gen(feat) for feat in enhanced_feats]
            weights = torch.cat(weights, dim=1)
            weights = F.softmax(weights, dim=1)

            fused_feat = 0
            for i, feat in enumerate(enhanced_feats):
                fused_feat += feat * weights[:, i:i + 1]
        else:
            fused_feat = enhanced_feats[0]

        # 输出投影和残差连接
        output = main_feat + self.out_proj(fused_feat)

        # 前馈网络和残差连接
        output = output + self.ffn(output)

        return output


class BidirectionalIneractiveNetwork(nn.Module):
    """创新型特征金字塔网络 - 简洁高效设计"""

    def __init__(self, in_channels=[768, 768, 768], out_channels=[256, 512, 1024]):
        super(BidirectionalIneractiveNetwork, self).__init__()


        self.txt_proj = nn.Sequential(
            nn.Linear(in_channels[2], out_channels[2]),
            nn.LayerNorm(out_channels[2]),
            nn.ReLU(inplace=True)
        )


        self.feature_gate = nn.Sequential(
            nn.Linear(in_channels[2], 3),
            nn.Sigmoid()
        )


        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels[i], out_channels[i], kernel_size=1)
            for i in range(3)
        ])

        self.top_down = nn.ModuleList([
            nn.Conv2d(out_channels[i + 1], out_channels[i], kernel_size=1)
            for i in range(2)
        ])

        self.bottom_up = nn.ModuleList([
            nn.Conv2d(out_channels[i], out_channels[i + 1], kernel_size=3, stride=2, padding=1)
            for i in range(2)
        ])

        self.fusion = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels[i], out_channels[i], kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels[i]),
                nn.ReLU(inplace=True)
            ) for i in range(3)
        ])

        # 创新点5: 通道重校准 - 统一通道数并重校准特征
        self.channel_calibration = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels[i], out_channels[1], kernel_size=1),
                nn.BatchNorm2d(out_channels[1]),
                nn.ReLU(inplace=True)
            ) for i in range(3)
        ])

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        self.output_layer = nn.Sequential(
            nn.Conv2d(out_channels[1], out_channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels[1]),
            nn.ReLU(inplace=True)
        )

    def forward(self, imgs, state):
        """
        imgs: 三个视觉特征 [v3, v4, v5]
        state: 文本特征 [B, 768]
        """
        v3, v4, v5 = imgs
        batch_size = v3.shape[0]

        text_feat = self.txt_proj(state).unsqueeze(-1).unsqueeze(-1)  # [B, 1024, 1, 1]

        gates = self.feature_gate(state).unsqueeze(-1).unsqueeze(-1)  # [B, 3, 1, 1]

        c3 = self.lateral_convs[0](v3) * gates[:, 0:1]
        c4 = self.lateral_convs[1](v4) * gates[:, 1:2]
        c5 = self.lateral_convs[2](v5) * gates[:, 2:3]


        c5 = c5 * text_feat

        p5 = c5
        p4 = self.fusion[1](
            c4 + self.top_down[1](F.interpolate(p5, size=c4.shape[2:], mode='bilinear', align_corners=False)))
        p3 = self.fusion[0](
            c3 + self.top_down[0](F.interpolate(p4, size=c3.shape[2:], mode='bilinear', align_corners=False)))

        n3 = p3


        n3_up = self.bottom_up[0](n3)
        if n3_up.shape[2:] != p4.shape[2:]:
            n3_up = F.interpolate(n3_up, size=p4.shape[2:], mode='bilinear', align_corners=False)
        n4 = self.fusion[1](p4 + n3_up)

        n4_up = self.bottom_up[1](n4)
        if n4_up.shape[2:] != p5.shape[2:]:
            n4_up = F.interpolate(n4_up, size=p5.shape[2:], mode='bilinear', align_corners=False)
        n5 = self.fusion[2](p5 + n4_up)

        f3 = p3 + n3
        f4 = p4 + n4
        f5 = p5 + n5

        f3 = self.channel_calibration[0](f3)
        f4 = self.channel_calibration[1](f4)
        f5 = self.channel_calibration[2](f5)

        if f4.shape[2:] != f3.shape[2:]:
            f4 = F.interpolate(f4, size=f3.shape[2:], mode='bilinear', align_corners=False)
        if f5.shape[2:] != f3.shape[2:]:
            f5 = F.interpolate(f5, size=f3.shape[2:], mode='bilinear', align_corners=False)


        spatial_weights = []
        for feat in [f3, f4, f5]:

            avg_feat = torch.mean(feat, dim=1, keepdim=True)
            max_feat, _ = torch.max(feat, dim=1, keepdim=True)
            spatial_weights.append(torch.cat([avg_feat, max_feat], dim=1))


        spatial_weights = torch.cat([sw[:, :1] for sw in spatial_weights], dim=1)
        attention_map = self.spatial_attention(spatial_weights)


        fused = f3 + f4 + f5
        fused = fused * attention_map

        output = self.output_layer(fused)

        return output

