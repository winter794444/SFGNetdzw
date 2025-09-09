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


class BidirectionalCrossAttentionBlock(nn.Module):
    """
    自适应跨模态注意力模块，动态融合文本和图像特征
    """

    def __init__(self, feat_dim, guide_dim, num_heads=4):
        super().__init__()
        self.feat_dim = feat_dim
        self.guide_dim = guide_dim
        self.num_heads = num_heads
        self.head_dim = feat_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # 归一化层
        self.img_norm = LayerNorm2d(feat_dim)
        self.txt_norm = nn.LayerNorm(guide_dim)

        self.img_q = nn.Conv2d(feat_dim, feat_dim, kernel_size=1, bias=False)
        self.img_k = nn.Conv2d(feat_dim, feat_dim, kernel_size=1, bias=False)
        self.img_v = nn.Conv2d(feat_dim, feat_dim, kernel_size=1, bias=False)

        self.txt_q = nn.Linear(guide_dim, feat_dim, bias=False)
        self.txt_k = nn.Linear(guide_dim, feat_dim, bias=False)
        self.txt_v = nn.Linear(guide_dim, feat_dim, bias=False)

        # 自适应门控机制
        self.txt_gate = nn.Sequential(
            nn.Linear(guide_dim, 1),
            nn.Sigmoid()
        )

        self.img_gate = nn.Sequential(
            nn.Conv2d(feat_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 输出投影
        self.output_proj = nn.Conv2d(feat_dim, feat_dim, kernel_size=1)

        # 前馈网络
        self.ffn = ConvMlp(feat_dim, feat_dim)

    def forward(self, img_feats, txt_feats):
        """
        img_feats: [B, C, H, W] - 图像特征
        txt_feats: [B, D] - 文本特征
        """
        batch_size, _, height, width = img_feats.shape

        # 应用归一化
        img_feats_norm = self.img_norm(img_feats)
        txt_feats_norm = self.txt_norm(txt_feats)

        # 图像特征投影
        q_img = self.img_q(img_feats_norm)
        k_img = self.img_k(img_feats_norm)
        v_img = self.img_v(img_feats_norm)

        # 文本特征投影
        q_txt = self.txt_q(txt_feats_norm).unsqueeze(1)  # [B, 1, C]
        k_txt = self.txt_k(txt_feats_norm).unsqueeze(1)  # [B, 1, C]
        v_txt = self.txt_v(txt_feats_norm).unsqueeze(1)  # [B, 1, C]

        # 重塑图像特征以进行注意力计算
        q_img_flat = rearrange(q_img, 'b c h w -> b (h w) c')
        k_img_flat = rearrange(k_img, 'b c h w -> b c (h w)')
        v_img_flat = rearrange(v_img, 'b c h w -> b (h w) c')

        # 计算自适应门控权重
        txt_importance = self.txt_gate(txt_feats_norm)  # [B, 1]
        img_importance = self.img_gate(img_feats_norm)  # [B, 1, H, W]
        img_importance_flat = rearrange(img_importance, 'b c h w -> b (h w) c')  # [B, H*W, 1]

        # 1. 文本引导的图像注意力 (Text->Image)
        # 文本查询关注图像
        attn_txt_to_img = torch.bmm(q_txt, k_img_flat) * self.scale  # [B, 1, H*W]
        attn_txt_to_img = F.softmax(attn_txt_to_img, dim=-1)
        txt_guided_feat = torch.bmm(attn_txt_to_img, v_img_flat)  # [B, 1, C]

        # 2. 图像引导的文本注意力 (Image->Text)
        # 图像查询关注文本
        attn_img_to_txt = torch.bmm(q_img_flat, k_txt.transpose(1, 2)) * self.scale  # [B, H*W, 1]
        attn_img_to_txt = F.softmax(attn_img_to_txt, dim=-1)
        img_guided_feat = torch.bmm(attn_img_to_txt, v_txt)  # [B, H*W, C]


        img_guided_feat = rearrange(img_guided_feat, 'b (h w) c -> b c h w', h=height, w=width)


        txt_guided_feat = txt_guided_feat.squeeze(1).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        txt_guided_feat = txt_guided_feat.expand(-1, -1, height, width)  # [B, C, H, W]


        txt_importance = txt_importance.view(batch_size, 1, 1, 1)  # [B, 1, 1, 1]
        txt_importance_map = txt_importance.expand(-1, self.feat_dim, height, width)

        # 确保 img_importance_map 的形状正确
        img_importance_map = img_importance.expand(-1, self.feat_dim, -1, -1)

        # 加权融合两种特征
        fused_feat = (img_guided_feat * img_importance_map +
                      txt_guided_feat * txt_importance_map) / (img_importance_map + txt_importance_map + 1e-8)

        # 输出投影
        enhanced_feat = self.output_proj(fused_feat)

        # 残差连接
        img_feats = img_feats + enhanced_feat

        # 前馈网络
        img_feats = img_feats + self.ffn(img_feats)

        return img_feats


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

