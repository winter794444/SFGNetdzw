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

import open_clip
import torch
#   Author:   DeZhen Wang, from QUT
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class CLIP336(nn.Module):
    def __init__(self, model_name="ViT-L-14-336", pretrained="openai"):
        super().__init__()
        self.clip_model, _, self.preprocess_val = open_clip.create_model_and_transforms(
            model_name,
            pretrained="path to/open_clip_pytorch_model.safetensors"
        )
        self.text_tokenizer = open_clip.get_tokenizer(model_name)

    @property
    def device(self):
        return next(self.clip_model.parameters()).device
    @property
    def dtype(self):
        return self.clip_model.visual.conv1.weight.dtype
    

    
    @torch.no_grad()
    def get_visual_feats_bchw(self, x):
        vit_model = self.clip_model.visual
        intermediate_features = []
        # 获取patch特征
        x = vit_model.conv1(x)  # 初始卷积，获取patch嵌入
        batch_size, num_channels, h_patches, w_patches = x.shape  # x shape = [B, C, H_patches, W_patches]

        # 将特征展平，准备输入到 Transformer
        x = x.reshape(batch_size, num_channels, -1)  # shape = [B, C, H_patches * W_patches]
        x = x.permute(0, 2, 1)  # shape = [B, H_patches * W_patches, C]

        # 添加分类 token
        x = torch.cat([vit_model.class_embedding.unsqueeze(0).expand(x.shape[0], 1, -1).to(x.dtype), x], dim=1)
        #   Author:   DeZhen Wang, from QUT
        # 添加位置嵌入
        x = x + vit_model.positional_embedding.to(x.dtype)

        # 预处理归一化
        x = vit_model.ln_pre(x)
        x = x.permute(1, 0, 2)
        selected_layers = [8, 16, 24]
        xx = x
        for i, blk in enumerate(vit_model.transformer.resblocks):
            xx = blk(xx)

            x = xx.permute(1, 0, 2)  # [L, B, C] -> [B, L, C]
            patch_features = x[:, 1:, :]  # 忽略CLS token，保留patch特征 [B, H_patches * W_patches, C]

            if (i + 1) in selected_layers:
                patch_features = patch_features.permute(0, 2, 1)  # [B, C, H_patches * W_patches]
                patch_features = patch_features.reshape(batch_size, num_channels, h_patches,
                                                        w_patches)  # [B, C, H_patches, W_patches]
                intermediate_features.append(patch_features)  # 保存每层特征
            #  print(patch_features.size())
        #   Author:   DeZhen Wang, from QUT
        x = xx.permute(1, 0, 2)  # [L, B, C] -> [B, L, C]
        patch_features = x[:, 1:, :]
        patch_features = patch_features.permute(0, 2, 1)  # [B, C, H_patches * W_patches]
        patch_features = patch_features.reshape(batch_size, num_channels, h_patches,
                                                w_patches)  # [B, C, H_patches, W_patches]
        intermediate_features.append(patch_features)  # 保存每层特征
        #   Author:   DeZhen Wang, from QUT
        return intermediate_features  # [B, C, H, W] 格式

    #   Author:   DeZhen Wang, from QUT