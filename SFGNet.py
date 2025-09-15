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
from Models.SFGNet.ISEB import InteractiveStructureEnhancementBlock
from Models.SFGNet.SomeNets import MultilevelDecoder
from Models.SFGNet.more_functions import BidirectionalCrossAttentionBlock, \
    BidirectionalIneractiveNetwork
from Models.SFGNet.CLIP336_A import CLIP336
from Models.SFGNet.some_functions import ConvMlp, ProjectionNetwork, LNConvAct, MultiscaleFeatureAlignment, TransformerDecoder
from Models.SFGNet.PVTIMP import pvt_v2_b5
def cosine_similarity_loss(ft, fv):
    ft = F.normalize(ft, p=2, dim=1)
    fv = F.normalize(fv, p=2, dim=1)
    cosine_sim = torch.sum(ft * fv, dim=1)
    return -cosine_sim.mean()

class FourierFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels, bands=3):
        super(FourierFeatureExtractor, self).__init__()
        self.bands = bands

        self.band_filters = nn.Parameter(torch.ones(bands, in_channels))
        nn.init.kaiming_normal_(self.band_filters)

        self.conv = nn.Conv2d(in_channels * bands, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        batch, c, h, w = x.shape

        x_freq = torch.fft.rfft2(x, dim=(-2, -1))

        magnitude = torch.abs(x_freq)
        phase = torch.angle(x_freq)

        features = []
        h_freq = magnitude.shape[-2]

        band_size = h_freq // self.bands
        for i in range(self.bands):
            start_idx = i * band_size
            end_idx = (i + 1) * band_size if i < self.bands - 1 else h_freq

            band_magnitude = magnitude.clone()

            band_magnitude[:, :, :start_idx, :] = 0
            if end_idx < h_freq:
                band_magnitude[:, :, end_idx:, :] = 0

            filter_weights = self.band_filters[i].view(1, c, 1, 1)
            band_magnitude = band_magnitude * filter_weights

            band_freq = torch.polar(band_magnitude, phase)

            band_spatial = torch.fft.irfft2(band_freq, s=(h, w), dim=(-2, -1))
            features.append(band_spatial)

        x_multi_band = torch.cat(features, dim=1)

        out = self.conv(x_multi_band)
        out = self.bn(out)
        out = self.relu(out)

        return out



class FrequencySpatialFusion(nn.Module):
    def __init__(self, in_channels):
        super(FrequencySpatialFusion, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, spatial_feat, freq_feat):

        channel_weights = self.channel_attention(spatial_feat)
        spatial_feat = spatial_feat * channel_weights


        avg_pool = torch.mean(freq_feat, dim=1, keepdim=True)
        max_pool, _ = torch.max(freq_feat, dim=1, keepdim=True)
        spatial_weights = self.spatial_attention(torch.cat([avg_pool, max_pool], dim=1))
        freq_feat = freq_feat * spatial_weights


        fused_feat = self.fusion_conv(torch.cat([spatial_feat, freq_feat], dim=1))

        return fused_feat


class SFGNet(nn.Module):
    """Semantic and Frequency Guided Network """

    def __init__(self, encoder=None, feature_levels=[64, 128, 320, 512], backbone=MultilevelDecoder):
        super().__init__()
        self.clip = CLIP336(pretrained="path to your/open_clip_pytorch_model.safetensors")
        self.encoder = encoder if encoder is not None else pvt_v2_b5()
        self.feature_levels = feature_levels
        self.hidden_dim = 768

        self.mlp_blocks = nn.ModuleList([ConvMlp(1024, self.hidden_dim) for _ in range(4)])

        self.cross_attention = BidirectionalCrossAttentionBlock(self.hidden_dim, guide_dim=self.hidden_dim)
        self.structure_merge_deep = InteractiveStructureEnhancementBlock(feature_levels[3])

        self.segmentation_head = nn.Conv2d(self.hidden_dim, 1, 1)
        self.refinement_head = nn.Sequential(
            LNConvAct(512, 512, 3, 1, 1, act_name="relu"),
            nn.Conv2d(512, 1, 3, 1, 1)
        )

        self.text_projection = ProjectionNetwork(input_dim=self.hidden_dim, proj_dim=512)
        self.visual_projection_mid = ProjectionNetwork(input_dim=self.hidden_dim, proj_dim=feature_levels[3])
        self.visual_projection_deep = ProjectionNetwork(input_dim=512, proj_dim=feature_levels[3])

        self.body_encoder = MultiscaleFeatureAlignment(self.hidden_dim)
        self.neck = BidirectionalIneractiveNetwork(in_channels=[self.hidden_dim] * 3, out_channels=[256, 512, 1024])
        self.decoder = TransformerDecoder(num_layers=1, d_model=512)

        self.backbone = backbone(channel=64, fl=feature_levels)

        self.freq_extractor_input = FourierFeatureExtractor(3, 64)

        self.freq_extractors = nn.ModuleList([
            FourierFeatureExtractor(feature_levels[i], feature_levels[i])
            for i in range(4)
        ])

        self.freq_spatial_fusions = nn.ModuleList([
            FrequencySpatialFusion(feature_levels[i])
            for i in range(4)
        ])

        self.final_enhancement = nn.Sequential(
            nn.Conv2d(feature_levels[3], feature_levels[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_levels[3]),
            nn.ReLU(inplace=True)
        )

    def get_visual_features(self, image, text_embeddings):
        visual_feats = self.clip.get_visual_feats_bchw(image)
        visual_feats = [mlp(f) for mlp, f in zip(self.mlp_blocks, visual_feats)]
        fused_feats = self.neck(visual_feats[:-1], text_embeddings)
        return *visual_feats, fused_feats

    def pool_features(self, features, pooling='avg'):
        # features: [B, C, H, W]
        if pooling == 'avg':
            return features.mean(dim=(2, 3))  # → [B, C]
        else:
            return features.amax(dim=(2, 3))  # → [B, C]

    def forward_pass(self, image, image_aux, text_embeddings):

        res1, res2, res3, res_deep, fused = self.get_visual_features(image_aux, text_embeddings)


        text_proj = self.text_projection(text_embeddings)


        b, c, h, w = fused.shape
        decoded = self.decoder(fused).view(b, c, h, w)
        refined = self.refinement_head(decoded)


        res1_mod = self.cross_attention(res1 * refined, text_embeddings)


        body_features = self.body_encoder(res1_mod, res3, res2)
        segmentation_map = self.segmentation_head(body_features)

        b_body, c_body, h_body, w_body = body_features.shape
        body_features_flat = body_features.flatten(2).permute(0, 2, 1)
        vmf = self.visual_projection_mid(body_features_flat)
        vm = vmf.permute(0, 2, 1).reshape(b, self.feature_levels[3], h_body, w_body)

        decoded_flat = decoded.flatten(2).permute(0, 2, 1)
        vdf = self.visual_projection_deep(decoded_flat)
        vd = vdf.permute(0, 2, 1).reshape(b, self.feature_levels[3], h, w)


        enc1, enc2, enc3, enc4 = self.encoder(image)


        input_freq_feat = self.freq_extractor_input(image)


        enc1_freq = self.freq_extractors[0](enc1)
        enc2_freq = self.freq_extractors[1](enc2)
        enc3_freq = self.freq_extractors[2](enc3)
        enc4_freq = self.freq_extractors[3](enc4)


        enc1 = self.freq_spatial_fusions[0](enc1, enc1_freq)
        enc2 = self.freq_spatial_fusions[1](enc2, enc2_freq)
        enc3 = self.freq_spatial_fusions[2](enc3, enc3_freq)
        enc4 = self.freq_spatial_fusions[3](enc4, enc4_freq)


        vm = F.interpolate(vm, size=enc4.shape[2:], mode='bilinear', align_corners=False)
        vd = F.interpolate(vd, size=enc4.shape[2:], mode='bilinear', align_corners=False)

        merged_output = self.structure_merge_deep(enc4, [vm, vd])
        merged_output = self.final_enhancement(merged_output)

        target_size = (enc4.shape[2] // 2 * 2, enc4.shape[3] // 2 * 2)

        enc1 = F.interpolate(enc1, size=(target_size[0] * 8, target_size[1] * 8), mode='bilinear', align_corners=False)
        enc2 = F.interpolate(enc2, size=(target_size[0] * 4, target_size[1] * 4), mode='bilinear', align_corners=False)
        enc3 = F.interpolate(enc3, size=(target_size[0] * 2, target_size[1] * 2), mode='bilinear', align_corners=False)
        enc4 = F.interpolate(enc4, size=target_size, mode='bilinear', align_corners=False)
        merged_output = F.interpolate(merged_output, size=target_size, mode='bilinear', align_corners=False)

        final_segmentation = self.backbone(enc1, enc2, enc3, enc4, merged_output)
        pooled_fused = self.pool_features(fused)  # [B, C]
        consistency = cosine_similarity_loss(pooled_fused, text_proj) * 0.2
        return final_segmentation, segmentation_map,consistency

    def forward(self, image, image_aux, class_names):

        if isinstance(class_names, list):
            text_tokens = self.clip.text_tokenizer(class_names).to(image.device)
        else:
            text_tokens = class_names

        text_embeddings = self.clip.get_text_embeddings(text_tokens)
        return self.forward_pass(image, image_aux, text_embeddings)