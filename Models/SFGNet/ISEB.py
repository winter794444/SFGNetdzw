import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.SFGNet.some_functions import LayerNorm2d, ConvMlp


class CBR(nn.Module):
    """Conv -> BN -> ReLU"""
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)


class SimpleMHSA(nn.Module):
    """简化版多头自注意力 (输入输出: [B, C, H, W])"""
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.scale = dim ** -0.5
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        q, k, v = qkv[:,0], qkv[:,1], qkv[:,2]  # [B, heads, c//heads, HW]

        q = q.permute(0,1,3,2)  # [B, heads, HW, c//heads]
        k = k.permute(0,1,2,3)  # [B, heads, c//heads, HW]

        attn = (q @ k) * self.scale # [B, heads, HW, HW]
        attn = attn.softmax(dim=-1)

        v = v.permute(0,1,3,2)  # [B, heads, HW, c//heads]
        out = attn @ v   # [B, heads, HW, c//heads]
        out = out.permute(0,1,3,2).reshape(B, C, H, W)

        return self.proj(out)


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

        # ====== 新增：三个 MHSA 和 CBR ======
        self.mhsa1 = SimpleMHSA(dim, num_heads)
        self.mhsa2 = SimpleMHSA(dim, num_heads)
        self.mhsa3 = SimpleMHSA(dim, num_heads)

        self.cbr1 = CBR(dim)
        self.cbr2 = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        self.cbr3 = CBR(dim)

        self.final_conv = nn.Conv2d(dim * 3, dim, 1) # 拼接后收敛

        # 前馈网络
        self.ffn = ConvMlp(dim, dim)

    def forward(self, main_feat, aux_feats):
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

        # ================== 新结构：三个 MHSA + CBR 级联 ==================
        feat1 = self.mhsa1(fused_feat)
        out1 = self.cbr1(feat1)

        feat2 = self.mhsa2(fused_feat)
        out2 = self.cbr2(torch.cat([out1, feat2], dim=1))

        feat3 = self.mhsa3(fused_feat)
        out3 = self.cbr3(feat3)

        merged = torch.cat([out1, out2, out3], dim=1)
        new_feat = self.final_conv(merged)  # 收敛 back to [B, C, H, W]

        # 输出投影和残差连接
        output = main_feat + self.out_proj(new_feat)

        # 前馈网络和残差连接
        output = output + self.ffn(output)

        return output