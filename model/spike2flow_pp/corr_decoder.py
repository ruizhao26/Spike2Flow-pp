import loguru
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

from einops.layers.torch import Rearrange
from einops import rearrange

from .utils import coords_grid, bilinear_sampler, upflow8
from .attention import MultiHeadAttention, LinearPositionEmbeddingSine, LinearPositionEmbeddingSine3D, ExpPositionEmbeddingSine
from typing import Optional, Tuple

from timm.models.layers import DropPath, to_2tuple, trunc_normal_


def initialize_flow(img):
    """ Flow is represented as difference between two means flow = mean1 - mean0"""
    N, C, H, W = img.shape
    mean = coords_grid(N, H, W).to(img.device)
    mean_init = coords_grid(N, H, W).to(img.device)

    # optical flow computed as difference: flow = mean1 - mean0
    return mean, mean_init

class CrossAttentionLayer(nn.Module):
    # def __init__(self, dim, cfg, num_heads=8, attn_drop=0., proj_drop=0., drop_path=0., dropout=0.):
    def __init__(self, qk_dim, v_dim, query_token_dim, tgt_token_dim, add_flow_token=True, num_heads=8, attn_drop=0., proj_drop=0., drop_path=0., dropout=0., pe='linear'):
        super(CrossAttentionLayer, self).__init__()

        head_dim = qk_dim // num_heads
        self.scale = head_dim ** -0.5
        self.query_token_dim = query_token_dim
        self.pe = pe

        self.norm1 = nn.LayerNorm(query_token_dim)
        # self.norm2 = nn.LayerNorm(query_token_dim)
        self.multi_head_attn = MultiHeadAttention(qk_dim, num_heads)
        self.q, self.k, self.v = nn.Linear(query_token_dim, qk_dim, bias=True), nn.Linear(tgt_token_dim, qk_dim, bias=True), nn.Linear(tgt_token_dim, v_dim, bias=True)

        self.proj = nn.Linear(v_dim*2, query_token_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # self.ffn = nn.Sequential(
        #     nn.Linear(query_token_dim, query_token_dim),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(query_token_dim, query_token_dim),
        #     nn.Dropout(dropout)
        # )
        self.add_flow_token = add_flow_token
        self.dim = qk_dim

    def add_posi_emb_for_query(self, query, query_coord):
        B, _, H1, W1 = query_coord.shape
        _, _, C = query.shape
        # [B, 2, H1, W1] -> [BH1W1, 1, 2]
        query_coord = query_coord.contiguous()
        query_coord = query_coord.view(B, 2, -1).permute(0, 2, 1)[:,:,None,:].contiguous().view(B*H1*W1, 1, 2)
        if self.pe == 'linear':
            query_coord_enc = LinearPositionEmbeddingSine(query_coord, dim=self.dim)
        elif self.pe == 'exp':
            query_coord_enc = ExpPositionEmbeddingSine(query_coord, dim=self.dim)
        
        query = query + query_coord_enc
        query = query.reshape(B, H1, W1, 1, C)[:, :, :, 0, :].permute(0, 3, 1, 2)
        return query        # B C H W


    def forward(self, query, key, value, memory, query_coord, t, patch_size, size_h3w3):
        """
            query_coord [B, 2, H1, W1]
        """
        B, _, H1, W1 = query_coord.shape

        if key is None and value is None:
            key = self.k(memory)
            value = self.v(memory)

        # [B, 2, H1, W1] -> [BH1W1, 1, 2]
        query_coord = query_coord.contiguous()
        query_coord = query_coord.view(B, 2, -1).permute(0, 2, 1)[:,:,None,:].contiguous().view(B*H1*W1, 1, 2)
        query_coord_t = torch.ones([B*H1*W1, 1, 1], device=query_coord.device, dtype=query_coord.dtype) * t
        query_coord = torch.cat([query_coord, query_coord_t], dim=2)
        if self.pe == 'linear':
            query_coord_enc = LinearPositionEmbeddingSine3D(query_coord, dim=self.dim)
        elif self.pe == 'exp':
            query_coord_enc = ExpPositionEmbeddingSine(query_coord, dim=self.dim)

        short_cut = query
        query = self.norm1(query)

        if self.add_flow_token:
            q = self.q(query+query_coord_enc)
        else:
            q = self.q(query_coord_enc)
        k, v = key, value

        x = self.multi_head_attn(q, k, v)

        x = self.proj(torch.cat([x, short_cut],dim=2))
        x = short_cut + self.proj_drop(x)

        # x = x + self.drop_path(self.ffn(self.norm2(x)))

        return x, k, v

class MemoryDecoderLayer(nn.Module):
    def __init__(self, patch_size=8, query_latent_dim=64, cost_latent_dim=128, output_ch=256, add_flow_token=True, dropout=0.0):
        super(MemoryDecoderLayer, self).__init__()
        self.patch_size = patch_size # for converting coords into H2', W2' space
        self.query_latent_dim = query_latent_dim

        query_token_dim, tgt_token_dim = query_latent_dim, cost_latent_dim
        qk_dim, v_dim = query_token_dim, query_token_dim
        self.cross_attend = CrossAttentionLayer(qk_dim, v_dim, query_token_dim, tgt_token_dim, add_flow_token=add_flow_token, dropout=dropout)

        self.norm = nn.LayerNorm(query_latent_dim)

        self.ffn = nn.Sequential(
            nn.Conv2d(self.query_latent_dim*3, self.query_latent_dim*3, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(self.query_latent_dim*3, self.query_latent_dim*3, 1, 1, 0),
        )

        self.proj = nn.Conv2d(self.query_latent_dim*3, output_ch, 1, 1, 0)

    def forward(self, query_list, key, value, memory, coords_list, size, size_h3w3):
        """
            x:      [B*H1*W1, 1, C]
            memory: [B*H1*W1, H2'*W2', C]
            coords1 [B, 2, H2, W2]
            size: B, C, H1, W1
            1. Note that here coords0 and coords1 are in H2, W2 space.
               Should first convert it into H2', W2' space.
            2. We assume the upper-left point to be [0, 0], instead of letting center of upper-left patch to be [0, 0]
        """
        x_global1, key, value = self.cross_attend(query_list[0], key, value, memory, coords_list[0], 20, self.patch_size, size_h3w3)
        x_global2, key, value = self.cross_attend(query_list[1], key, value, memory, coords_list[1], 30, self.patch_size, size_h3w3)
        x_global3, key, value = self.cross_attend(query_list[2], key, value, memory, coords_list[2], 40, self.patch_size, size_h3w3)
        
        x_global1 = self.norm(x_global1)
        x_global2 = self.norm(x_global2)
        x_global3 = self.norm(x_global3)

        B, C, H1, W1 = size
        C = self.query_latent_dim
        x_global1 = x_global1.view(B, H1, W1, C).permute(0, 3, 1, 2)
        x_global2 = x_global2.view(B, H1, W1, C).permute(0, 3, 1, 2)
        x_global3 = x_global3.view(B, H1, W1, C).permute(0, 3, 1, 2)
        
        x_global_all = torch.cat([x_global1, x_global2, x_global3], dim=1)
        x_global_all = x_global_all + self.ffn(x_global_all)
        x_global = self.proj(x_global_all)

        return x_global, key, value


if __name__ == '__main__':
    query_coord = torch.rand(48*48, 1, 2)
    query_coord_enc = LinearPositionEmbeddingSine(query_coord, dim=64)
    print(query_coord_enc.shape)