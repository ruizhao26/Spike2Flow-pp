import torch
import torch.nn.functional as F
from .utils import bilinear_sampler, coords_grid


class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)
        
        self.corr_shape = corr.shape
        batch, h1, w1, dim, h2, w2 = self.corr_shape
        corr = corr.reshape(batch*h1*w1, dim, h2, w2)
        
        self.corr_pyramid.append(corr)
        for i in range(self.num_levels-1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def get_global_corr(self):
        batch, h1, w1, dim, h2, w2 = self.corr_shape
        return self.corr_pyramid[0].reshape(batch, h1, w1, dim, h2, w2).permute(0, 3, 1, 2, 4, 5)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2*r+1, device=coords.device)
            dy = torch.linspace(-r, r, 2*r+1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)

            centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht*wd)
        fmap2 = fmap2.view(batch, dim, ht*wd) 
        
        corr = torch.matmul(fmap1.transpose(1,2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        
        corr = corr.reshape(batch, ht, wd, ht*wd)
        corr = F.layer_norm(corr, (corr.shape[-1], ), eps=1e-12)
        corr = corr.reshape(batch, ht, wd, 1, ht, wd)
        return corr


class CorrBlock_Dual:
    def __init__(self, fmap1, fmap2, fmap1_rcp, fmap2_rcp, mask, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock_Dual.corr(fmap1, fmap2, fmap1_rcp, fmap2_rcp, mask)

        self.corr_shape = corr.shape
        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch*h1*w1, dim, h2, w2)
        
        self.corr_pyramid.append(corr)
        for i in range(self.num_levels-1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def get_global_corr(self):
        batch, h1, w1, dim, h2, w2 = self.corr_shape
        return self.corr_pyramid[0].reshape(batch, h1, w1, dim, h2, w2).permute(0, 3, 1, 2, 4, 5)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2*r+1, device=coords.device)
            dy = torch.linspace(-r, r, 2*r+1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)

            centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2, fmap1_rcp, fmap2_rcp, mask):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht*wd)
        fmap2 = fmap2.view(batch, dim, ht*wd) 
        fmap1_rcp = fmap1_rcp.view(batch, dim, ht*wd)
        fmap2_rcp = fmap2_rcp.view(batch, dim, ht*wd) 
        
        corr1 = torch.matmul(fmap1.transpose(1,2), fmap2)
        corr1 = corr1.view(batch, ht, wd, 1, ht, wd)
        corr1 = corr1.reshape(batch, ht, wd, ht*wd)
        corr1 = F.layer_norm(corr1, (corr1.shape[-1], ), eps=1e-12)
        corr1 = corr1.reshape(batch, ht, wd, 1, ht, wd)
        # corr1 = corr1 / torch.sqrt(torch.tensor(dim).float())
        
        corr2 = torch.matmul(fmap1_rcp.transpose(1,2), fmap2_rcp)
        corr2 = corr2.view(batch, ht, wd, 1, ht, wd)
        corr2 = corr2.reshape(batch, ht, wd, ht*wd)
        corr2 = F.layer_norm(corr2, (corr2.shape[-1], ), eps=1e-12)
        corr2 = corr2.reshape(batch, ht, wd, 1, ht, wd)
        # corr2 = corr2 / torch.sqrt(torch.tensor(dim).float())

        mask = mask.permute(0, 2, 3, 1).unsqueeze(dim=4).unsqueeze(dim=5)

        corr = corr1 * mask + corr2 * (1 - mask)
        return corr


