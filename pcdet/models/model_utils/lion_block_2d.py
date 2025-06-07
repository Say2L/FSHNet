from functools import partial

import math
import numpy as np
import torch
import torch.nn as nn
import torch_scatter
import einops
from torch.nn import functional as F
from .scatterformer_utils import rearrange, scatter_matmul_kv, scatter_matmul_qc, torch_scatter
from ...utils.spconv_utils import replace_feature, spconv
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import autocast

@torch.inference_mode()
def get_window_coors_shift_v2(coords, sparse_shape, window_shape, shift=False):
    sparse_shape_y, sparse_shape_x = sparse_shape
    win_shape_x = win_shape_y = window_shape

    if shift:
        shift_x, shift_y = win_shape_x // 2, win_shape_y // 2
    else:
        shift_x, shift_y = 0, 0 # win_shape_x, win_shape_y, win_shape_z

    max_num_win_x = int(np.ceil((sparse_shape_x / win_shape_x)) + 1)  # plus one here to meet the needs of shift.
    max_num_win_y = int(np.ceil((sparse_shape_y / win_shape_y)) + 1)  # plus one here to meet the needs of shift.
    
    max_num_win_per_sample = max_num_win_x * max_num_win_y

    x = coords[:, 2] + shift_x
    y = coords[:, 1] + shift_y

    win_coors_x = x // win_shape_x
    win_coors_y = y // win_shape_y
    
    coors_in_win_x = x % win_shape_x
    coors_in_win_y = y % win_shape_y
    
    batch_win_inds_x = coords[:, 0] * max_num_win_per_sample + win_coors_x * max_num_win_y + \
                       win_coors_y
    batch_win_inds_y = coords[:, 0] * max_num_win_per_sample + win_coors_y * max_num_win_x + \
                       win_coors_x

    coors_in_win = torch.stack([coors_in_win_y, coors_in_win_x], dim=-1)

    return batch_win_inds_x, batch_win_inds_y, coors_in_win

class FlattenedWindowMapping(nn.Module):
    def __init__(
            self,
            window_shape,
            group_size,
            shift,
            win_version='v2'
    ) -> None:
        super().__init__()
        self.window_shape = window_shape
        self.group_size = group_size
        self.win_version = win_version
        self.shift = shift

    def forward(self, coords: torch.Tensor, batch_size: int, sparse_shape: list):
        coords = coords.long()
        _, num_per_batch = torch.unique(coords[:, 0], sorted=False, return_counts=True)
        batch_start_indices = F.pad(torch.cumsum(num_per_batch, dim=0), (1, 0))
        num_per_batch_p = (
                torch.div(
                    batch_start_indices[1:] - batch_start_indices[:-1] + self.group_size - 1,
                    self.group_size,
                    rounding_mode="trunc",
                )
                * self.group_size
        )

        batch_start_indices_p = F.pad(torch.cumsum(num_per_batch_p, dim=0), (1, 0))
        flat2win = torch.arange(batch_start_indices_p[-1], device=coords.device)  # .to(coords.device)
        win2flat = torch.arange(batch_start_indices[-1], device=coords.device)  # .to(coords.device)

        for i in range(batch_size):
            if num_per_batch[i] != num_per_batch_p[i]:
                
                bias_index = batch_start_indices_p[i] - batch_start_indices[i]
                flat2win[
                    batch_start_indices_p[i + 1] - self.group_size + (num_per_batch[i] % self.group_size):
                    batch_start_indices_p[i + 1]
                    ] = flat2win[
                        batch_start_indices_p[i + 1]
                        - 2 * self.group_size
                        + (num_per_batch[i] % self.group_size): batch_start_indices_p[i + 1] - self.group_size
                        ] if (batch_start_indices_p[i + 1] - batch_start_indices_p[i]) - self.group_size != 0 else \
                        win2flat[batch_start_indices[i]: batch_start_indices[i + 1]].repeat(
                            (batch_start_indices_p[i + 1] - batch_start_indices_p[i]) // num_per_batch[i] + 1)[
                        : self.group_size - (num_per_batch[i] % self.group_size)] + bias_index


            win2flat[batch_start_indices[i]: batch_start_indices[i + 1]] += (
                    batch_start_indices_p[i] - batch_start_indices[i]
            )

            flat2win[batch_start_indices_p[i]: batch_start_indices_p[i + 1]] -= (
                    batch_start_indices_p[i] - batch_start_indices[i]
            )

        mappings = {"flat2win": flat2win, "win2flat": win2flat}

        batch_win_inds_x, batch_win_inds_y, coors_in_win = get_window_coors_shift_v2(coords, sparse_shape,
                                                                                        self.window_shape, self.shift)
        vx = batch_win_inds_x * self.window_shape * self.window_shape
        vx += coors_in_win[..., 1] * self.window_shape + coors_in_win[..., 0]

        vy = batch_win_inds_y * self.window_shape * self.window_shape
        vy += coors_in_win[..., 0] * self.window_shape + coors_in_win[..., 1]

        _, mappings["x"] = torch.sort(vx)
        _, mappings["y"] = torch.sort(vy)

        return mappings

class LIONBlock2D(spconv.SparseModule):
    def __init__(self, dim: int, num_layers: int, nhead, win_size, group_size, directions, shift=False):
        super().__init__()

        self.encoder = nn.ModuleList()

        shift = [False, shift]
        for idx in range(num_layers):
            self.encoder.append(LIONLayer(dim, nhead, win_size, group_size, directions, shift[idx]))

    def forward(self, x):
        
        for idx, enc in enumerate(self.encoder):
    
            x = enc(x)
          
        return x
    
class LIONLayer(nn.Module):
    def __init__(self, embed_dim, nhead, win_size, group_size, directions=2, shift=False):
        super(LIONLayer, self).__init__()

        self.win_size = win_size
        self.group_size = group_size
        self.embed_dim = embed_dim
        self.directions = directions

        self.self_norm_layers = list()
        self.self_attn_layers = list()
        self.ffn_layers = list()
        
        for dir in range(len(directions)):
            self.self_norm_layers.append( nn.BatchNorm1d(embed_dim, eps=1e-3, momentum=0.01) )
            self.self_attn_layers.append( MultiHeadAttention(embed_dim, nhead, group_size) )
            self.ffn_layers.append( CWI_FFN_Layer(embed_dim, conv_size=self.win_size+1) )
            
        self.self_norm_layers = nn.ModuleList(self.self_norm_layers)
        self.self_attn_layers = nn.ModuleList(self.self_attn_layers)
        self.ffn_layers = nn.ModuleList(self.ffn_layers)

        self.window_partition = FlattenedWindowMapping(self.win_size, self.group_size, shift)

    def forward(self, x):
        mappings = self.window_partition(x.indices, x.batch_size, x.spatial_shape)
        
        for i in range(len(self.directions)):
            indices = mappings[self.directions[i]]
            x_features = x.features[indices][mappings["flat2win"]]

            x_features = self.self_norm_layers[i](x_features)

            # x_features = rearrange(x_features, "(n g) c -> n g c", g=self.group_size).contiguous()
            x_features = checkpoint(self.self_attn_layers[i], x_features.float())

            new_x_features = x.features.clone().float()
            new_x_features[indices] = x_features.reshape(-1, x_features.shape[-1])[mappings["win2flat"]]

            x = replace_feature(x, x.features + new_x_features)

            x = self.ffn_layers[i](x)

            return x
    
class SelfAttnLayer(nn.Module):

    def __init__(self, dim, num_heads, group_size, qkv_bias=False):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.group_size = group_size  
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)


    def forward(self, x, eps=1e-3):
        with autocast(enabled=True, dtype=torch.float32):
            N, C = x.shape
            
            qkv = self.qkv(x).reshape(N, 3, C)

            q, k, v = qkv.unbind(1)

            q, k, v = (rearrange(x, "(n g) (h c) -> (n h) g c", h=self.num_heads, g=self.group_size).contiguous() for x in [q, k, v])

            q = F.relu(q)
            k = F.relu(k)

            kv = torch.einsum("b j c, b j d -> b c d", k, v)
            
            s = torch.sum(k, dim=1, keepdim=True)

            y = torch.einsum("b j c, b c d -> b j d", q, kv)
            
            z = torch.sum(s * q, -1, keepdim=True)
        
            y = y / (z + eps)

            y = rearrange(y, "(n h) g c-> (n g) (h c)", h=self.num_heads, g=self.group_size)
            
            y = self.proj(y)
    
        return y


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads, group_size):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.group_size = group_size
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.qkv = nn.Linear(self.embed_size, self.embed_size*3, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        with autocast(enabled=True, dtype=torch.float32):
            N, C = x.shape
            qkv = self.qkv(x).reshape(N, 3, C)

            q, k, v = qkv.unbind(1)

            q, k, v = (rearrange(x, "(n g) (h c) -> (n h) g c", h=self.heads, g=self.group_size).contiguous() for x in [q, k, v])

            # Einsum does matrix multiplication for query*keys for each training example
            # with every other training example, don't be confused by einsum
            # it's just a way to do batch matrix multiplication
            energy = torch.einsum("nqd,nkd->nqk", [q, k])
           
            attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=-1)

            out = torch.einsum("nql,nld->nqd", [attention, v])

            out = rearrange(out, "(n h) g c-> (n g) (h c)", h=self.heads, g=self.group_size)
            
            out = self.fc_out(out)
        return out

class CWI_FFN_Layer(spconv.SparseModule):
    
    def __init__(self, embed_dim, indice_key='former', conv_size=13):
        super(CWI_FFN_Layer, self).__init__()

        self.bn = nn.BatchNorm1d(embed_dim, eps=1e-3, momentum=0.01)

        self.conv_k = spconv.SubMConv2d(
            embed_dim // 4, embed_dim // 4,kernel_size=3, stride=1, padding=1, indice_key=indice_key + 'k'
        )
        self.conv_h = spconv.SubMConv2d(
            embed_dim // 4, embed_dim // 4,kernel_size=(1, conv_size), stride=(1, 1), padding=(0, conv_size//2), indice_key=indice_key + 'h'
        )
        self.conv_w = spconv.SubMConv2d(
            embed_dim // 4, embed_dim // 4,kernel_size=(conv_size, 1), stride=(1, 1), padding=(conv_size//2, 0), indice_key=indice_key + 'w'
        )

        self.bn2 = nn.BatchNorm1d(embed_dim, eps=1e-3, momentum=0.01)
        self.fc1 = nn.Linear(embed_dim, embed_dim*2)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(embed_dim*2, embed_dim)
        
        self.group_dim = embed_dim // 4

    def forward(self, src):

        src = replace_feature(src, self.bn(src.features))
        src_k = replace_feature(src, src.features[:, :self.group_dim])
        src_h = replace_feature(src, src.features[:, self.group_dim:2*self.group_dim])
        src_w = replace_feature(src, src.features[:, 2*self.group_dim:3*self.group_dim])
        
        src_k = self.conv_k(src_k).features
        src_h = self.conv_h(src_h).features
        src_w = self.conv_w(src_w).features
        src_i = src.features[:, 3*self.group_dim:]
        src2 = replace_feature(src, torch.cat([src_k, src_h, src_w, src_i], 1))
        src = replace_feature(src, src.features + src2.features)

        src2 = replace_feature(src2, self.fc2(self.act(self.fc1(self.bn2(src.features)))))
        src = replace_feature(src, src.features + src2.features)
        
        return src