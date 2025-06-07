import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from ...utils.spconv_utils import replace_feature, spconv
from .scatterformer_utils import rearrange, scatter_matmul_kv, scatter_matmul_qc, torch_scatter
from torch.cuda.amp import autocast

class SlotFormer(spconv.SparseModule):
    def __init__(self, embed_dim, nhead, num_layers=2, win_size=12, shift=True):
        super().__init__()
        self.nhead = nhead
        self.d_model = embed_dim
        self.num_layers = num_layers
        self.win_size = win_size
        
        self.sla_layers = list()
        
        shift = [False, shift]
        for i in range(num_layers):
            self.sla_layers.append( SFLayer(embed_dim, nhead=nhead, directions=2, win_size=win_size, shift=shift[i]) )
            
        self.sla_layers = nn.ModuleList(self.sla_layers)
    
    def forward(self, src):
        
        for idx, enc in enumerate(self.sla_layers):
            src = enc(src)

        return src
    
class SFLayer(nn.Module):
    def __init__(self, embed_dim, nhead, directions=2, win_size=12, shift=False):
        super().__init__()
        self.nhead = nhead
        self.d_model = embed_dim
        self.directions = directions
        self.win_size = win_size
        self.shift = shift
        self.self_norm_layers = list()
        self.self_attn_layers = list()
        self.ffn_layers = list()
        
        for _ in range(directions):
            self.self_norm_layers.append( nn.BatchNorm1d(embed_dim, eps=1e-3, momentum=0.01) )
            self.self_attn_layers.append( SelfAttnLayer(embed_dim, nhead) )
            self.ffn_layers.append( CWI_FFN_Layer_3D(embed_dim, conv_size=win_size+1) )
            
        self.self_norm_layers = nn.ModuleList(self.self_norm_layers)
        self.self_attn_layers = nn.ModuleList(self.self_attn_layers)
        self.ffn_layers = nn.ModuleList(self.ffn_layers)
    
    def get_direction_attrs(self, src, direction=0):
        if self.shift:
            batch_win_coords = torch.cat([ src.indices[:, :1], (src.indices[:, direction+2:direction+3] + self.win_size // 2) // self.win_size], dim=1) 
        else:
            batch_win_coords = torch.cat([ src.indices[:, :1], src.indices[:, direction+2:direction+3] // self.win_size ], dim=1) 
        
        _, batch_win_inds = torch.unique(batch_win_coords, return_inverse=True, dim=0)
        
        batch_win_inds, perm = torch.sort(batch_win_inds)
        counts = torch.bincount(batch_win_inds)
        offsets = F.pad( torch.cumsum(counts, dim=0), (1,0), mode='constant', value=0)[:-1]
        return counts, offsets, batch_win_inds, perm

    def forward(self, src):

        for i in range(self.directions):
            counts, offsets, win_inds, perm = self.get_direction_attrs(src, i)
            norm_features = self.self_norm_layers[i](src.features[perm])
            x = checkpoint(self.self_attn_layers[i], norm_features, offsets, counts, win_inds)
            inv_x = x.clone()
            inv_x[perm] = x
            src = replace_feature(src, src.features + inv_x)

            src = self.ffn_layers[i](src)
           
        return src
    
class SelfAttnLayer(nn.Module):

    def __init__(self, dim, num_heads, qkv_bias=False):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads    
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)


    def forward(self, x, offsets, counts, batch_win_inds, eps=1e-3):
        with autocast(enabled=True, dtype=torch.float32):
            N, C = x.shape
            
            qkv = self.qkv(x).reshape(N, 3, C)

            q, k, v = qkv.unbind(1)

            q, k, v = (rearrange(x, "n (h c) -> n h c", h=self.num_heads).contiguous() for x in [q, k, v])

            q = F.relu(q)
            k = F.relu(k)

            kv = scatter_matmul_kv(k, v, offsets, counts)
            s = torch_scatter.scatter_add(k, batch_win_inds, dim=0)
        
            y = scatter_matmul_qc(q, kv, offsets, counts)
            z = torch.sum(s[batch_win_inds, ...] * q, -1, keepdim=True)
        
            y = y / (z + eps)

            y = rearrange(y, "n h c -> n (h c)", h=self.num_heads)
            
            y = self.proj(y)
    
        return y
    

class CWI_FFN_Layer_3D(spconv.SparseModule):
    
    def __init__(self, embed_dim, indice_key='former', conv_size=13):
        super(CWI_FFN_Layer_3D, self).__init__()

        self.bn = nn.BatchNorm1d(embed_dim, eps=1e-3, momentum=0.01)

        self.conv_k = spconv.SubMConv3d(
            embed_dim // 4, embed_dim // 4,kernel_size=3, stride=1, padding=1, indice_key=indice_key + 'k'
        )
        self.conv_h = spconv.SubMConv3d(
            embed_dim // 4, embed_dim // 4,kernel_size=(1, 1, conv_size), stride=(1, 1, 1), padding=(0, 0, conv_size//2), indice_key=indice_key + 'h'
        )
        self.conv_w = spconv.SubMConv3d(
            embed_dim // 4, embed_dim // 4,kernel_size=(1, conv_size, 1), stride=(1, 1, 1), padding=(0, conv_size//2, 0), indice_key=indice_key + 'w'
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
    
class FFN_Layer(nn.Module):
    
    def __init__(self, embed_dim):
        super(FFN_Layer, self).__init__()
        
        self.bn = nn.BatchNorm1d(embed_dim, eps=1e-3, momentum=0.01)
        self.fc1 = nn.Linear(embed_dim, embed_dim*2)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(embed_dim*2, embed_dim)
    
    def forward(self, src):

        ff_features = self.fc2(self.act(self.fc1(self.bn(src.features))))
        src = replace_feature(src, src.features + ff_features)
       
        return src