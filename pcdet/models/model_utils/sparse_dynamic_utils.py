import torch
from torch import nn
import torch.nn.functional as F
from .slotformer_utils import CrossSlotLayer, SlotLayer

def clip_sigmoid(x, eps=1e-4):
    y = torch.clamp(x.sigmoid_(), min=eps, max=1 - eps)
    return y


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, input_channel, num_pos_feats=288):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    def forward(self, xyz):
        xyz = xyz.transpose(1, 2).contiguous()
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding


class SparseTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, slot_width=12, self_only=False):
        super().__init__()
        self.self_only = self_only
        if self.self_only:
            self.attn_layer = SlotLayer(d_model, nhead, 2, slot_width, cwi=True)
        else:
            self.attn_layer = CrossSlotLayer(d_model, nhead, 2, slot_width=12)

    def forward(self, query_tensor, bev_tensor):

        if self.self_only:
            query_tensor = self.attn_layer(query_tensor)
        else:
            query_tensor = self.attn_layer(query_tensor, bev_tensor)

        return query_tensor