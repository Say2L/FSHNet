import torch
import torch.nn as nn
import numpy as np
from functools import partial
from .spconv_backbone import post_act_block, SparseBasicBlock
import torch.nn.functional as F

from pcdet.models.model_utils.hednet_utils import post_act_block_sparse_3d
from pcdet.models.model_utils.hednet_utils import post_act_block_sparse_2d
from pcdet.models.model_utils.hednet_utils import post_act_block_dense_2d
from pcdet.models.model_utils.hednet_utils import SparseBasicBlock3D
from pcdet.models.model_utils.hednet_utils import SparseBasicBlock2D
from pcdet.models.model_utils.slotformer_3d_utils import SlotFormer
from ...utils.spconv_utils import replace_feature, spconv
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils.loss_utils import focal_loss_sparse

norm_fn_1d = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)


def scatter_nd(indices: torch.Tensor, updates: torch.Tensor,
               shape: torch.Tensor) -> torch.Tensor:
    """pytorch edition of tensorflow scatter_nd.

    this function don't contain except handle code. so use this carefully when
    indice repeats, don't support repeat add which is supported in tensorflow.
    """
    ret = torch.zeros(*shape, dtype=updates.dtype, device=updates.device)
    ndim = indices.shape[-1] 
    output_shape = list(indices.shape[:-1]) + shape[indices.shape[-1]:]
    flatted_indices = indices.view(-1, ndim) 
    slices = [flatted_indices[:, i] for i in range(ndim)] 
    slices += [Ellipsis]
    ret[slices] = updates.view(*output_shape) 
    return ret


class SEDLayer(spconv.SparseModule):

    def __init__(self, dim: int, down_kernel_size: list, down_stride: list, num_SBB: list, indice_key, xy_only=False, bias=False):
        super().__init__()

        block = SparseBasicBlock2D if xy_only else SparseBasicBlock3D
        post_act_block = post_act_block_sparse_2d if xy_only else post_act_block_sparse_3d

        self.encoder = nn.ModuleList(
            [spconv.SparseSequential(
                *[block(dim, indice_key=f"{indice_key}_0") for _ in range(num_SBB[0])])]
        )

        num_levels = len(down_stride)
        for idx in range(1, num_levels):
            cur_layers = [
                post_act_block(
                    dim, dim, down_kernel_size[idx], down_stride[idx], down_kernel_size[idx] // 2,
                    conv_type='spconv', indice_key=f'spconv_{indice_key}_{idx}'),

                *[block(dim, indice_key=f"{indice_key}_{idx}", bias=bias) for _ in range(num_SBB[idx])]
            ]
            self.encoder.append(spconv.SparseSequential(*cur_layers))

        self.decoder = nn.ModuleList()
        self.decoder_norm = nn.ModuleList()
        for idx in range(num_levels - 1, 0, -1):
            self.decoder.append(
                post_act_block(
                    dim, dim, down_kernel_size[idx],
                    conv_type='inverseconv', indice_key=f'spconv_{indice_key}_{idx}'))
            self.decoder_norm.append(norm_fn_1d(dim))

    def forward(self, x):
        feats = []
        for conv in self.encoder:
            x = conv(x)
            feats.append(x)

        x = feats[-1]
        for deconv, norm, up_x in zip(self.decoder, self.decoder_norm, feats[:-1][::-1]):
            x = deconv(x)
            x = replace_feature(x, norm(x.features + up_x.features))
        return x
    
class AttnPillarPool(nn.Module):
    def __init__(self, dim, pillar_size=6):
        super().__init__()
        self.dim = dim
        self.pillar_size = pillar_size
        self.query_func = spconv.SparseMaxPool3d(
            (pillar_size, 1, 1), stride=(pillar_size, 1, 1), padding=0)
        self.norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, 8, batch_first=True)
        self.pos_embedding = nn.Embedding(pillar_size, dim)
        nn.init.normal_(self.pos_embedding.weight, std=.01)

    def forward(self, x):
       
        src = self.query_func(x)
        _, batch_win_inds = torch.unique(x.indices[:, [0, 2, 3]], return_inverse=True, dim=0)
     
        scatter_indices = torch.stack([batch_win_inds, x.indices[:, 1]], -1)
        num_pillars = int(batch_win_inds.max() + 1)
        key = value = scatter_nd(
            scatter_indices,  x.features, 
            [num_pillars, self.pillar_size, self.dim]
        )
        key_padding_mask = ~scatter_nd(
            scatter_indices, 
            torch.ones_like(x.features[:, 0]),
            [num_pillars, self.pillar_size]
        ).bool()
        key = key + self.pos_embedding.weight.unsqueeze(0).repeat(num_pillars, 1, 1)
        
        output = self.self_attn(src.features.unsqueeze(1), key, value, key_padding_mask)[0].squeeze(1)
        src = replace_feature(src, self.norm(output + src.features))
        
        return src
    

class FSHNet_nusc(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, class_names, voxel_size, point_cloud_range, **kwargs):
        super().__init__()

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        dim = model_cfg.FEATURE_DIM
        win_size = model_cfg.WIN_SIZE

        self.cpe = spconv.SparseSequential(
            post_act_block(input_channels, dim, 3, norm_fn=norm_fn, padding=1, indice_key='subm1', conv_type='subm'),
            SparseBasicBlock(dim, dim, norm_fn=norm_fn, indice_key='stem'),
            SparseBasicBlock(dim, dim, norm_fn=norm_fn, indice_key='stem'),
            SparseBasicBlock(dim, dim, norm_fn=norm_fn, indice_key='stem'),
            post_act_block(dim, dim, (3, 1, 1), norm_fn=norm_fn, stride=(2, 1, 1), padding=0, indice_key='spconv1', conv_type='spconv'),
        )

        #  [472, 472, 11] -> [236, 236, 6]
        self.stage1 = spconv.SparseSequential(
            SlotFormer(dim, 4, num_layers=2, win_size=win_size),
            nn.BatchNorm1d(dim, eps=1e-3, momentum=0.01),
            spconv.SparseConv3d(dim, dim, 3, stride=2, padding=1, bias=False, indice_key='down'),
        )
       
        self.stage2 = spconv.SparseSequential(
            SlotFormer(dim, 4, num_layers=2, win_size=win_size),
            nn.BatchNorm1d(dim, eps=1e-3, momentum=0.01),
        )

        self.app = AttnPillarPool(dim, self.sparse_shape[0]//4)

        self.adaptive_feature_diffusion = model_cfg.get('AFD', False)
        if self.adaptive_feature_diffusion:
            self.class_names = class_names
            self.voxel_size = voxel_size
            self.point_cloud_range = point_cloud_range
            self.fg_thr = model_cfg['FG_THRESHOLD']
            self.featmap_stride = model_cfg['FEATMAP_STRIDE']
            self.group_pooling_kernel_size = model_cfg['GREOUP_POOLING_KERNEL_SIZE']
            self.detach_feature = model_cfg['DETACH_FEATURE']

            self.class_names = class_names
            self.group_class_names = []
            for names in model_cfg['GROUP_CLASS_NAMES']:
                self.group_class_names.append([x for x in names if x in class_names])

            self.cls_conv = spconv.SparseSequential(
                spconv.SubMConv2d(dim, dim, 3, stride=1, padding=1, bias=False, indice_key='conv_cls'),
                norm_fn(dim),
                nn.ReLU(),
                spconv.SubMConv2d(dim, len(self.group_class_names), 1, bias=True, indice_key='cls_out')
            )
            self.forward_ret_dict = {}

        afd_dim = model_cfg.AFD_FEATURE_DIM
        afd_num_layers = model_cfg.AFD_NUM_LAYERS
        afd_num_SBB = model_cfg.AFD_NUM_SBB
        afd_down_kernel_size = model_cfg.AFD_DOWN_KERNEL_SIZE
        afd_down_stride = model_cfg.AFD_DOWN_STRIDE
        assert afd_down_stride[0] == 1
        assert len(afd_num_SBB) == len(afd_down_stride)

        self.afd_layers = nn.ModuleList()
        for idx in range(afd_num_layers):
            layer = SEDLayer(
                afd_dim, afd_down_kernel_size, afd_down_stride, afd_num_SBB,
                indice_key=f'afdlayer{idx}', xy_only=True)
            self.afd_layers.append(layer)

        self.shared_conv = spconv.SparseSequential(
            spconv.SubMConv2d(afd_dim, afd_dim, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(afd_dim),
            nn.ReLU()
        )

        self.num_point_features = dim
        self.num_bev_features = afd_dim
        self.init_weights()

    def init_weights(self):
        for _, m in self.named_modules():
            if isinstance(m, (spconv.SubMConv2d, spconv.SubMConv3d)):
                nn.init.kaiming_normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        if self.adaptive_feature_diffusion:
            self.cls_conv[-1].bias.data.fill_(-2.19)

    def assign_target(self, batch_spatial_indices, batch_gt_boxes):
        all_names = np.array(['bg', *self.class_names])
        inside_box_target = batch_spatial_indices.new_zeros((len(self.group_class_names), batch_spatial_indices.shape[0]))

        for gidx, names in enumerate(self.group_class_names):
            batch_inside_box_mask = []
            for bidx in range(len(batch_gt_boxes)):
                spatial_indices = batch_spatial_indices[batch_spatial_indices[:, 0] == bidx][:, [2, 1]]
                points = spatial_indices.clone() + 0.5
                points[:, 0] = points[:, 0] * self.featmap_stride * self.voxel_size[0] + self.point_cloud_range[0]
                points[:, 1] = points[:, 1] * self.featmap_stride * self.voxel_size[1] + self.point_cloud_range[1]
                points = torch.cat([points, points.new_zeros((points.shape[0], 1))], dim=-1)

                gt_boxes = batch_gt_boxes[bidx].clone()
                gt_boxes = gt_boxes[(gt_boxes[:, 3] > 0) & (gt_boxes[:, 4] > 0)]
                gt_class_names = all_names[gt_boxes[:, -1].cpu().long().numpy()]

                gt_boxes_single_head = []
                for _, name in enumerate(gt_class_names):
                    if name in names:
                        gt_boxes_single_head.append(gt_boxes[_])

                inside_box_mask = points.new_zeros((points.shape[0]))
                if len(gt_boxes_single_head) > 0:
                    boxes = torch.stack(gt_boxes_single_head)[:, :7]
                    boxes[:, 2] = 0
                    inside_box_mask[roiaware_pool3d_utils.points_in_boxes_gpu(points[None], boxes[None])[0] > -1] = 1
                batch_inside_box_mask.append(inside_box_mask)
            inside_box_target[gidx] = torch.cat(batch_inside_box_mask)
        return inside_box_target

    def get_loss(self):
        spatial_indices = self.forward_ret_dict['spatial_indices']
        batch_size = self.forward_ret_dict['batch_size']
        batch_index = spatial_indices[:, 0]

        inside_box_pred = self.forward_ret_dict['inside_box_pred']
        inside_box_target = self.forward_ret_dict['inside_box_target']
        inside_box_pred = torch.cat([inside_box_pred[:, batch_index == bidx] for bidx in range(batch_size)], dim=1)
        inside_box_pred = torch.clamp(inside_box_pred.sigmoid(), min=1e-4, max=1 - 1e-4)

        cls_loss = 0.0
        recall_dict = {}
        for gidx in range(len(self.group_class_names)):
            group_cls_loss = focal_loss_sparse(inside_box_pred[gidx], inside_box_target[gidx].float())
            cls_loss += group_cls_loss

            fg_mask = inside_box_target[gidx] > 0
            pred_mask = inside_box_pred[gidx][fg_mask] > self.fg_thr
            recall_dict[f'afd_recall_{gidx}'] = (pred_mask.sum() / fg_mask.sum().clamp(min=1.0)).item()
            recall_dict[f'afd_cls_loss_{gidx}'] = group_cls_loss.item()

        return cls_loss, recall_dict
    
    def feature_diffusion(self, x, batch_dict):
        if not self.adaptive_feature_diffusion:
            return x

        detached_x = x
        if self.detach_feature:
            detached_x = spconv.SparseConvTensor(
                features=x.features.detach(),
                indices=x.indices,
                spatial_shape=x.spatial_shape,
                batch_size=x.batch_size
            )

        inside_box_pred = self.cls_conv(detached_x).features.permute(1, 0)

        if self.training:
            inside_box_target = self.assign_target(x.indices, batch_dict['gt_boxes'])
            self.forward_ret_dict['batch_size'] = x.batch_size
            self.forward_ret_dict['spatial_indices'] = x.indices
            self.forward_ret_dict['inside_box_pred'] = inside_box_pred
            self.forward_ret_dict['inside_box_target'] = inside_box_target

        group_inside_mask = inside_box_pred.sigmoid() > self.fg_thr
        bg_mask = ~group_inside_mask.max(dim=0, keepdim=True)[0]
        group_inside_mask = torch.cat([group_inside_mask, bg_mask], dim=0)

        one_mask = x.features.new_zeros((x.batch_size, 1, x.spatial_shape[0], x.spatial_shape[1]))
        for gidx, inside_mask in enumerate(group_inside_mask):
            selected_indices = x.indices[inside_mask]
            single_one_mask = spconv.SparseConvTensor(
                features=x.features.new_ones(selected_indices.shape[0], 1),
                indices=selected_indices,
                spatial_shape=x.spatial_shape,
                batch_size=x.batch_size
            ).dense()
            pooling_size = self.group_pooling_kernel_size[gidx]
            single_one_mask = F.max_pool2d(single_one_mask, kernel_size=pooling_size, stride=1, padding=pooling_size // 2)
            one_mask = torch.maximum(one_mask, single_one_mask)

        zero_indices = (one_mask[:, 0] > 0).nonzero().int()
        zero_features = x.features.new_zeros((len(zero_indices), x.features.shape[1]))

        cat_indices = torch.cat([x.indices, zero_indices], dim=0)
        cat_features = torch.cat([x.features, zero_features], dim=0)
        indices_unique, _inv = torch.unique(cat_indices, dim=0, return_inverse=True)
        features_unique = x.features.new_zeros((indices_unique.shape[0], x.features.shape[1]))
        features_unique.index_add_(0, _inv, cat_features)

        x = spconv.SparseConvTensor(
            features=features_unique,
            indices=indices_unique,
            spatial_shape=x.spatial_shape,
            batch_size=x.batch_size
        )
        return x
    
    def to_bev(self, x):
        features = x.features
        indices = x.indices[:, [0, 2, 3]]
        spatial_shape = x.spatial_shape[1:]

        x = spconv.SparseConvTensor(
            features=features,
            indices=indices,
            spatial_shape=spatial_shape,
            batch_size=x.batch_size
        )
        return x
    
    def forward(self, batch_dict):
        voxel_features = batch_dict['voxel_features']
        voxel_coords = batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']

        x = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        
        x = self.cpe(x)
        x = self.stage1(x)
        x = self.stage2(x)
      
        x = self.app(x)
        x = self.to_bev(x)
        
        x = self.feature_diffusion(x, batch_dict)
        for layer in self.afd_layers:
            x = layer(x)

        x = self.shared_conv(x)

        batch_dict.update({
            'spatial_features_2d': x,
        })
        return batch_dict   