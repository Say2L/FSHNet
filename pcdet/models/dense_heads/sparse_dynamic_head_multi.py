import copy
import numpy as np

import torch
import torch.nn as nn
import spconv.pytorch as spconv
from torch.nn.init import kaiming_normal_
from ..model_utils import centernet_utils
from ..model_utils import model_nms_utils
from ...utils import loss_utils
from ...utils.spconv_utils import spconv
from ...ops.iou3d_nms import iou3d_nms_utils

class SeparateHead(nn.Module):

    def __init__(self, input_channels, sep_head_dict, kernel_size, init_bias=-2.19, use_bias=False):
        super().__init__()

        self.sep_head_dict = sep_head_dict

        for cur_name in self.sep_head_dict:
            output_channels = self.sep_head_dict[cur_name]['out_channels']
            num_conv = self.sep_head_dict[cur_name]['num_conv']

            fc_list = []
            for k in range(num_conv - 1):
                fc_list.append(spconv.SparseSequential(
                    spconv.SubMConv2d(input_channels, input_channels, kernel_size, padding=int(kernel_size//2), bias=use_bias, indice_key=cur_name),
                    nn.BatchNorm1d(input_channels),
                    nn.ReLU()
                ))
            fc_list.append(spconv.SubMConv2d(input_channels, output_channels, 1, bias=True, indice_key=cur_name+'out'))
            fc = nn.Sequential(*fc_list)
            if 'heatmap' in cur_name:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, spconv.SubMConv2d):
                        nn.init.normal_(m.weight, mean=0, std=0.001)

            self.__setattr__(cur_name, fc)

    def forward(self, x):
        ret_dict = {}
        for cur_name in self.sep_head_dict:
            ret_dict[cur_name] = self.__getattr__(cur_name)(x)

        return ret_dict


class SparseDynamicHead_multi(nn.Module):

    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size,
                 predict_boxes_when_training=False):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_class = num_class
        self.grid_size = grid_size
        self.point_cloud_range = torch.Tensor(point_cloud_range).cuda()
        self.voxel_size = torch.Tensor(voxel_size).cuda()
        self.feature_map_stride = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('FEATURE_MAP_STRIDE', None)

        self.bn_momentum = model_cfg.get('BN_MOM', 0.1)
        self.bn_eps = model_cfg.get('BN_EPS', 1e-5)

        self.class_names = class_names
        self.class_names_each_head = []
        self.class_id_mapping_each_head = []

        self.up_strides = self.model_cfg.get('UP_STRIDES', [1, 1, 1])
        # self.cross_area_r = self.model_cfg.get('CROSS_AREA_R', 2)
        self.dynamic_pos_num = self.model_cfg.get('DYNAMIC_POS_NUM', [5])
        self.candidate_num = self.model_cfg.get('CANDIDATE_NUM', [5])

        for cur_class_names in self.model_cfg.CLASS_NAMES_EACH_HEAD:
            self.class_names_each_head.append([x for x in cur_class_names if x in class_names])
            cur_class_id_mapping = torch.from_numpy(np.array(
                [self.class_names.index(x) for x in cur_class_names if x in class_names]
            )).cuda()
            self.class_id_mapping_each_head.append(cur_class_id_mapping)

        total_classes = sum([len(x) for x in self.class_names_each_head])
        assert total_classes == len(self.class_names), f'class_names_each_head={self.class_names_each_head}'

        kernel_size_head = self.model_cfg.get('KERNEL_SIZE_HEAD', 3)

        self.heads_list = nn.ModuleList()
        self.separate_head_cfg = model_cfg.SEPARATE_HEAD_CFG
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            cur_head_dict = copy.deepcopy(self.separate_head_cfg.HEAD_DICT)
            cur_head_dict['heatmap'] = dict(out_channels=len(cur_class_names), num_conv=model_cfg.NUM_HM_CONV)
            self.heads_list.append(
                SeparateHead(
                    input_channels=input_channels,
                    sep_head_dict=cur_head_dict,
                    kernel_size=kernel_size_head,
                    init_bias=-np.log((1 - 0.01) / 0.01),
                    use_bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False),
                )
            )
        
        self.get_dynamic_masks = loss_utils.DynamicPositiveMask_Multi(1, self.model_cfg.get('DCLA_REG_WEIGHT', 3), 
                                                                self.voxel_size[:2] * self.feature_map_stride)
        self.predict_boxes_when_training = predict_boxes_when_training
        self.forward_ret_dict = {}
        self.init_weights()
        self.build_losses()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.momentum = self.bn_momentum
                m.eps = self.bn_eps

    def build_losses(self):
        self.add_module('hm_loss_func', loss_utils.FocalLossSparse())
        self.add_module('reg_loss_func', loss_utils.RWIoULoss(self.voxel_size[:2] * self.feature_map_stride))
        self.add_module('crit_iou', loss_utils.SlotFormerIoULoss())

    def sigmoid(self, x):
        y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
        return y

    def distance(self, voxel_indices, centers):
        """
        voxel_indices: [N, 2]
        centers: [M, 2]
        """
        voxel_indices = voxel_indices[None, :, :]
        centers = centers[:, None, :]
        distances = ((voxel_indices - centers)**2).sum(-1)
        return distances

    def manhattan_distance(self, voxel_indices, centers):
        """
        voxel_indices: [N, 2]
        centers: [M, 2]
        """
        voxel_indices = voxel_indices[None, :, :]
        centers = centers[:, None, :]
        distances = (torch.abs(voxel_indices - centers)).sum(-1)
        return distances

    def assign_target_of_single_head(self, num_classes, gt_boxes, feature_map_size, 
                                     feature_map_stride, spatial_indices, num_max_objs=500,
                                     dynamic_pos_num=5, gaussian_overlap=0.1,
                                     min_radius=2):
        """
        Args:
            gt_boxes: (N, 8)
            feature_map_size: (2), [x, y]

        Returns:

        """
        heatmap = gt_boxes.new_zeros(num_classes, len(spatial_indices))

        ret_boxes = gt_boxes.new_zeros((num_max_objs, dynamic_pos_num, gt_boxes.shape[-1] - 1 + 1))
        inds = gt_boxes.new_zeros(num_max_objs, dynamic_pos_num).long()
        mask = gt_boxes.new_zeros(num_max_objs, dynamic_pos_num).long()
        center_distances = gt_boxes.new_zeros(num_max_objs, dynamic_pos_num)
        candidate_indices = gt_boxes.new_zeros(num_max_objs, dynamic_pos_num, 2)
        box_radius = gt_boxes.new_zeros(num_max_objs)
        #voxel_occ_mask = gt_boxes.new_zeros(num_voxels).long()

        box_masks = ((gt_boxes[:, 3] > 0) & (gt_boxes[:, 4] > 0) & (gt_boxes[:, 5] > 0) & (gt_boxes[:, 0] >= self.point_cloud_range[0]) \
                     & (gt_boxes[:, 1] >= self.point_cloud_range[1]) & (gt_boxes[:, 0] < self.point_cloud_range[3]) \
                     & (gt_boxes[:, 1] < self.point_cloud_range[4]))
        
        box_num = box_masks.sum()
        if box_num > 0:
            gt_boxes = gt_boxes[box_masks]

            x, y, z = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2]
            coord_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / feature_map_stride
            coord_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / feature_map_stride

            coord_x = torch.clamp(coord_x, min=0, max=feature_map_size[0] - 0.5)  # bugfixed: 1e-6 does not work for center.int()
            coord_y = torch.clamp(coord_y, min=0, max=feature_map_size[1] - 0.5)  #

            center = torch.cat((coord_x[:, None], coord_y[:, None]), dim=-1)
            center_int = center.int()
            
            dx, dy, dz = gt_boxes[:, 3], gt_boxes[:, 4], gt_boxes[:, 5]
            dx = dx / self.voxel_size[0] / feature_map_stride
            dy = dy / self.voxel_size[1] / feature_map_stride
            
            # box_radius = torch.sqrt((dx / 2)**2 + (dy / 2)**2)
            radius = centernet_utils.gaussian_radius(dx, dy, min_overlap=gaussian_overlap)
            radius = torch.clamp_min(radius.int(), min=min_radius)

            distances = self.distance(spatial_indices.float() + 0.5, center)
            sort_res = torch.sort(distances, dim=-1)
            sort_distance, sort_inds = sort_res[0][:, :dynamic_pos_num], sort_res[1][:, :dynamic_pos_num]

            inds[:box_num] = sort_inds
            mask[:box_num] = sort_distance <= radius[:, None]
            mask[:box_num][0] = True
            center_distances[:box_num] = sort_distance
            candidate_indices[:box_num] = spatial_indices[sort_inds][..., :2]
            box_radius[:box_num] = radius

            ret_boxes[:box_num, :, 0:2] = center[:, None, :] - spatial_indices[inds[:box_num]][..., :2] - 0.5
            ret_boxes[:box_num, :, 2:3] = z[:, None, None]
            ret_boxes[:box_num, :, 3:6] = gt_boxes[:, None, 3:6].log()
            ret_boxes[:box_num, :, 6:7] = torch.cos(gt_boxes[:, None, 6:7])
            ret_boxes[:box_num, :, 7:] = torch.sin(gt_boxes[:, None, 6:7])
            if gt_boxes.shape[1] > 8:
                    ret_boxes[:box_num, :, 8:] = gt_boxes[:, None, 7:-1]

            cur_class_id = (gt_boxes[:, -1] - 1).long()

            heatmap[cur_class_id[:, None], sort_inds] = 1

        return heatmap, ret_boxes, inds, mask, center_distances, candidate_indices, box_radius
    
    def assign_targets(self, gt_boxes, feature_map_size, spatial_indices, **kwargs):
        """
        Args:
            gt_boxes: (B, M, 8)
        Returns:
        """
        target_assigner_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG

        batch_size = gt_boxes.shape[0]
        ret_dict = {
            'heatmaps': [],
            'target_boxes': [],
            'inds': [],
            'masks': [],
            'center_distances': [],
            'candidate_indices': [],
            'box_radius': [],
            'gt_boxes': []
        }

        all_names = np.array(['bg', *self.class_names])
        for head_idx, cur_class_names in enumerate(self.class_names_each_head):
            heatmap_list, target_boxes_list, inds_list, masks_list, gt_boxes_list, \
            center_distances_list, candidate_indices_list, box_radius_list = [], [], [], [], [], [], [], []
            for bs_idx in range(batch_size):
                cur_gt_boxes = gt_boxes[bs_idx]
                gt_class_names = all_names[cur_gt_boxes[:, -1].cpu().long().numpy()]

                gt_boxes_single_head = []

                for idx, name in enumerate(gt_class_names):
                    if name not in cur_class_names:
                        continue
                    temp_box = cur_gt_boxes[idx]
                    temp_box[-1] = cur_class_names.index(name) + 1
                    gt_boxes_single_head.append(temp_box[None, :])

                if len(gt_boxes_single_head) == 0:
                    gt_boxes_single_head = cur_gt_boxes[:0, :]
                else:
                    gt_boxes_single_head = torch.cat(gt_boxes_single_head, dim=0)

                heatmap, ret_boxes, inds, mask, center_distances, candidate_indices, box_radius = self.assign_target_of_single_head(
                    num_classes=len(cur_class_names), gt_boxes=gt_boxes_single_head,
                    feature_map_size=feature_map_size, feature_map_stride=target_assigner_cfg.FEATURE_MAP_STRIDE,
                    spatial_indices=spatial_indices[spatial_indices[:, 0] == bs_idx][:, [2, 1]],
                    num_max_objs=target_assigner_cfg.NUM_MAX_OBJS,
                    dynamic_pos_num=self.dynamic_pos_num[head_idx],
                    gaussian_overlap=target_assigner_cfg.GAUSSIAN_OVERLAP,
                    min_radius=target_assigner_cfg.get('MIN_RADIUS', 2),
                )
                heatmap_list.append(heatmap.to(gt_boxes_single_head.device))
                target_boxes_list.append(ret_boxes.to(gt_boxes_single_head.device))
                inds_list.append(inds.to(gt_boxes_single_head.device))
                masks_list.append(mask.to(gt_boxes_single_head.device))
                gt_boxes_list.append(gt_boxes_single_head)
                center_distances_list.append(center_distances)
                candidate_indices_list.append(candidate_indices)
                box_radius_list.append(box_radius)

            """v_hm = spconv.SparseConvTensor(
                features=torch.cat(heatmap_list, dim=1).transpose(1, 0).contiguous(),
                indices=spatial_indices.int(),
                spatial_shape=feature_map_size,
                batch_size=batch_size
            )
            v_hm = v_hm.dense().permute(2, 3, 0, 1)
            from matplotlib import pyplot as plt
            data = v_hm.view(*feature_map_size, -1).clone()
            data = (torch.sigmoid(data)).contiguous().detach().cpu().numpy()
            plt.imshow(data, interpolation='nearest')
            plt.show()"""
            
            ret_dict['heatmaps'].append(torch.cat(heatmap_list, dim=1).permute(1, 0))
            ret_dict['target_boxes'].append(torch.stack(target_boxes_list, dim=0))
            ret_dict['inds'].append(torch.stack(inds_list, dim=0))
            ret_dict['masks'].append(torch.stack(masks_list, dim=0))
            ret_dict['gt_boxes'].append(gt_boxes_list)
            ret_dict['center_distances'].append(torch.stack(center_distances_list, dim=0))
            ret_dict['candidate_indices'].append(torch.stack(candidate_indices_list, dim=0))
            ret_dict['box_radius'].append(torch.stack(box_radius_list, dim=0))

        return ret_dict

    def get_loss(self):
        pred_dicts = self.forward_ret_dict['pred_dicts']
        target_dicts = self.forward_ret_dict['target_dicts']
        
        batch_size = self.forward_ret_dict['batch_size']
        
        tb_dict = {}
        loss = 0
        batch_indices = self.forward_ret_dict['spatial_indices'][:, 0]
        spatial_indices = self.forward_ret_dict['spatial_indices'][:, 1:]

        for idx, pred_dict in enumerate(pred_dicts):
            pred_hm = self.sigmoid(pred_dict['heatmap'])
            pred_reg = torch.cat([pred_dict[head_name] for head_name in self.separate_head_cfg.HEAD_ORDER], dim=1)
            # pred_reg[..., -2:] = pred_reg[..., -2:].sigmoid() * 2 - 1

            target_boxes = target_dicts['target_boxes'][idx]
            target_hm = target_dicts['heatmaps'][idx]

            masks = target_dicts['masks'][idx]
            inds = target_dicts['inds'][idx]
            center_distances = target_dicts['center_distances'][idx]
            candidate_indices = target_dicts['candidate_indices'][idx]
            box_radius = target_dicts['box_radius'][idx]
            
            pred_boxes = []
            pred_cls = []
            target_cls = []
            batch_spatial_indices = []
            for bs_idx in range(batch_size):
                batch_inds = batch_indices==bs_idx
                pred_boxes.append(pred_reg[batch_inds][inds[bs_idx]])
                pred_cls.append(pred_hm[batch_inds][inds[bs_idx]])
                target_cls.append(target_hm[batch_inds][inds[bs_idx]])
                batch_spatial_indices.append(spatial_indices[batch_inds][inds[bs_idx]])

            batch_spatial_indices = torch.stack(batch_spatial_indices)
            target_cls = torch.stack(target_cls)
            pred_boxes = torch.stack(pred_boxes)
            pred_cls = torch.stack(pred_cls)

            #  calculate IoU    
            iou_targets = self.get_iou_targets(pred_boxes.float(), target_boxes, masks, batch_spatial_indices)
 
            # get dynamic positive masks 
            pos_masks = self.get_dynamic_masks(pred_cls, target_cls, pred_boxes, target_boxes, masks, iou_targets, \
                                               self.candidate_num[idx])
            #pos_masks = masks

            # calculate cls loss
            for bs_idx in range(batch_size):
                batch_inds = (batch_indices==bs_idx).nonzero().squeeze()
                cur_target_hm = target_hm[batch_inds]
                cur_target_hm[inds[bs_idx]] *= pos_masks[bs_idx][..., None]
                target_hm[batch_inds] = cur_target_hm

            """v_hm = spconv.SparseConvTensor(
                features=target_hm,
                indices=self.forward_ret_dict['spatial_indices'].int(),
                spatial_shape=self.forward_ret_dict['feature_map_size'],
                batch_size=batch_size
            )
            v_hm = v_hm.dense().permute(2, 3, 0, 1)
            from matplotlib import pyplot as plt
            data = v_hm.view(*self.forward_ret_dict['feature_map_size'], -1).clone()
            data = data.contiguous().detach().cpu().numpy()
            plt.imshow(data, interpolation='nearest')
            plt.show()"""

            hm_loss = self.hm_loss_func(pred_hm, target_hm)
            hm_loss *= self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']

            # calculate reg loss
            loc_loss = self.reg_loss_func(
                pred_boxes.view(-1, pred_boxes.shape[-1]), target_boxes.view(-1, target_boxes.shape[-1]), pos_masks.eq(1).view(-1)
            )
            # loc_loss = (loc_loss * loc_loss.new_tensor(self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['code_weights'])).sum()
            loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']

            tb_dict['hm_loss_head_%d' % idx] = hm_loss.item()
            tb_dict['loc_loss_head_%d' % idx] = loc_loss.item()

            if 'iou' in pred_dict:
                pred_ious = pred_dict['iou']
                batch_pred_ious = []
                for bs_idx in range(batch_size):
                    batch_inds = batch_indices==bs_idx
                    batch_pred_ious.append(pred_ious[batch_inds][inds[bs_idx]])
                batch_pred_ious = torch.stack(batch_pred_ious)
                iou_loss = self.crit_iou(batch_pred_ious.view(-1), iou_targets.view(-1) * 2 - 1, pos_masks.eq(1).view(-1))
                loss += (hm_loss + loc_loss + iou_loss)
                #loss += (hm_loss + loc_loss)
                tb_dict['iou_loss_head_%d' % idx] = iou_loss.item()
            else:
                loss += (hm_loss + loc_loss)

        tb_dict['rpn_loss'] = loss.item()
        return loss, tb_dict

    def get_pseudo_loss(self, batch_dict):
        spatial_features = batch_dict['spatial_features_2d'].features
        target_dicts = self.forward_ret_dict['target_dicts']
        pred_dicts = self.forward_ret_dict['pred_dicts']
        batch_size = self.forward_ret_dict['batch_size']
        batch_indices = self.forward_ret_dict['spatial_indices'][:, 0]
       
        all_object_features = []
        for idx, pred_dict in enumerate(pred_dicts):
            masks = target_dicts['masks'][idx][..., 0].bool()
            inds = target_dicts['inds'][idx][..., 0]
            
            object_features = []
            for bs_idx in range(batch_size):
                batch_inds = batch_indices==bs_idx
                object_features.append(spatial_features[batch_inds][inds[bs_idx]])

            object_features = torch.stack(object_features)
            all_object_features.append(object_features.view(-1, object_features.shape[-1])[masks.view(-1)])

        all_object_features = torch.cat(all_object_features, dim=0)

        return all_object_features
    
    def get_iou_targets(self, box_preds, box_targets, masks, spatial_indices):
        iou_targets = torch.zeros_like(masks).float()
        input_shape = box_preds.shape
        box_preds = box_preds.reshape(-1, input_shape[-1])
        box_targets = box_targets.reshape(-1, input_shape[-1])
        spatial_indices = spatial_indices.reshape(-1, spatial_indices.shape[-1])
        box_inds = masks.view(-1).nonzero().squeeze(-1)

        qboxes = self._get_predicted_boxes(box_preds[box_inds], spatial_indices[box_inds])
        gboxes = self._get_predicted_boxes(box_targets[box_inds], spatial_indices[[box_inds]])

        iou_pos_targets = iou3d_nms_utils.boxes_aligned_iou3d_gpu(qboxes, gboxes).detach()

        iou_targets.view(-1)[box_inds] = iou_pos_targets.squeeze(-1)
        iou_targets = torch.clamp(iou_targets, 0, 1)

        return iou_targets
    
    def _get_predicted_boxes(self, pred_boxes, spatial_indices):
        center, center_z, dim, rot_cos, rot_sin = pred_boxes[..., :2], pred_boxes[..., 2:3], pred_boxes[..., 3:6], \
                                                  pred_boxes[..., 6:7], pred_boxes[..., 7:8]
        dim = torch.exp(torch.clamp(dim, min=-5, max=5))
        angle = torch.atan2(rot_sin, rot_cos)
        xs = (spatial_indices[:, 1:2] + center[:, 0:1]) * self.feature_map_stride * self.voxel_size[0] + self.point_cloud_range[0]
        ys = (spatial_indices[:, 0:1] + center[:, 1:2]) * self.feature_map_stride * self.voxel_size[1] + self.point_cloud_range[1]
        box_part_list = [xs, ys, center_z, dim, angle]
        pred_box = torch.cat((box_part_list), dim=-1)
        return pred_box
    
    def generate_predicted_boxes(self, batch_size, pred_dicts, sparse_indices):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        post_center_limit_range = torch.tensor(post_process_cfg.POST_CENTER_LIMIT_RANGE).cuda().float()

        ret_dict = [{'pred_boxes': [], 'pred_scores': [], 'pred_labels': []} for _ in range(batch_size)]
        for idx, pred_dict in enumerate(pred_dicts):
            batch_hm = pred_dict['heatmap'].sigmoid()
            batch_center = pred_dict['center']
            batch_center_z = pred_dict['center_z']
            batch_dim = pred_dict['dim'].exp()
            batch_rot_cos = pred_dict['rot'][:, 0].unsqueeze(dim=1)
            batch_rot_sin = pred_dict['rot'][:, 1].unsqueeze(dim=1)
            batch_vel = pred_dict['vel'] if 'vel' in self.separate_head_cfg.HEAD_ORDER else None
            batch_iou = (pred_dict['iou'] + 1) * 0.5 if 'iou' in pred_dict else None

            final_pred_dicts = centernet_utils.decode_bbox_from_sparse_heatmap(
                heatmap=batch_hm, rot_cos=batch_rot_cos, rot_sin=batch_rot_sin,
                center=batch_center, center_z=batch_center_z, dim=batch_dim, vel=batch_vel, iou=batch_iou,
                point_cloud_range=self.point_cloud_range,
                voxel_size=self.voxel_size,
                feature_map_stride=self.feature_map_stride,
                sparse_indices=sparse_indices,
                K=post_process_cfg.MAX_OBJ_PER_SAMPLE,
                score_thresh=post_process_cfg.SCORE_THRESH,
                post_center_limit_range=post_center_limit_range
            )

            for bidx, final_dict in enumerate(final_pred_dicts):
                final_dict['pred_labels'] = self.class_id_mapping_each_head[idx][final_dict['pred_labels'].long()]

                if post_process_cfg.get('USE_IOU_TO_RECTIFY_SCORE', False) and 'pred_ious' in final_dict:
                    pred_scores, pred_labels, pred_ious = final_dict['pred_scores'], final_dict['pred_labels'], final_dict['pred_ious']
                    IOU_RECTIFIER = pred_scores.new_tensor(post_process_cfg.IOU_RECTIFIER)
                    final_dict['pred_scores'] = torch.pow(pred_scores, 1 - IOU_RECTIFIER[pred_labels]) * torch.pow(pred_ious, IOU_RECTIFIER[pred_labels])

                if post_process_cfg.NMS_CONFIG.NMS_TYPE == 'class_specific_nms':
                    selected, selected_scores = model_nms_utils.class_specific_nms(
                        box_scores=final_dict['pred_scores'], box_preds=final_dict['pred_boxes'],
                        box_labels=final_dict['pred_labels'], nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=post_process_cfg.NMS_CONFIG.get('SCORE_THRESH', None)
                    )
                else:
                    selected, selected_scores = model_nms_utils.class_agnostic_nms(
                        box_scores=final_dict['pred_scores'], box_preds=final_dict['pred_boxes'],
                        nms_config=post_process_cfg.NMS_CONFIG, score_thresh=None,
                    )

                ret_dict[bidx]['pred_boxes'].append(final_dict['pred_boxes'][selected])
                ret_dict[bidx]['pred_scores'].append(selected_scores)
                ret_dict[bidx]['pred_labels'].append(final_dict['pred_labels'][selected])

        for bidx in range(batch_size):
            ret_dict[bidx]['pred_boxes'] = torch.cat(ret_dict[bidx]['pred_boxes'], dim=0)
            ret_dict[bidx]['pred_scores'] = torch.cat(ret_dict[bidx]['pred_scores'], dim=0)
            ret_dict[bidx]['pred_labels'] = torch.cat(ret_dict[bidx]['pred_labels'], dim=0) + 1

        return ret_dict

    def generate_predicted_boxes_v2(self, batch_size, pred_dicts, sparse_indices):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        post_center_limit_range = torch.tensor(post_process_cfg.POST_CENTER_LIMIT_RANGE).cuda().float()

        ret_dict = [{'pred_boxes': [], 'pred_scores': [], 'pred_labels': []} for _ in range(batch_size)]
        for idx, pred_dict in enumerate(pred_dicts):
            batch_hm = pred_dict['heatmap'].sigmoid()
            batch_bbox = pred_dict['bbox']
            batch_center = batch_bbox[:, :2]
            batch_center_z = batch_bbox[:, 2:3]
            batch_dim = batch_bbox[:, 3:6].exp()
            batch_rot_cos = batch_bbox[:, 6:7].sigmoid() * 2 - 1
            batch_rot_sin = batch_bbox[:, 7:8].sigmoid() * 2 - 1
            batch_vel = pred_dict['vel'] if 'vel' in self.separate_head_cfg.HEAD_ORDER else None
            batch_iou = (pred_dict['iou'] + 1) * 0.5 if 'iou' in pred_dict else None

            final_pred_dicts = centernet_utils.decode_bbox_from_sparse_heatmap(
                heatmap=batch_hm, rot_cos=batch_rot_cos, rot_sin=batch_rot_sin,
                center=batch_center, center_z=batch_center_z, dim=batch_dim, vel=batch_vel, iou=batch_iou,
                point_cloud_range=self.point_cloud_range,
                voxel_size=self.voxel_size,
                feature_map_stride=self.feature_map_stride,
                sparse_indices=sparse_indices,
                K=post_process_cfg.MAX_OBJ_PER_SAMPLE,
                score_thresh=post_process_cfg.SCORE_THRESH,
                post_center_limit_range=post_center_limit_range
            )

            for bidx, final_dict in enumerate(final_pred_dicts):
                final_dict['pred_labels'] = self.class_id_mapping_each_head[idx][final_dict['pred_labels'].long()]

                if post_process_cfg.get('USE_IOU_TO_RECTIFY_SCORE', False) and 'pred_ious' in final_dict:
                    pred_scores, pred_labels, pred_ious = final_dict['pred_scores'], final_dict['pred_labels'], final_dict['pred_ious']
                    IOU_RECTIFIER = pred_scores.new_tensor(post_process_cfg.IOU_RECTIFIER)
                    final_dict['pred_scores'] = torch.pow(pred_scores, 1 - IOU_RECTIFIER[pred_labels]) * torch.pow(pred_ious, IOU_RECTIFIER[pred_labels])

                if post_process_cfg.NMS_CONFIG.NMS_TYPE == 'class_specific_nms':
                    selected, selected_scores = model_nms_utils.class_specific_nms(
                        box_scores=final_dict['pred_scores'], box_preds=final_dict['pred_boxes'],
                        box_labels=final_dict['pred_labels'], nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=post_process_cfg.NMS_CONFIG.get('SCORE_THRESH', None)
                    )
                else:
                    selected, selected_scores = model_nms_utils.class_agnostic_nms(
                        box_scores=final_dict['pred_scores'], box_preds=final_dict['pred_boxes'],
                        nms_config=post_process_cfg.NMS_CONFIG, score_thresh=None,
                    )

                ret_dict[bidx]['pred_boxes'].append(final_dict['pred_boxes'][selected])
                ret_dict[bidx]['pred_scores'].append(selected_scores)
                ret_dict[bidx]['pred_labels'].append(final_dict['pred_labels'][selected])

        for bidx in range(batch_size):
            ret_dict[bidx]['pred_boxes'] = torch.cat(ret_dict[bidx]['pred_boxes'], dim=0)
            ret_dict[bidx]['pred_scores'] = torch.cat(ret_dict[bidx]['pred_scores'], dim=0)
            ret_dict[bidx]['pred_labels'] = torch.cat(ret_dict[bidx]['pred_labels'], dim=0) + 1

        return ret_dict
    
    def forward(self, data_dict):
        x = data_dict['spatial_features_2d']
        spatial_indices = None

        pred_dicts = []
        for head in self.heads_list:
            pred_dict = head(x)
            for k, v in pred_dict.items():
                spatial_indices = v.indices
                pred_dict[k] = v.features
            pred_dicts.append(pred_dict)
        self.forward_ret_dict['pred_dicts'] = pred_dicts
        self.forward_ret_dict['spatial_indices'] = spatial_indices
        self.forward_ret_dict['batch_size'] = data_dict['batch_size']
        self.forward_ret_dict['feature_map_size'] = x.spatial_shape

        if self.training:
            target_dicts = self.assign_targets(data_dict['gt_boxes'], x.spatial_shape[-2:], spatial_indices)
            self.forward_ret_dict['target_dicts'] = target_dicts
        
        else:
            data_dict['final_box_dicts'] = self.generate_predicted_boxes(data_dict['batch_size'], pred_dicts, \
                                                                         spatial_indices)

        return data_dict
