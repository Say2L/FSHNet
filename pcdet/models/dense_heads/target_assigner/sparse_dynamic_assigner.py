import torch
from scipy.optimize import linear_sum_assignment
from pcdet.ops.iou3d_nms import iou3d_nms_cuda
from pcdet.utils import box_utils

def height_overlaps(boxes1, boxes2):
    """
    Calculate height overlaps of two boxes.
    """
    boxes1_top_height = (boxes1[:,2]+ boxes1[:,5]).view(-1, 1)
    boxes1_bottom_height = boxes1[:,2].view(-1, 1)
    boxes2_top_height = (boxes2[:,2]+boxes2[:,5]).view(1, -1)
    boxes2_bottom_height = boxes2[:,2].view(1, -1)

    heighest_of_bottom = torch.max(boxes1_bottom_height, boxes2_bottom_height)
    lowest_of_top = torch.min(boxes1_top_height, boxes2_top_height)
    overlaps_h = torch.clamp(lowest_of_top - heighest_of_bottom, min=0)
    return overlaps_h


def overlaps(boxes1, boxes2):
    """
    Calculate 3D overlaps of two boxes.
    """
    rows = len(boxes1)
    cols = len(boxes2)
    if rows * cols == 0:
        return boxes1.new(rows, cols)

    # height overlap
    overlaps_h = height_overlaps(boxes1, boxes2)
    boxes1_bev = boxes1[:,:7]
    boxes2_bev = boxes2[:,:7]

    # bev overlap
    overlaps_bev = boxes1_bev.new_zeros(
        (boxes1_bev.shape[0], boxes2_bev.shape[0])
    ).cuda()  # (N, M)
    iou3d_nms_cuda.boxes_overlap_bev_gpu(
        boxes1_bev.contiguous().cuda(), boxes2_bev.contiguous().cuda(), overlaps_bev
    )

    # 3d overlaps
    overlaps_3d = overlaps_bev.to(boxes1.device) * overlaps_h

    volume1 = (boxes1[:, 3] * boxes1[:, 4] * boxes1[:, 5]).view(-1, 1)
    volume2 = (boxes2[:, 3] * boxes2[:, 4] * boxes2[:, 5]).view(1, -1)
        
    iou3d = overlaps_3d / torch.clamp(volume1 + volume2 - overlaps_3d, min=1e-8)

    return iou3d


class SparseDynamicAssigner3D:
    def __init__(self, cls_cost, reg_cost, iou_cost, candidate_num, voxel_size):
        self.cls_cost = cls_cost
        self.reg_cost = reg_cost
        self.iou_cost = iou_cost
        self.candidate_num = candidate_num
        self.voxel_size = voxel_size

    def decode_bbox(self, bboxes):
        center = bboxes[..., :2]
        height = bboxes[..., [2]]
        dim = bboxes[..., 3:6]
        rot = bboxes[..., 6:8]
        if bboxes.shape[-1] > 8:
            vel = bboxes[..., 8:]
        else:
            vel = None

        center[:, 0] = center[:, 0] * self.voxel_size[0]
        center[:, 1] = center[:, 1] * self.voxel_size[1]
        dim = dim.exp()
        rotc, rots = rot[:, 0:1], rot[:, 1:2]
        rot = torch.atan2(rots, rotc)

        if vel is None:
            final_box_preds = torch.cat([center, height, dim, rot], dim=1)
        else:
            final_box_preds = torch.cat([center, height, dim, rot, vel], dim=1)

        return final_box_preds
    
    def heat_cost(self, gt_labels, cls_pred):
        weight = self.cls_cost.get('weight', 1)

        cls_pred = cls_pred.sigmoid()
        cls_cost = 1 - cls_pred[gt_labels, :]
    
        return cls_cost * weight
    
    def bevbox_cost(self, gt_bboxes, bboxes, r_factor=0.5):
        weight = self.reg_cost.get('weight', 0.25)
        center_dist = torch.cdist(gt_bboxes[:, :2], bboxes[:, :2] , p=1)
        u, rwiou = box_utils.get_rwiou(bboxes[None, ...], gt_bboxes[:, None, :], r_factor, self.voxel_size)
        reg_cost = 1 - torch.clamp(rwiou, min=0, max = 1.0) + u

        return reg_cost * weight, center_dist
    
    def iou3d_cost(self, gt_bboxes, pd_bboxes):
        gt_bboxes_decoded = self.decode_bbox(gt_bboxes)
        pd_bboxes_decoded = self.decode_bbox(pd_bboxes)
        iou = overlaps(gt_bboxes_decoded, pd_bboxes_decoded)
        weight = self.iou_cost.get('weight', 0.25)
        iou_cost = 1 - iou
        return iou_cost * weight, iou

    def assign(self, bboxes, gt_bboxes, gt_labels, cls_pred, **kwargs):
        num_gts, num_bboxes = gt_bboxes.size(0), bboxes.size(0)
        # candidate_num = min(self.candidate_num, num_bboxes // num_gts)
        # 1. assign -1 by default
        assigned_gt_inds = bboxes.new_full((num_bboxes,), -1, dtype=torch.long)
        assigned_pos_inds = bboxes.new_full((num_gts, self.candidate_num), -1, dtype=torch.long)
        assigned_labels = bboxes.new_full((num_bboxes,), -1, dtype=torch.long)
        assigned_gt_masks = bboxes.new_full((num_bboxes,), 0, dtype=torch.long)
        assigned_pos_masks = bboxes.new_full((num_gts, self.candidate_num), 0, dtype=torch.bool)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return num_gts, assigned_gt_inds, max_overlaps, assigned_labels

        # 2. compute the weighted costs
        cls_cost = self.heat_cost(gt_labels, cls_pred.T)
        reg_cost, center_dist = self.bevbox_cost(gt_bboxes.clone(), bboxes.clone())
        iou_cost, iou = self.iou3d_cost(gt_bboxes.clone(), bboxes.clone())

        # dist mask
        dx, dy = gt_bboxes[:, 3].exp() * 1.2, gt_bboxes[:, 4].exp() * 1.2
        dx, dy = dx / self.voxel_size[0], dy / self.voxel_size[1]
        box_radius = torch.sqrt((dx / 2)**2 + (dy / 2)**2)
        dist_mask = center_dist < box_radius[:, None]

        # weighted sum of above costs
        cost = cls_cost + reg_cost + (1 - dist_mask.float()) * 100

        candidate_dist, candidate_col_inds = torch.topk(center_dist, self.candidate_num, dim=-1, largest=False)
        candidate_row_inds = torch.arange(num_gts).type_as(candidate_col_inds)[:, None].repeat(1, self.candidate_num)
        candidate_cost = cost[candidate_row_inds, candidate_col_inds]

        sort_cost, sort_inds = torch.sort(candidate_cost, dim=-1)
        candidate_col_inds = candidate_col_inds.gather(index=sort_inds, dim=-1)
        assigned_pos_inds = candidate_col_inds

        assigned_iou = iou[candidate_row_inds, candidate_col_inds] * dist_mask[candidate_row_inds, candidate_col_inds]

        tmp_mask = dist_mask[candidate_row_inds, candidate_col_inds]
        posnum = assigned_iou.sum(1).long().clamp(1)
        
        for i in range(num_gts):
            assigned_pos_masks[i, :posnum[i]] = True
            # pos_mask[:posnum[i], i] = True
            
        matched_row_inds = candidate_row_inds.flatten()[assigned_pos_masks.flatten()]
        matched_col_inds = candidate_col_inds.flatten()[assigned_pos_masks.flatten()]
        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_col_inds] = matched_row_inds + 1
        assigned_labels[matched_col_inds] = gt_labels[matched_row_inds]

        max_overlaps = torch.zeros_like(iou.max(0).values)
        max_overlaps[matched_col_inds] = iou[matched_row_inds, matched_col_inds]

        assigned_gt_masks[matched_col_inds] = dist_mask[matched_row_inds, matched_col_inds].long()
        # tmp_mask = dist_mask[matched_row_inds, matched_col_inds].long()

        return assigned_gt_inds, assigned_pos_inds, assigned_gt_masks, assigned_pos_masks, assigned_iou, max_overlaps