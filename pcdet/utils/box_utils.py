import numpy as np
import scipy
import torch
import copy
from scipy.spatial import Delaunay

from ..ops.roiaware_pool3d import roiaware_pool3d_utils
from . import common_utils


def get_rwiou(bboxes1, bboxes2, r_factor=1, voxel_size=[0.8, 0.8]):
    x1u, y1u, z1u = bboxes1[...,0], bboxes1[...,1], bboxes1[...,2]
    l1, w1, h1 =  torch.exp(bboxes1[...,3]), torch.exp(bboxes1[...,4]), torch.exp(bboxes1[...,5])
    s1, c1 = bboxes1[...,6], bboxes1[...,7]
    x2u, y2u, z2u = bboxes2[...,0], bboxes2[...,1], bboxes2[...,2]
    l2, w2, h2 =  torch.exp(bboxes2[...,3]), torch.exp(bboxes2[...,4]), torch.exp(bboxes2[...,5])
    s2, c2 = bboxes2[...,6], bboxes2[...,7]

    x1 = x1u * voxel_size[0]
    y1 = y1u * voxel_size[1]
    z1 = z1u
    x2 = x2u * voxel_size[0]
    y2 = y2u * voxel_size[1]
    z2 = z2u

    # clamp is necessray to aviod inf.
    eps = 1e-4
    l1, w1, h1 = torch.clamp(l1, min=eps, max=30), torch.clamp(w1, min=eps, max=10), torch.clamp(h1, min=eps, max=10)
    s1, c1 = torch.clamp(s1, min=-1, max=1), torch.clamp(c1, min=-1, max=1)
    volume_1 = l1 * w1 * h1
    volume_2 = l2 * w2 * h2

    inter_l = torch.max(x1 - l1 / 2, x2 - l2 / 2)
    inter_r = torch.min(x1 + l1 / 2, x2 + l2 / 2)
    inter_t = torch.max(y1 - w1 / 2, y2 - w2 / 2)
    inter_b = torch.min(y1 + w1 / 2, y2 + w2 / 2)
    inter_u = torch.max(z1 - h1 / 2, z2 - h2 / 2)
    inter_d = torch.min(z1 + h1 / 2, z2 + h2 / 2)

    inter_volume = torch.clamp((inter_r - inter_l),min=0) * torch.clamp((inter_b - inter_t),min=0) \
        * torch.clamp((inter_d - inter_u),min=0)
    inter_volume *= (1 - r_factor * torch.abs(s1 - s2) / 2) * (1 - r_factor * torch.abs(c1 - c2) / 2)
   
    c_l = torch.min(x1 - l1 / 2,x2 - l2 / 2)
    c_r = torch.max(x1 + l1 / 2,x2 + l2 / 2)
    c_t = torch.min(y1 - w1 / 2,y2 - w2 / 2)
    c_b = torch.max(y1 + w1 / 2,y2 + w2 / 2)
    c_u = torch.min(z1 - h1 / 2,z2 - h2 / 2)
    c_d = torch.max(z1 + h1 / 2,z2 + h2 / 2)

    inter_diag = (x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2
    c_diag = torch.clamp((c_r - c_l),min=0)**2 + torch.clamp((c_b - c_t),min=0)**2 + torch.clamp((c_d - c_u),min=0)**2

    union = volume_1 + volume_2 - inter_volume
    u = (inter_diag) / c_diag
    rdiou = inter_volume / union
    return u, rdiou

def in_hull(p, hull):
    """
    :param p: (N, K) test points
    :param hull: (M, K) M corners of a box
    :return (N) bool
    """
    try:
        if not isinstance(hull, Delaunay):
            hull = Delaunay(hull)
        flag = hull.find_simplex(p) >= 0
    except scipy.spatial.qhull.QhullError:
        print('Warning: not a hull %s' % str(hull))
        flag = np.zeros(p.shape[0], dtype=np.bool)

    return flag


def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """
    boxes3d, is_numpy = common_utils.check_numpy_to_torch(boxes3d)

    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = common_utils.rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d

def corners_rect_to_camera(corners):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        corners:  (8, 3) [x0, y0, z0, ...], (x, y, z) is the point coordinate in image rect

    Returns:
        boxes_rect:  (7,) [x, y, z, l, h, w, r] in rect camera coords
    """
    height_group = [(0, 4), (1, 5), (2, 6), (3, 7)]
    width_group = [(0, 1), (2, 3), (4, 5), (6, 7)]
    length_group = [(0, 3), (1, 2), (4, 7), (5, 6)]
    vector_group = [(0, 3), (1, 2), (4, 7), (5, 6)]
    height, width, length = 0., 0., 0.
    vector = np.zeros(2, dtype=np.float32)
    for index_h, index_w, index_l, index_v in zip(height_group, width_group, length_group, vector_group):
        height += np.linalg.norm(corners[index_h[0], :] - corners[index_h[1], :])
        width += np.linalg.norm(corners[index_w[0], :] - corners[index_w[1], :])
        length += np.linalg.norm(corners[index_l[0], :] - corners[index_l[1], :])
        vector[0] += (corners[index_v[0], :] - corners[index_v[1], :])[0]
        vector[1] += (corners[index_v[0], :] - corners[index_v[1], :])[2]

    height, width, length = height*1.0/4, width*1.0/4, length*1.0/4
    rotation_y = -np.arctan2(vector[1], vector[0])

    center_point = corners.mean(axis=0)
    center_point[1] += height/2
    camera_rect = np.concatenate([center_point, np.array([length, height, width, rotation_y])])

    return camera_rect


def mask_boxes_outside_range_numpy(boxes, limit_range, min_num_corners=1, use_center_to_filter=True):
    """
    Args:
        boxes: (N, 7) [x, y, z, dx, dy, dz, heading, ...], (x, y, z) is the box center
        limit_range: [minx, miny, minz, maxx, maxy, maxz]
        min_num_corners:

    Returns:

    """
    if boxes.shape[1] > 7:
        boxes = boxes[:, 0:7]
    if use_center_to_filter:
        box_centers = boxes[:, 0:3]
        mask = ((box_centers >= limit_range[0:3]) & (box_centers <= limit_range[3:6])).all(axis=-1)
    else:
        corners = boxes_to_corners_3d(boxes)  # (N, 8, 3)
        corners = corners[:, :, 0:2]
        mask = ((corners >= limit_range[0:2]) & (corners <= limit_range[3:5])).all(axis=2)
        mask = mask.sum(axis=1) >= min_num_corners  # (N)

    return mask


def remove_points_in_boxes3d(points, boxes3d):
    """
    Args:
        points: (num_points, 3 + C)
        boxes3d: (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center, each box DO NOT overlaps

    Returns:

    """
    boxes3d, is_numpy = common_utils.check_numpy_to_torch(boxes3d)
    points, is_numpy = common_utils.check_numpy_to_torch(points)
    point_masks = roiaware_pool3d_utils.points_in_boxes_cpu(points[:, 0:3], boxes3d)
    points = points[point_masks.sum(dim=0) == 0]

    return points.numpy() if is_numpy else points


def boxes3d_kitti_camera_to_lidar(boxes3d_camera, calib):
    """
    Args:
        boxes3d_camera: (N, 7) [x, y, z, l, h, w, r] in rect camera coords
        calib:

    Returns:
        boxes3d_lidar: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    """
    boxes3d_camera_copy = copy.deepcopy(boxes3d_camera)
    xyz_camera, r = boxes3d_camera_copy[:, 0:3], boxes3d_camera_copy[:, 6:7]
    l, h, w = boxes3d_camera_copy[:, 3:4], boxes3d_camera_copy[:, 4:5], boxes3d_camera_copy[:, 5:6]

    xyz_lidar = calib.rect_to_lidar(xyz_camera)
    xyz_lidar[:, 2] += h[:, 0] / 2
    return np.concatenate([xyz_lidar, l, w, h, -(r + np.pi / 2)], axis=-1)


def boxes3d_kitti_fakelidar_to_lidar(boxes3d_lidar):
    """
    Args:
        boxes3d_fakelidar: (N, 7) [x, y, z, w, l, h, r] in old LiDAR coordinates, z is bottom center

    Returns:
        boxes3d_lidar: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    """
    boxes3d_lidar_copy = copy.deepcopy(boxes3d_lidar)
    w, l, h = boxes3d_lidar_copy[:, 3:4], boxes3d_lidar_copy[:, 4:5], boxes3d_lidar_copy[:, 5:6]
    r = boxes3d_lidar_copy[:, 6:7]

    boxes3d_lidar_copy[:, 2] += h[:, 0] / 2
    return np.concatenate([boxes3d_lidar_copy[:, 0:3], l, w, h, -(r + np.pi / 2)], axis=-1)


def boxes3d_kitti_lidar_to_fakelidar(boxes3d_lidar):
    """
    Args:
        boxes3d_lidar: (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
        boxes3d_fakelidar: [x, y, z, w, l, h, r] in old LiDAR coordinates, z is bottom center

    """
    boxes3d_lidar_copy = copy.deepcopy(boxes3d_lidar)
    dx, dy, dz = boxes3d_lidar_copy[:, 3:4], boxes3d_lidar_copy[:, 4:5], boxes3d_lidar_copy[:, 5:6]
    heading = boxes3d_lidar_copy[:, 6:7]

    boxes3d_lidar_copy[:, 2] -= dz[:, 0] / 2
    return np.concatenate([boxes3d_lidar_copy[:, 0:3], dy, dx, dz, -heading - np.pi / 2], axis=-1)


def enlarge_box3d(boxes3d, extra_width=(0, 0, 0)):
    """
    Args:
        boxes3d: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
        extra_width: [extra_x, extra_y, extra_z]

    Returns:

    """
    boxes3d, is_numpy = common_utils.check_numpy_to_torch(boxes3d)
    large_boxes3d = boxes3d.clone()

    large_boxes3d[:, 3:6] += boxes3d.new_tensor(extra_width)[None, :]
    return large_boxes3d


def boxes3d_lidar_to_kitti_camera(boxes3d_lidar, calib):
    """
    :param boxes3d_lidar: (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
    :param calib:
    :return:
        boxes3d_camera: (N, 7) [x, y, z, l, h, w, r] in rect camera coords
    """
    boxes3d_lidar_copy = copy.deepcopy(boxes3d_lidar)
    xyz_lidar = boxes3d_lidar_copy[:, 0:3]
    l, w, h = boxes3d_lidar_copy[:, 3:4], boxes3d_lidar_copy[:, 4:5], boxes3d_lidar_copy[:, 5:6]
    r = boxes3d_lidar_copy[:, 6:7]

    xyz_lidar[:, 2] -= h.reshape(-1) / 2
    xyz_cam = calib.lidar_to_rect(xyz_lidar)
    # xyz_cam[:, 1] += h.reshape(-1) / 2
    r = -r - np.pi / 2
    return np.concatenate([xyz_cam, l, h, w, r], axis=-1)


def boxes3d_to_corners3d_kitti_camera(boxes3d, bottom_center=True):
    """
    :param boxes3d: (N, 7) [x, y, z, l, h, w, ry] in camera coords, see the definition of ry in KITTI dataset
    :param bottom_center: whether y is on the bottom center of object
    :return: corners3d: (N, 8, 3)
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    """
    boxes_num = boxes3d.shape[0]
    l, h, w = boxes3d[:, 3], boxes3d[:, 4], boxes3d[:, 5]
    x_corners = np.array([l / 2., l / 2., -l / 2., -l / 2., l / 2., l / 2., -l / 2., -l / 2], dtype=np.float32).T
    z_corners = np.array([w / 2., -w / 2., -w / 2., w / 2., w / 2., -w / 2., -w / 2., w / 2.], dtype=np.float32).T
    if bottom_center:
        y_corners = np.zeros((boxes_num, 8), dtype=np.float32)
        y_corners[:, 4:8] = -h.reshape(boxes_num, 1).repeat(4, axis=1)  # (N, 8)
    else:
        y_corners = np.array([h / 2., h / 2., h / 2., h / 2., -h / 2., -h / 2., -h / 2., -h / 2.], dtype=np.float32).T

    ry = boxes3d[:, 6]
    zeros, ones = np.zeros(ry.size, dtype=np.float32), np.ones(ry.size, dtype=np.float32)
    rot_list = np.array([[np.cos(ry), zeros, -np.sin(ry)],
                         [zeros, ones, zeros],
                         [np.sin(ry), zeros, np.cos(ry)]])  # (3, 3, N)
    R_list = np.transpose(rot_list, (2, 0, 1))  # (N, 3, 3)

    temp_corners = np.concatenate((x_corners.reshape(-1, 8, 1), y_corners.reshape(-1, 8, 1),
                                   z_corners.reshape(-1, 8, 1)), axis=2)  # (N, 8, 3)
    rotated_corners = np.matmul(temp_corners, R_list)  # (N, 8, 3)
    x_corners, y_corners, z_corners = rotated_corners[:, :, 0], rotated_corners[:, :, 1], rotated_corners[:, :, 2]

    x_loc, y_loc, z_loc = boxes3d[:, 0], boxes3d[:, 1], boxes3d[:, 2]

    x = x_loc.reshape(-1, 1) + x_corners.reshape(-1, 8)
    y = y_loc.reshape(-1, 1) + y_corners.reshape(-1, 8)
    z = z_loc.reshape(-1, 1) + z_corners.reshape(-1, 8)

    corners = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1), z.reshape(-1, 8, 1)), axis=2)

    return corners.astype(np.float32)


def boxes3d_kitti_camera_to_imageboxes(boxes3d, calib, image_shape=None):
    """
    :param boxes3d: (N, 7) [x, y, z, l, h, w, r] in rect camera coords
    :param calib:
    :return:
        box_2d_preds: (N, 4) [x1, y1, x2, y2]
    """
    corners3d = boxes3d_to_corners3d_kitti_camera(boxes3d)
    pts_img, _ = calib.rect_to_img(corners3d.reshape(-1, 3))
    corners_in_image = pts_img.reshape(-1, 8, 2)

    min_uv = np.min(corners_in_image, axis=1)  # (N, 2)
    max_uv = np.max(corners_in_image, axis=1)  # (N, 2)
    boxes2d_image = np.concatenate([min_uv, max_uv], axis=1)
    if image_shape is not None:
        boxes2d_image[:, 0] = np.clip(boxes2d_image[:, 0], a_min=0, a_max=image_shape[1] - 1)
        boxes2d_image[:, 1] = np.clip(boxes2d_image[:, 1], a_min=0, a_max=image_shape[0] - 1)
        boxes2d_image[:, 2] = np.clip(boxes2d_image[:, 2], a_min=0, a_max=image_shape[1] - 1)
        boxes2d_image[:, 3] = np.clip(boxes2d_image[:, 3], a_min=0, a_max=image_shape[0] - 1)

    return boxes2d_image


def boxes_iou_normal(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 4) [x1, y1, x2, y2]
        boxes_b: (M, 4) [x1, y1, x2, y2]

    Returns:

    """
    assert boxes_a.shape[1] == boxes_b.shape[1] == 4
    x_min = torch.max(boxes_a[:, 0, None], boxes_b[None, :, 0])
    x_max = torch.min(boxes_a[:, 2, None], boxes_b[None, :, 2])
    y_min = torch.max(boxes_a[:, 1, None], boxes_b[None, :, 1])
    y_max = torch.min(boxes_a[:, 3, None], boxes_b[None, :, 3])
    x_len = torch.clamp_min(x_max - x_min, min=0)
    y_len = torch.clamp_min(y_max - y_min, min=0)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    a_intersect_b = x_len * y_len
    iou = a_intersect_b / torch.clamp_min(area_a[:, None] + area_b[None, :] - a_intersect_b, min=1e-6)
    return iou


def boxes3d_lidar_to_aligned_bev_boxes(boxes3d):
    """
    Args:
        boxes3d: (N, 7 + C) [x, y, z, dx, dy, dz, heading] in lidar coordinate

    Returns:
        aligned_bev_boxes: (N, 4) [x1, y1, x2, y2] in the above lidar coordinate
    """
    rot_angle = common_utils.limit_period(boxes3d[:, 6], offset=0.5, period=np.pi).abs()
    choose_dims = torch.where(rot_angle[:, None] < np.pi / 4, boxes3d[:, [3, 4]], boxes3d[:, [4, 3]])
    aligned_bev_boxes = torch.cat((boxes3d[:, 0:2] - choose_dims / 2, boxes3d[:, 0:2] + choose_dims / 2), dim=1)
    return aligned_bev_boxes


def boxes3d_nearest_bev_iou(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (N, 7) [x, y, z, dx, dy, dz, heading]

    Returns:

    """
    boxes_bev_a = boxes3d_lidar_to_aligned_bev_boxes(boxes_a)
    boxes_bev_b = boxes3d_lidar_to_aligned_bev_boxes(boxes_b)

    return boxes_iou_normal(boxes_bev_a, boxes_bev_b)


def area(box) -> torch.Tensor:
    """
    Computes the area of all the boxes.

    Returns:
        torch.Tensor: a vector with areas of each box.
    """
    area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
    return area


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def pairwise_iou(boxes1, boxes2) -> torch.Tensor:
    """
    Given two lists of boxes of size N and M,
    compute the IoU (intersection over union)
    between __all__ N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.

    Returns:
        Tensor: IoU, sized [N,M].
    """
    area1 = area(boxes1)
    area2 = area(boxes2)

    width_height = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.max(
        boxes1[:, None, :2], boxes2[:, :2]
    )  # [N,M,2]

    width_height.clamp_(min=0)  # [N,M,2]
    inter = width_height.prod(dim=2)  # [N,M]
    del width_height

    # handle empty boxes
    iou = torch.where(
        inter > 0,
        inter / (area1[:, None] + area2 - inter),
        torch.zeros(1, dtype=inter.dtype, device=inter.device),
    )
    return iou


def center_to_corner2d(center, dim):
    corners_norm = torch.tensor([[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]], device=dim.device).type_as(center)  # (4, 2)
    corners = dim.view([-1, 1, 2]) * corners_norm.view([1, 4, 2])  # (N, 4, 2)
    corners = corners + center.view(-1, 1, 2)
    return corners


def bbox3d_overlaps_diou(pred_boxes, gt_boxes):
    """
    https://github.com/agent-sgs/PillarNet/blob/master/det3d/core/utils/center_utils.py
    Args:
        pred_boxes (N, 7): 
        gt_boxes (N, 7): 

    Returns:
        _type_: _description_
    """
    assert pred_boxes.shape[0] == gt_boxes.shape[0]

    qcorners = center_to_corner2d(pred_boxes[:, :2], pred_boxes[:, 3:5])  # (N, 4, 2)
    gcorners = center_to_corner2d(gt_boxes[:, :2], gt_boxes[:, 3:5])  # (N, 4, 2)   

    inter_max_xy = torch.minimum(qcorners[:, 2], gcorners[:, 2])
    inter_min_xy = torch.maximum(qcorners[:, 0], gcorners[:, 0])
    out_max_xy = torch.maximum(qcorners[:, 2], gcorners[:, 2])
    out_min_xy = torch.minimum(qcorners[:, 0], gcorners[:, 0])

    # calculate area
    volume_pred_boxes = pred_boxes[:, 3] * pred_boxes[:, 4] * pred_boxes[:, 5]
    volume_gt_boxes = gt_boxes[:, 3] * gt_boxes[:, 4] * gt_boxes[:, 5]

    inter_h = torch.minimum(pred_boxes[:, 2] + 0.5 * pred_boxes[:, 5], gt_boxes[:, 2] + 0.5 * gt_boxes[:, 5]) - \
              torch.maximum(pred_boxes[:, 2] - 0.5 * pred_boxes[:, 5], gt_boxes[:, 2] - 0.5 * gt_boxes[:, 5])
    inter_h = torch.clamp(inter_h, min=0)

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    volume_inter = inter[:, 0] * inter[:, 1] * inter_h
    volume_union = volume_gt_boxes + volume_pred_boxes - volume_inter

    # boxes_iou3d_gpu(pred_boxes, gt_boxes)
    inter_diag = torch.pow(gt_boxes[:, 0:3] - pred_boxes[:, 0:3], 2).sum(-1)

    outer_h = torch.maximum(gt_boxes[:, 2] + 0.5 * gt_boxes[:, 5], pred_boxes[:, 2] + 0.5 * pred_boxes[:, 5]) - \
              torch.minimum(gt_boxes[:, 2] - 0.5 * gt_boxes[:, 5], pred_boxes[:, 2] - 0.5 * pred_boxes[:, 5])
    outer_h = torch.clamp(outer_h, min=0)
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = outer[:, 0] ** 2 + outer[:, 1] ** 2 + outer_h ** 2

    dious = volume_inter / volume_union - inter_diag / outer_diag
    dious = torch.clamp(dious, min=-1.0, max=1.0)

    return dious