import numpy as np


'''
select bbox
'''


'''
Corruptions
'''

def density(pointcloud, severity):
    N, C = pointcloud.shape
    num = int(N*0.1)
    c = [int(0.1 * N), int(0.2 * N), int(0.3 * N), int(0.4 * N), int(0.5 * N)][severity - 1]
    idx = np.random.choice(N, c, replace=False)
    pointcloud = np.delete(pointcloud, idx, axis=0)
    return pointcloud

def cutout(pointcloud, severity):
    N, C = pointcloud.shape
    #from 30 changed to 3000 to qualify kitti
    c = [(1,int(N*0.3)), (1,int(N*0.4)), (1,int(N*0.5)), (1,int(N*0.6)), (1,int(N*0.7))][severity-1]
    for _ in range(c[0]):
        i = np.random.choice(pointcloud.shape[0],1)
        picked = pointcloud[i]
        dist = np.sum((pointcloud - picked)**2, axis=1, keepdims=True)
        idx = np.argpartition(dist, c[1], axis=0)[:c[1]]
        # pointcloud[idx.squeeze()] = 0
        pointcloud = np.delete(pointcloud, idx.squeeze(), axis=0)
    # print(pointcloud.shape)
    return pointcloud

def gaussian(pointcloud, severity):
    N, C = pointcloud.shape # N*3
    c = [0.02, 0.04, 0.06, 0.08, 0.10][severity-1]
    jitter = np.random.normal(size=(N, C)) * c
    new_pc = (pointcloud + jitter).astype('float32')
    return new_pc


def uniform(pointcloud, severity):
    N, C = pointcloud.shape
    c = [0.02, 0.04, 0.06, 0.08, 0.10][severity - 1]
    jitter = np.random.uniform(-c, c, (N, C))
    new_pc = (pointcloud + jitter).astype('float32')
    return new_pc


def impulse(pointcloud, severity):
    N, C = pointcloud.shape
    c = [N // 30, N // 25, N // 20, N // 15, N // 10][severity - 1]
    index = np.random.choice(N, c, replace=False)
    pointcloud[index] += np.random.choice([-1, 1], size=(c, C)) * 0.1
    return pointcloud

'''
bbox_convert
'''

def to_Max2(points, gt_boxes_lidar):
    """
    Args:
        points: N x 3+C
        gt_boxes_lidar: 7
    Returns:
        points normalized to max-2 unit square box: N x 3+C
    """
    # shift
    points[:, :3] = points[:, :3] - gt_boxes_lidar[:3]
    # normalize to 2 units
    points[:, :3] = points[:, :3] / np.max(gt_boxes_lidar[3:6]) * 2
    # reversely rotate
    angle = -gt_boxes_lidar[6]
    cosa = np.cos(angle)
    sina = np.sin(angle)
    rot_matrix = np.array(
        [cosa, sina, 0.0,
         -sina, cosa, 0.0,
         0.0, 0.0, 1.0]).reshape(3, 3)
    points_rot = np.matmul(points[:, 0:3], rot_matrix)
    points = np.hstack((points_rot, points[:, 3:].reshape(-1, 1)))

    return points


def to_Lidar(points, gt_boxes_lidar):
    """
    Args:
        points: N x 3+C
        gt_boxes_lidar: 7
    Returns:
        points denormalized to lidar coordinates
    """
    angle = gt_boxes_lidar[6]
    # along_z
    cosa = np.cos(angle)
    sina = np.sin(angle)
    rot_matrix = np.array(
        [cosa, sina, 0.0,
         -sina, cosa, 0.0,
         0.0, 0.0, 1.0]).reshape(3, 3)
    points_rot = np.matmul(points[:, 0:3], rot_matrix)
    points = np.hstack((points_rot, points[:, 3:].reshape(-1, 1)))
    # denormalize to lidar
    points[:, :3] = points[:, :3] * np.max(gt_boxes_lidar[3:6]) / 2
    # shift
    points[:, :3] = points[:, :3] + gt_boxes_lidar[:3]

    return points

# normalize
def normalize_gt(points, gt_box_ratio):
    """
    Args:
        points: N x 3+C
        gt_box_ratio: 3
    Returns:
        limit points to gt: N x 3+C
    """
    if points.shape[0] != 0:
        box_boundary_normalized = gt_box_ratio/np.max(gt_box_ratio)
        for i in range(3):
            indicator = np.max(np.abs(points[:,i])) / box_boundary_normalized[i]
            if indicator > 1:
                points[:,i] /= indicator
    return points

def shear(pointcloud, severity,gt_boxes):

    N, _ = pointcloud.shape
    c = [0.05, 0.1, 0.15, 0.2, 0.25][severity - 1]

    # convert to max-2
    pts_obj_max2 = to_Max2(pointcloud, gt_boxes)
    # shear
    b = np.random.uniform(c - 0.05, c + 0.05) * np.random.choice([-1, 1])
    d = np.random.uniform(c - 0.05, c + 0.05) * np.random.choice([-1, 1])
    e = np.random.uniform(c - 0.05, c + 0.05) * np.random.choice([-1, 1])
    f = np.random.uniform(c - 0.05, c + 0.05) * np.random.choice([-1, 1])
    matrix = np.array([1, 0, b,
                       d, 1, e,
                       f, 0, 1]).reshape(3, 3)
                       
    new_pc = np.matmul(pts_obj_max2[:, :3], matrix).astype('float32')

    pts_obj_max2_crp = np.hstack((new_pc, pts_obj_max2[:, 3].reshape(-1, 1)))
    pts_obj_max2_crp = normalize_gt(pts_obj_max2_crp, gt_boxes[3:6])
    pts_cor = to_Lidar(pts_obj_max2_crp, gt_boxes)

    return pts_cor



def scale(pointcloud, severity,gt_boxes):
    N, _ = pointcloud.shape
    c = [0.04, 0.08, 0.12, 0.16, 0.20][severity-1]
    xs_list,ys_list,zs_list=[],[],[]

    # convert to max-2
    pts_obj_max2 = to_Max2(pointcloud, gt_boxes)
    ## scale on two randomly selected directions
    xs, ys, zs = 1.0, 1.0, 1.0
    r = np.random.randint(0,3)
    t = np.random.choice([-1,1])
    if r == 0:
        xs += c * t
    elif r == 1:
        ys += c * t
    else:
        zs += c * t
    matrix = np.array([[xs,0,0,0],[0,ys,0,0],[0,0,zs,0],[0,0,0,1]])
    pts_obj_max2_crp = np.matmul(pts_obj_max2, matrix)
    pts_obj_max2_crp[:,2] += (zs-1) * gt_boxes[5]/np.max(gt_boxes[3:6])
    xs_list.append(xs)
    ys_list.append(ys)
    zs_list.append(zs)
    # convert to Lidar
    pts_cor = to_Lidar(pts_obj_max2_crp, gt_boxes)

    return pts_cor


def rotation(pointcloud,severity,gt_boxes):
    N, _ = pointcloud.shape
    c = [1, 3, 5, 7, 9][severity-1]
    beta = np.random.uniform(c-1,c+1) * np.random.choice([-1,1]) * np.pi / 180.
    # convert to max-2
    pts_obj_max2 = to_Max2(pointcloud, gt_boxes)
    ## rotation
    matrix_roration_z = np.array([[np.cos(beta),np.sin(beta),0],[-np.sin(beta),np.cos(beta),0],[0,0,1]])
    pts_rotated = np.matmul(pts_obj_max2[:,:3], matrix_roration_z)
    pts_obj_max2_crp = np.hstack((pts_rotated, pts_obj_max2[:,3].reshape(-1,1)))
    # convert to lidar
    pts_cor = to_Lidar(pts_obj_max2_crp, gt_boxes)

    return pts_cor

def moving_object(pointcloud, severity):
    # for kitti: the x is forward
    N, C = pointcloud.shape
    c = [0.2, 0.4, 0.6, 0.8, 1.0][severity - 1]
    m1, m2 = float(c/2), c
    x_min, x_max = pointcloud[:, 0].min(), pointcloud[:, 0].max()
    x_l = (x_max - x_min) / 3
    mask1 = (pointcloud[:, 0] > x_min) & (pointcloud[:, 0] <= x_min + x_l)
    mask2 = (pointcloud[:, 0] <= x_max) & (pointcloud[:, 0] > x_min + x_l)
    pointcloud[mask1, 0] += m1
    pointcloud[mask2, 0] += m2
    return pointcloud

#! dxg 重新修改了后缀规则，需要GT的加bbox后缀，不需要的，没有data这一参数的，不加bbox后缀
MAP = {
    'density_dec':density,
    'cutout':cutout,
    'gaussian_noise':gaussian,
    'uniform_noise':uniform,
    'impulse_noise':impulse,
    'scale_bbox':scale, # 需要bbox
    'shear_bbox':shear, # 需要bbox
    'rotation_bbox':rotation, # 需要bbox
    'moving_noise':moving_object,
}
# MAP = {
#     'density_dec_bbox':density,
#     'cutout_bbox':cutout,
#     'gaussian_noise_bbox':gaussian,
#     'uniform_noise_bbox':uniform,
#     'impulse_noise_bbox':impulse,
#     'scale_bbox':scale,
#     'shear_bbox':shear,
#     'rotation_bbox':rotation,
#     'moving_noise_bbox':moving_object,
# }



def pick_bbox(cor, slevel, data, pointcloud):

    xyz = pointcloud
    bboxes = data[0]

    for box in bboxes:
        # Vectorized point-in-box test for all points at once
        shift = xyz[:, :3] - box[:3]
        cos_a = np.cos(box[6])
        sin_a = np.sin(box[6])
        local_x = shift[:, 0] * cos_a + shift[:, 1] * sin_a
        local_y = shift[:, 1] * cos_a - shift[:, 0] * sin_a
        half_dx, half_dy, half_dz = box[3] / 2.0, box[4] / 2.0, box[5] / 2.0

        inside_mask = (np.abs(shift[:, 2]) <= half_dz) & \
                      (np.abs(local_x) <= half_dx) & \
                      (np.abs(local_y) <= half_dy)

        if not inside_mask.any():
            continue

        pcd_2 = xyz[inside_mask]
        pcd_1 = xyz[~inside_mask]

        if 'bbox' in cor:
            pcd_2 = MAP[cor](pcd_2, slevel, box)
        else:
            pcd_2 = MAP[cor](pcd_2, slevel)

        xyz = np.append(pcd_2, pcd_1, axis=0)

    return xyz








