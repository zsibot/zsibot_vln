import numpy as np
import torch
from collections import Counter
import cv2
import open3d as o3d
from omegaconf import DictConfig
import faiss
from .slam_classes import MapObjectList, DetectionList
from .iou import compute_3d_iou_accuracte_batch, compute_iou_batch, mask_subtract_contained
import quaternion as npq  # pip install numpy-quaternion


def points_from_xf_yl_zu(points_prime):
    return torch.stack([-points_prime[..., 1], points_prime[..., 0], points_prime[..., 2]], dim=-1)

def points_to_xf_yl_zu(points): # x right, y forward, z up to x forward, y left z up
    return torch.stack([points[..., 1], -points[..., 0], points[..., 2]], dim=-1)

def quat_numpy_to_torch_wxyz(q_np: np.quaternion, dtype, device) -> torch.Tensor:
    arr = npq.as_float_array(q_np)  # shape (4,), order wxyz
    return torch.tensor(arr, dtype=dtype, device=device)

def rotate_around_z_neg_compass(points: torch.Tensor, compass_rad) -> torch.Tensor:
    """
    绕Z轴旋转角度 -compass_rad（单位：弧度）。
    points: [..., 3]
    compass_rad: 标量（float/np/tensor均可）
    """
    theta = -torch.as_tensor(compass_rad, dtype=points.dtype, device=points.device)
    c, s = torch.cos(theta), torch.sin(theta)

    Rz = torch.stack([
        torch.stack([ c, -s, torch.tensor(0., dtype=points.dtype, device=points.device)], dim=-1),
        torch.stack([ s,  c, torch.tensor(0., dtype=points.dtype, device=points.device)], dim=-1),
        torch.stack([ torch.tensor(0., dtype=points.dtype, device=points.device),
                      torch.tensor(0., dtype=points.dtype, device=points.device),
                      torch.tensor(1., dtype=points.dtype, device=points.device)], dim=-1),
    ], dim=-2)  # [3,3]

    # p' = Rz @ p
    rotated = torch.einsum('ij,...j->...i', Rz, points)
    return rotated


def rotate_points_wxyz(points: torch.Tensor, q_wxyz: torch.Tensor) -> torch.Tensor:
    """
    先用四元数 q=(w,x,y,z) 计算 **内禀XYZ欧拉角 (x,y,z)**，
    再按 **R = Rz(z) @ Ry(y) @ Rx(x)** 构造旋转矩阵，并作用到点云。
    - points: [...,3]  (如 [1,160,120,3])
    - q_wxyz: [4] (torch，单个四元数)
    返回: 与 points 同形状
    """
    if points.shape[-1] != 3:
        raise ValueError("points 最后一维必须为 3")
    if q_wxyz.shape[-1] != 4:
        raise ValueError("q_wxyz 最后一维必须为 4 (w,x,y,z)")

    # ---------- 1) q -> 临时R（只为提取欧拉角） ----------
    q = q_wxyz.to(device=points.device, dtype=points.dtype)
    q = q / torch.clamp(q.norm(p=2), min=1e-12)  # 归一化
    w, x, y, z = q.unbind(-1)
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    r00 = ww + xx - yy - zz
    r01 = 2*(xy - wz)
    r02 = 2*(xz + wy)
    r10 = 2*(xy + wz)
    r11 = ww - xx + yy - zz
    r12 = 2*(yz - wx)
    r20 = 2*(xz - wy)
    r21 = 2*(yz + wx)
    r22 = ww - xx - yy + zz

    R_tmp = torch.stack([
        torch.stack([r00, r01, r02], dim=-1),
        torch.stack([r10, r11, r12], dim=-1),
        torch.stack([r20, r21, r22], dim=-1),
    ], dim=-2)  # [3,3]

    # ---------- 2) 从 R_tmp 提取内禀XYZ欧拉角 (x,y,z) ----------
    # R = Rz(z) @ Ry(y) @ Rx(x)
    # 提取公式：
    #   y = asin( R[0,2] )
    #   x = atan2( -R[1,2], R[2,2] )
    #   z = atan2( -R[0,1], R[0,0] )
    r02, r12, r22 = R_tmp[0,2], R_tmp[1,2], R_tmp[2,2]
    r01, r00       = R_tmp[0,1], R_tmp[0,0]

    y_ang = torch.asin(torch.clamp(r02, -1.0, 1.0))
    x_ang = torch.atan2(-r12, r22)
    z_ang = torch.atan2(-r01, r00)

    # 调试：打印欧拉角（度）
    with torch.no_grad():
        deg = 180.0 / 3.141592653589793
        #print(f"[Euler XYZ deg] x={x_ang.item()*deg:.6f}, y={y_ang.item()*deg:.6f}, z={z_ang.item()*deg:.6f}")

    cx, sx = torch.cos(x_ang), torch.sin(x_ang)
    cy, sy = torch.cos(y_ang), torch.sin(y_ang)
    cz, sz = torch.cos(z_ang), torch.sin(z_ang)

    one  = torch.tensor(1.0, dtype=points.dtype, device=points.device)
    zero = torch.tensor(0.0, dtype=points.dtype, device=points.device)

    Rx = torch.stack([
        torch.stack([ one,  zero, zero], dim=-1),
        torch.stack([ zero,   cx,  -sx], dim=-1),
        torch.stack([ zero,   sx,   cx], dim=-1),
    ], dim=-2)

    Ry = torch.stack([
        torch.stack([  cy, zero,  sy], dim=-1),
        torch.stack([ zero,  one, zero], dim=-1),
        torch.stack([ -sy, zero,  cy], dim=-1),
    ], dim=-2)

    Rz = torch.stack([
        torch.stack([  cz,  -sz, zero], dim=-1),
        torch.stack([  sz,   cz, zero], dim=-1),
        torch.stack([ zero, zero,  one], dim=-1),
    ], dim=-2)

    #R_euler = Rx @ (Ry @ Rz)   # 注意顺序：RxRyRz
    R_euler = Rx @ (Ry @ Rz)
    # 1) 先把点云改到 x前/yleft/z上 坐标
    points_body = points_to_xf_yl_zu(points)            # points 形状 [...,3]
    #show_point_cloud(points_body,fig_id=344)
    # 2) 应用旋转（行向量右乘转置，等价于列向量左乘）
    rotated_body = torch.matmul(points_body, R_euler.T) # R_euler 为 RxRyRz
    #show_point_cloud(rotated_body,fig_id=345)
    # 3) 再映射回原坐标 x右/y前/z上
    #rotated = points_from_xf_yl_zu(rotated_body)
    #rotated = rotated_body
    return rotated_body


def transform_to_local_map(XYZ, sensor_x, sensor_y, sensor_z):

    XYZ[...,0] = XYZ[...,0] + sensor_x
    XYZ[...,1] = XYZ[...,1] + sensor_y
    XYZ[...,2] = XYZ[...,2] + sensor_z

    return XYZ

def rotate_around_x(point_cloud_camera, theta_deg):
    """
    point_cloud_camera: (..., 3) with axes (X, Y, Z)
    θ (theta) in degrees represents the yaw angle (rotation around X-axis)
    """
    # Convert angle from degrees to radians
    theta_rad = torch.deg2rad(torch.tensor(theta_deg, dtype=torch.float32))

    c = torch.cos(theta_rad)
    s = torch.sin(theta_rad)

    # Extract the coordinates
    X = point_cloud_camera[..., 0]
    Y = point_cloud_camera[..., 1]
    Z = point_cloud_camera[..., 2]

    # Rotation formula for yaw (around X-axis)
    Y_new = c * Y - s * Z
    Z_new = s * Y + c * Z

    # Return the rotated point cloud
    return torch.stack((X, Y_new, Z_new), dim=-1)

def filter_objects(cfg, objects: MapObjectList):
    # Remove the object that has very few points or viewed too few times
    objects_to_keep = []
    for obj in objects:
        if len(obj['pcd'].points) >= cfg.obj_min_points and obj['num_detections'] >= cfg.obj_min_detections:
            objects_to_keep.append(obj)
    objects = MapObjectList(objects_to_keep)
    
    return objects


def filter_gobs(
    cfg: DictConfig,
    gobs: dict,
    image: np.ndarray,
    BG_CLASSES = ["wall", "floor", "ceiling"],
):
    # If no detection at all
    if len(gobs['xyxy']) == 0:
        return gobs
    
    # Filter out the objects based on various criteria
    idx_to_keep = []
    for mask_idx in range(len(gobs['xyxy'])):
        local_class_id = gobs['class_id'][mask_idx]
        class_name = gobs['classes'][local_class_id]
        
        # SKip masks that are too small
        if gobs['mask'][mask_idx].sum() < max(cfg.mask_area_threshold, 10):
            continue
        
        # Skip the BG classes
        if cfg.skip_bg and class_name in BG_CLASSES:
            continue
        
        # Skip the non-background boxes that are too large
        if class_name not in BG_CLASSES:
            x1, y1, x2, y2 = gobs['xyxy'][mask_idx]
            bbox_area = (x2 - x1) * (y2 - y1)
            image_area = image.shape[0] * image.shape[1]
            if bbox_area > cfg.max_bbox_area_ratio * image_area:
                continue
            
        # Skip masks with low confidence
        if gobs['confidence'][mask_idx] < cfg.mask_conf_threshold:
            continue
        
        idx_to_keep.append(mask_idx)
    
    for k in gobs.keys():
        if k == "image_rgb":
            continue
        if isinstance(gobs[k], str) or k == "classes": # Captions
            continue
        elif isinstance(gobs[k], list):
            gobs[k] = [gobs[k][i] for i in idx_to_keep]
        elif isinstance(gobs[k], np.ndarray):
            gobs[k] = gobs[k][idx_to_keep]
        else:
            raise NotImplementedError(f"Unhandled type {type(gobs[k])}")
    
    return gobs


def resize_gobs(
    gobs,
    image
):
    n_masks = len(gobs['xyxy'])

    new_mask = []
    
    for mask_idx in range(n_masks):
        # TODO: rewrite using interpolation/resize in numpy or torch rather than cv2
        mask = gobs['mask'][mask_idx]
        if mask.shape != image.shape[:2]:
            # Rescale the xyxy coordinates to the image shape
            x1, y1, x2, y2 = gobs['xyxy'][mask_idx]
            x1 = round(x1 * image.shape[1] / mask.shape[1])
            y1 = round(y1 * image.shape[0] / mask.shape[0])
            x2 = round(x2 * image.shape[1] / mask.shape[1])
            y2 = round(y2 * image.shape[0] / mask.shape[0])
            gobs['xyxy'][mask_idx] = [x1, y1, x2, y2]
            
            # Reshape the mask to the image shape
            mask = cv2.resize(mask.astype(np.uint8), image.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            mask = mask.astype(bool)
            new_mask.append(mask)

    if len(new_mask) > 0:
        gobs['mask'] = np.asarray(new_mask)
        
    return gobs


def to_scalar(d):
    '''
    Convert the d to a scalar
    '''
    if isinstance(d, float):
        return d
    
    elif "numpy" in str(type(d)):
        assert d.size == 1
        return d.item()
    
    elif isinstance(d, torch.Tensor):
        assert d.numel() == 1
        return d.item()
    
    else:
        raise TypeError(f"Invalid type for conversion: {type(d)}")


def from_intrinsics_matrix(K: torch.Tensor):
    '''
    Get fx, fy, cx, cy from the intrinsics matrix
    
    return 4 scalars
    '''
    fx = to_scalar(K[0, 0])
    fy = to_scalar(K[1, 1])
    cx = to_scalar(K[0, 2])
    cy = to_scalar(K[1, 2])
    return fx, fy, cx, cy



def create_object_pcd(args_camera_theta, args_camera_x, args_camera_z, pose_rotation, depth_array, mask, cam_K, image, obj_color=None, is_navigation=False) -> o3d.geometry.PointCloud:
    if is_navigation:
        f, cx, cz = cam_K.f, cam_K.xc, cam_K.zc
    else:
        fx, fy, cx, cy = from_intrinsics_matrix(cam_K)
    
    # Also remove points with invalid depth values
    mask = np.logical_and(mask, depth_array > 0)
    if is_navigation:
        mask = np.logical_and(mask, depth_array < 60)

    if mask.sum() == 0:
        pcd = o3d.geometry.PointCloud()
        return pcd
        
    height, width = depth_array.shape
    x = np.arange(0, width, 1.0)
    y = np.arange(0, height, 1.0)
    u, v = np.meshgrid(x, y)
    
    # Apply the mask, and unprojection is done only on the valid points
    masked_depth = depth_array[mask] # (N, )
    u = u[mask] # (N, )
    v = v[mask] # (N, )

    # Convert to 3D coordinates
    if not is_navigation:
        x = (u - cx) * masked_depth / fx
        y = (v - cy) * masked_depth / fy
        z = masked_depth
        #print("not navigation")
    else:
        x = (u - cx) * masked_depth / f
        y = masked_depth
        z = (v - cz) * masked_depth / f
        #print("is_navigation")

    # Stack x, y, z coordinates into a 3D point cloud
    points = np.stack((x, y, z), axis=-1)
    #print(points.shape)
    points = points.reshape(-1, 3)
    #print(points.shape)

    points = torch.from_numpy(points.astype(np.float32))  # -> Tensor

    q_wxyz = quat_numpy_to_torch_wxyz(
            pose_rotation,
            dtype  = points.dtype,
            device = points.device
        )

    #now
    #points      = rotate_around_x(points, args_camera_theta)
    #points      = transform_to_local_map(points, 0, args_camera_x, args_camera_z)
    #points      = rotate_points_wxyz(points, q_wxyz)
    #points      = transform_to_local_map(points, 0,0, agent_heights*100)
    #points      = rotate_around_z_neg_compass(points, obs['compass'][0])
    #points      = points_from_xf_yl_zu(points)

    points = points.cpu().numpy()
     #

    # Perturb the points a bit to avoid colinearity
    points += np.random.normal(0, 4e-3, points.shape)

    if obj_color is None: # color using RGB
        # # Apply mask to image
        colors = image[mask] / 255.0
    else: # color using group ID
        # Use the assigned obj_color for all points
        colors = np.full(points.shape, obj_color)
    
    if points.shape[0] == 0:
        import pdb; pdb.set_trace()

    # Create an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd


def gobs_to_detection_list(
    args_camera_theta,
    args_camera_x,
    args_camera_z,
    pose_rotation,
    cfg, 
    image, 
    depth_array,
    cam_K, 
    idx, 
    gobs, 
    trans_pose = None,
    class_names = None,
    BG_CLASSES = ["wall", "floor", "ceiling"],
    color_path = None,
    is_navigation = False,
):
    '''
    Return a DetectionList object from the gobs
    All object are still in the camera frame. 
    '''
    fg_detection_list = DetectionList()
    bg_detection_list = DetectionList()
    
    gobs = resize_gobs(gobs, image)
    gobs = filter_gobs(cfg, gobs, image, BG_CLASSES)
    
    if len(gobs['xyxy']) == 0:
        return fg_detection_list, bg_detection_list
    
    # Compute the containing relationship among all detections and subtract fg from bg objects
    xyxy = gobs['xyxy']
    mask = gobs['mask']
    gobs['mask'] = mask_subtract_contained(xyxy, mask)
    
    n_masks = len(gobs['xyxy'])
    for mask_idx in range(n_masks):
        local_class_id = gobs['class_id'][mask_idx]
        mask = gobs['mask'][mask_idx]
        class_name = gobs['classes'][local_class_id]
        global_class_id = -1 if class_names is None else class_names.index(class_name)
        
        # make the pcd and color it
        camera_object_pcd = create_object_pcd(
            args_camera_theta,
            args_camera_x,
            args_camera_z,
            pose_rotation,
            depth_array,
            mask,
            cam_K,
            image,
            obj_color = None,
            is_navigation = is_navigation,
        )
        
        # It at least contains 5 points
        if len(camera_object_pcd.points) < max(cfg.min_points_threshold, 5): 
            continue
        
        if trans_pose is not None:
            global_object_pcd = camera_object_pcd.transform(trans_pose)
        else:
            global_object_pcd = camera_object_pcd
        
        # get largest cluster, filter out noise 
        global_object_pcd = process_pcd(global_object_pcd, cfg)
        
        pcd_bbox = get_bounding_box(cfg, global_object_pcd)
        pcd_bbox.color = [0,1,0]
        
        if pcd_bbox.volume() < 1e-6:
            continue
        
        # Treat the detection in the same way as a 3D object
        # Store information that is enough to recover the detection
        detected_object = {
            'image_idx' : [idx],                             # idx of the image
            'mask_idx' : [mask_idx],                         # idx of the mask/detection
            'color_path' : [color_path],                     # path to the RGB image
            'class_name' : [class_name],                         # global class id for this detection
            'class_id' : [global_class_id],                         # global class id for this detection
            'num_detections' : 1,                            # number of detections in this object
            'mask': [mask],
            'xyxy': [gobs['xyxy'][mask_idx]],
            'conf': [gobs['confidence'][mask_idx]],
            'n_points': [len(global_object_pcd.points)],
            'pixel_area': [mask.sum()],
            'contain_number': [None],                          # This will be computed later
            "inst_color": np.random.rand(3),                 # A random color used for this segment instance
            'is_background': class_name in BG_CLASSES,
            
            # These are for the entire 3D object
            'pcd': global_object_pcd,
            'bbox': pcd_bbox,
        }
        
        if class_name in BG_CLASSES:
            bg_detection_list.append(detected_object)
        else:
            fg_detection_list.append(detected_object)
    
    return fg_detection_list, bg_detection_list


def compute_overlap_matrix_2set(cfg, objects_map: MapObjectList, objects_new: DetectionList) -> np.ndarray:
    '''
    compute pairwise overlapping between two set of objects in terms of point nearest neighbor. 
    objects_map is the existing objects in the map, objects_new is the new objects to be added to the map
    Suppose len(objects_map) = m, len(objects_new) = n
    Then we want to construct a matrix of size m x n, where the (i, j) entry is the ratio of points 
    in point cloud i that are within a distance threshold of any point in point cloud j.
    '''
    m = len(objects_map)
    n = len(objects_new)
    overlap_matrix = np.zeros((m, n))
    
    # Convert the point clouds into numpy arrays and then into FAISS indices for efficient search
    points_map = [np.asarray(obj['pcd'].points, dtype=np.float32) for obj in objects_map] # m arrays
    for i, points in enumerate(points_map):
        num_points = len(points)
        if num_points > cfg.max_num_points:
            choice = np.random.choice(range(num_points), cfg.max_num_points)
            points_map[i] = points[choice]
    indices = [faiss.IndexFlatL2(arr.shape[1]) for arr in points_map] # m indices
    
    # Add the points from the numpy arrays to the corresponding FAISS indices
    for index, arr in zip(indices, points_map):
        index.add(arr)
        
    points_new = [np.asarray(obj['pcd'].points, dtype=np.float32) for obj in objects_new] # n arrays
    for i, points in enumerate(points_new):
        num_points = len(points)
        if num_points > cfg.max_num_points:
            choice = np.random.choice(range(num_points), cfg.max_num_points)
            points_new[i] = points[choice]
        
    bbox_map = objects_map.get_stacked_values_torch('bbox')
    bbox_new = objects_new.get_stacked_values_torch('bbox')
    try:
        iou = compute_3d_iou_accuracte_batch(bbox_map, bbox_new) # (m, n)
    except ValueError:
        print("Met `Plane vertices are not coplanar` error, use axis aligned bounding box instead")
        bbox_map = []
        bbox_new = []
        for pcd in objects_map.get_values('pcd'):
            bbox_map.append(np.asarray(
                pcd.get_axis_aligned_bounding_box().get_box_points()))
        for pcd in objects_new.get_values('pcd'):
            bbox_new.append(np.asarray(
                pcd.get_axis_aligned_bounding_box().get_box_points()))
        bbox_map = torch.from_numpy(np.stack(bbox_map))
        bbox_new = torch.from_numpy(np.stack(bbox_new))
        
        iou = compute_iou_batch(bbox_map, bbox_new) # (m, n)
            

    # Compute the pairwise overlaps
    for i in range(m):
        for j in range(n):
            if iou[i,j] < 1e-6:
                continue
            
            D, I = indices[i].search(points_new[j], 1) # search new object j in map object i

            overlap = (D < cfg.downsample_voxel_size ** 2).sum() # D is the squared distance

            # Calculate the ratio of points within the threshold
            overlap_matrix[i, j] = overlap / len(points_new[j])

    return overlap_matrix


def pcd_denoise_dbscan(pcd: o3d.geometry.PointCloud, eps=0.02, min_points=10) -> o3d.geometry.PointCloud:
    ### Remove noise via clustering
    pcd_clusters = pcd.cluster_dbscan(
        eps=eps,
        min_points=min_points,
    )
    
    # Convert to numpy arrays
    obj_points = np.asarray(pcd.points)
    obj_colors = np.asarray(pcd.colors)
    pcd_clusters = np.array(pcd_clusters)

    # Count all labels in the cluster
    counter = Counter(pcd_clusters)

    # Remove the noise label
    if counter and (-1 in counter):
        del counter[-1]

    if counter:
        # Find the label of the largest cluster
        most_common_label, _ = counter.most_common(1)[0]
        
        # Create mask for points in the largest cluster
        largest_mask = pcd_clusters == most_common_label

        # Apply mask
        largest_cluster_points = obj_points[largest_mask]
        largest_cluster_colors = obj_colors[largest_mask]
        
        # If the largest cluster is too small, return the original point cloud
        if len(largest_cluster_points) < 5:
            return pcd

        # Create a new PointCloud object
        largest_cluster_pcd = o3d.geometry.PointCloud()
        largest_cluster_pcd.points = o3d.utility.Vector3dVector(largest_cluster_points)
        largest_cluster_pcd.colors = o3d.utility.Vector3dVector(largest_cluster_colors)
        
        pcd = largest_cluster_pcd
        
    return pcd


def process_pcd(pcd, cfg, run_dbscan=True):
    
    pcd = pcd.voxel_down_sample(voxel_size=cfg.downsample_voxel_size)
        
    if cfg.dbscan_remove_noise and run_dbscan:
        pcd = pcd_denoise_dbscan(
            pcd, 
            eps=cfg.dbscan_eps, 
            min_points=cfg.dbscan_min_points
        )
        
    return pcd

def get_bounding_box(cfg, pcd):
    if ("accurate" in cfg.spatial_sim_type or "overlap" in cfg.spatial_sim_type) and len(pcd.points) >= 4:
        try:
            return pcd.get_oriented_bounding_box(robust=True)
        except RuntimeError as e:
            print(f"Met {e}, use axis aligned bounding box instead")
            return pcd.get_axis_aligned_bounding_box()
    else:
        return pcd.get_axis_aligned_bounding_box()


def merge_obj2_into_obj1(cfg, obj1, obj2, run_dbscan=True):
    '''
    Merge the new object to the old object
    This operation is done in-place
    '''
    n_obj1_det = obj1['num_detections']
    n_obj2_det = obj2['num_detections']
    
    for k in obj1.keys():
        if k in ['caption']:
            # Here we need to merge two dictionaries and adjust the key of the second one
            for k2, v2 in obj2['caption'].items():
                obj1['caption'][k2 + n_obj1_det] = v2
        elif k not in ['pcd', 'bbox', 'clip_ft', "text_ft", "score", "captions", "reason", "id", "node"]:
            if isinstance(obj1[k], list) or isinstance(obj1[k], int):
                obj1[k] += obj2[k]
            elif k == "inst_color":
                obj1[k] = obj1[k] # Keep the initial instance color
            else:
                # TODO: handle other types if needed in the future
                raise NotImplementedError
        else: # pcd, bbox, clip_ft, text_ft are handled below
            continue

    # merge pcd and bbox
    obj1['pcd'] += obj2['pcd']
    obj1['pcd'] = process_pcd(obj1['pcd'], cfg, run_dbscan=run_dbscan)
    obj1['bbox'] = get_bounding_box(cfg, obj1['pcd'])
    obj1['bbox'].color = [0,1,0]
    
    return obj1
