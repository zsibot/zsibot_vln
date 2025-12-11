import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from agents.zeroshot.unigoal.utils.fmm.depth_utils import (
    transform_camera_view_t_Y,
    get_point_cloud_from_z_t,
    transform_pose_t,
    splat_feat_nd
)
from agents.zeroshot.unigoal.utils.model import get_grid, ChannelPool
from agents.zeroshot.unigoal.utils.camera import get_camera_matrix
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import quaternion as npq  # pip install numpy-quaternion
plt.ion()

import torch
import math

import numpy as np
from typing import Optional, Tuple
from PIL import Image
#from agents.utils.disk_viz           import DiskViz
from utils.disk_viz           import DiskViz
from matplotlib.figure               import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib import colors as mcolors


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

def rotate_around_y(point_cloud_camera, theta_deg):
    """
    point_cloud_camera: (..., 3) with axes (X, Y, Z)
    θ (theta) in degrees represents the pitch angle (rotation around Y-axis)
    """
    # Convert angle from degrees to radians
    theta_rad = torch.deg2rad(torch.tensor(theta_deg, dtype=torch.float32))

    c = torch.cos(theta_rad)
    s = torch.sin(theta_rad)

    # Extract the coordinates
    X = point_cloud_camera[..., 0]
    Y = point_cloud_camera[..., 1]
    Z = point_cloud_camera[..., 2]

    # Rotation formula for pitch (around Y-axis)
    X_new = c * X + s * Z
    Z_new = -s * X + c * Z

    # Return the rotated point cloud
    return torch.stack((X_new, Y, Z_new), dim=-1)


def rotate_around_z(point_cloud_camera, theta_deg):
    """
    point_cloud_camera: (..., 3) with axes (X, Y, Z)
    θ (theta) in degrees represents the rotation around Z-axis
    """
    # Convert angle from degrees to radians
    theta_rad = torch.deg2rad(torch.tensor(theta_deg, dtype=torch.float32))

    c = torch.cos(theta_rad)
    s = torch.sin(theta_rad)

    # Extract the coordinates
    X = point_cloud_camera[..., 0]
    Y = point_cloud_camera[..., 1]
    Z = point_cloud_camera[..., 2]

    # Rotation formula for rotating around the Z-axis
    X_new = c * X - s * Y
    Y_new = s * X + c * Y

    # Return the rotated point cloud
    return torch.stack((X_new, Y_new, Z), dim=-1)


def quat_numpy_to_torch_wxyz(q_np: np.quaternion, dtype, device) -> torch.Tensor:
    arr = npq.as_float_array(q_np)  # shape (4,), order wxyz
    return torch.tensor(arr, dtype=dtype, device=device)

def points_to_xf_yl_zu(points): # x right, y forward, z up to x forward, y left z up
    return torch.stack([points[..., 1], -points[..., 0], points[..., 2]], dim=-1)

# 逆映射：[x_forward, y_left, z_up] -> [x_right, y_forward, z_up]
# 由上式可得：x = -y',  y = x',  z = z'
def points_from_xf_yl_zu(points_prime):
    return torch.stack([-points_prime[..., 1], points_prime[..., 0], points_prime[..., 2]], dim=-1)

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

def show_point_cloud(point_cloud,
                     fig_id=1,
                     viz=None, step=None, stream="cloud_dist",
                     color="distance",           # "distance" 或 "z"
                     cmap="viridis",
                     scale=0.01,                 # cm→m 用 0.01；若已是米用 1.0
                     max_points=None,
                     vmin=0.0, vmax=None,
                     elev=None, azim=None,       # ✅ 显式传入就固定视角；否则两边都用默认
                     size=(640, 480)):           # 离屏图尺寸
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

    # ------- 读取与规整 -------
    if isinstance(point_cloud, torch.Tensor):
        pc_np = point_cloud.detach().cpu().numpy()
    else:
        pc_np = np.asarray(point_cloud)
    pc_np = np.squeeze(pc_np)
    if pc_np.ndim == 4:  # B,H,W,3
        pc_np = pc_np[0]
    if pc_np.ndim != 3 or pc_np.shape[2] != 3:
        raise ValueError(f"期望点云形状 (H, W, 3)，拿到 {pc_np.shape}")

    points = pc_np.reshape(-1, 3).astype(np.float32, copy=False)
    valid_mask = np.isfinite(points).all(axis=1) & (points[:, 2] != 0)
    points = points[valid_mask]
    if points.size == 0:
        if viz is None:
            print("没有有效的点云数据")
        return

    # 单位缩放
    if scale != 1.0:
        points = points * float(scale)
    # 可选下采样
    if (max_points is not None) and (points.shape[0] > max_points):
        sel = np.random.choice(points.shape[0], max_points, replace=False)
        points = points[sel]

    # 着色标量
    if color == "distance":
        scalars = np.linalg.norm(points, axis=1)
    elif color == "z":
        scalars = points[:, 2]
    else:
        raise ValueError("color 只能是 'distance' 或 'z'")
    if vmax is None:
        vmax = float(np.max(scalars)) if scalars.size else 1.0

    # 统一：等比例包围盒
    def _set_equal_limits(ax, pts):
        max_range = np.array([
            pts[:, 0].max() - pts[:, 0].min(),
            pts[:, 1].max() - pts[:, 1].min(),
            pts[:, 2].max() - pts[:, 2].min()
        ]).max() / 2.0
        mid_x = (pts[:, 0].max() + pts[:, 0].min()) * 0.5
        mid_y = (pts[:, 1].max() + pts[:, 1].min()) * 0.5
        mid_z = (pts[:, 2].max() + pts[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    # ------- 分支A：交互窗口（与你原来一致） -------
    if viz is None or step is None:
        plt.figure(fig_id); plt.clf()
        ax = plt.axes(projection='3d')
        if elev is not None and azim is not None:  # ✅ 仅当显式传入时设置视角
            ax.view_init(elev=elev, azim=azim)
        scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                            c=scalars, cmap=cmap, norm=norm,
                            s=4, alpha=1.0, marker='.', linewidths=0, edgecolors='none')

        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title('Point Cloud')
        plt.colorbar(scatter, ax=ax, label=('Distance' if color == 'distance' else 'Z value'))
        _set_equal_limits(ax, points)
        plt.tight_layout(); plt.show(block=False); plt.pause(0.001)
        return

    # ------- 分支B：离屏渲染 → Dashboard -------
    W, H = size
    fig = Figure(figsize=(W/100.0, H/100.0), dpi=100)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111, projection='3d')
    if elev is not None and azim is not None:  # ✅ 与交互分支保持一致
        ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
            c=scalars, cmap=cmap, norm=norm,
            s=4, alpha=1.0, marker='.', linewidths=0, edgecolors='none')

    _set_equal_limits(ax, points)
    fig.tight_layout(pad=0)
    canvas.draw()
    buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    img = buf.reshape(H, W, 3)
    viz.save(stream, img, int(step))
    return img

class Mapping(nn.Module):
    def __init__(self, args):
        super(Mapping, self).__init__()

        self.args=args
        self.device = args.device
        self.screen_h = args.frame_height
        self.screen_w = args.frame_width
        self.resolution = args.map_resolution
        self.z_resolution = args.map_resolution
        self.map_size_cm = args.map_size_cm // args.global_downscaling
        self.n_channels = 3
        self.vision_range = args.vision_range
        self.dropout = 0.5
        self.fov = args.hfov
        self.du_scale = args.du_scale
        self.cat_pred_threshold = args.cat_pred_threshold
        self.exp_pred_threshold = args.exp_pred_threshold
        self.map_pred_threshold = args.map_pred_threshold
        self.num_sem_categories = args.num_sem_categories

        self.max_height = int(360 / self.z_resolution)
        self.min_height = int(-80 / self.z_resolution)
        self.shift_loc = [self.vision_range *self.resolution // 2, 0, np.pi / 2.0]
        self.camera_matrix = get_camera_matrix(self.screen_w, self.screen_h, self.fov)

        self.vfov = np.arctan((self.screen_h/2.) / self.camera_matrix.f)
        self.pool = ChannelPool(1)
        vr = self.vision_range
        self.init_grid = torch.zeros( args.num_processes, 1 + self.num_sem_categories, vr, vr,self.max_height - self.min_height).float().to(self.device)
        self.feat = torch.ones(args.num_processes, 1 + self.num_sem_categories, self.screen_h // self.du_scale * self.screen_w // self.du_scale).float().to(self.device)

    def forward(self, obs,rgbd, pose_obs, maps_last, poses_last, agent_heights,viz,step):

        bs, c, h, w = rgbd.size()
        depth = rgbd[:, 3, :, :] # depth in cm

        rgb_slice = rgbd[:, 0:3, :, :].to(torch.uint8)

        if self.args.show_rgb:
            self.rgb_bev = rgb_slice
        if self.args.show_depth:
            self.depth_bev = depth

        point_cloud_camera = get_point_cloud_from_z_t(depth, self.camera_matrix, self.device, scale=self.du_scale)

        if self.args.show_point_cloud:
            point_cloud_filtered = point_cloud_camera.clone()

            mask = point_cloud_filtered[0,:,:, 1] >= 5000
            point_cloud_filtered[0,mask, :] = 0

            show_point_cloud(point_cloud_filtered, viz=viz, step=step, stream="cloud points (color for dist)", color="distance", scale=0.01, vmax=10.0)

        q_wxyz = quat_numpy_to_torch_wxyz(
            obs['pose']['rotation'],
            dtype=point_cloud_camera.dtype,
            device=point_cloud_camera.device
        )

        point_cloud                              = point_cloud_camera

        point_cloud                              = rotate_around_x(point_cloud, self.args.camera_theta)

        point_cloud                              = transform_to_local_map(point_cloud, 0, self.args.camera_x, self.args.camera_z) #

        point_cloud                              = rotate_points_wxyz(point_cloud, q_wxyz)

        point_cloud                              = transform_to_local_map(point_cloud, 0,0, agent_heights*100)

        point_cloud                              = rotate_around_z_neg_compass(point_cloud, obs['compass'][0])

        point_cloud                              = points_from_xf_yl_zu(point_cloud)

        agent_view_centered_t                    = transform_pose_t(point_cloud, self.shift_loc, self.device)

        max_h = self.max_height
        min_h = self.min_height
        xy_resolution = self.resolution
        z_resolution = self.z_resolution
        vision_range = self.vision_range

        XYZ_cm_std = agent_view_centered_t.float()
        XYZ_cm_std[..., :2] = (XYZ_cm_std[..., :2] / xy_resolution)
        XYZ_cm_std[..., :2] = (XYZ_cm_std[..., :2] -
                               vision_range // 2.) / vision_range * 2.
        XYZ_cm_std[..., 2] = XYZ_cm_std[..., 2] / z_resolution
        XYZ_cm_std[..., 2] = (XYZ_cm_std[..., 2] -
                              (max_h + min_h) // 2.) / (max_h - min_h) * 2.

        self.feat[:, 1:, :] = nn.AvgPool2d(self.du_scale)(
            rgbd[:, 4:, :, :]
        ).view(bs, c - 4, h // self.du_scale * w // self.du_scale)

        XYZ_cm_std = XYZ_cm_std.permute(0, 3, 1, 2)
        XYZ_cm_std = XYZ_cm_std.view(XYZ_cm_std.shape[0],
                                     XYZ_cm_std.shape[1],
                                     XYZ_cm_std.shape[2] * XYZ_cm_std.shape[3])

        voxels = splat_feat_nd(
            self.init_grid * 0., self.feat, XYZ_cm_std).transpose(2, 3) #torch.Size([1, 17, 100, 100, 88])

        min_z = 25 #20
        max_z = 30 #int((self.agent_height + 1) / z_resolution - min_h)

        #agent_height_proj = voxels[..., min_z:max_z].sum(4)
        #all_height_proj = voxels.sum(4)
        agent_height_proj = voxels[..., min_z:-30].sum(4)

        all_height_proj = voxels[..., :-30].sum(4)

        fp_map_pred = agent_height_proj[:, 0:1, :, :]
        fp_exp_pred = all_height_proj[:, 0:1, :, :]

        fp_map_pred = fp_map_pred / self.map_pred_threshold
        fp_exp_pred = fp_exp_pred / self.exp_pred_threshold
        fp_map_pred = torch.clamp(fp_map_pred, min=0.0, max=1.0)
        fp_exp_pred = torch.clamp(fp_exp_pred, min=0.0, max=1.0)

        pose_pred = poses_last
        agent_view = torch.zeros(bs, c,
                                 self.map_size_cm // self.resolution,
                                 self.map_size_cm // self.resolution
                                 ).to(self.device)

        x1 = self.map_size_cm // (self.resolution * 2) - self.vision_range // 2
        x2 = x1 + self.vision_range
        y1 = self.map_size_cm // (self.resolution * 2)
        y2 = y1 + self.vision_range
        agent_view[:, 0:1, y1:y2, x1:x2] = fp_map_pred
        agent_view[:, 1:2, y1:y2, x1:x2] = fp_exp_pred
        agent_view[:, 4:, y1:y2, x1:x2] = torch.clamp(
            agent_height_proj[:, 1:, :, :] / self.cat_pred_threshold,
            min=0.0, max=1.0)

        corrected_pose = pose_obs

        def get_new_pose_batch(pose, rel_pose_change):

            pose[:, 1] += rel_pose_change[:, 0] * \
                torch.sin(pose[:, 2] / 57.29577951308232) \
                + rel_pose_change[:, 1] * \
                torch.cos(pose[:, 2] / 57.29577951308232)
            pose[:, 0] += rel_pose_change[:, 0] * \
                torch.cos(pose[:, 2] / 57.29577951308232) \
                - rel_pose_change[:, 1] * \
                torch.sin(pose[:, 2] / 57.29577951308232)
            pose[:, 2] += rel_pose_change[:, 2] * 57.29577951308232

            pose[:, 2] = torch.fmod(pose[:, 2] - 180.0, 360.0) + 180.0
            pose[:, 2] = torch.fmod(pose[:, 2] + 180.0, 360.0) - 180.0

            return pose

        current_poses = get_new_pose_batch(poses_last, corrected_pose)

        st_pose = current_poses.clone().detach()

        st_pose[:, :2] = - (st_pose[:, :2]
                            * 100.0 / self.resolution
                            - self.map_size_cm // (self.resolution * 2)) /\
            (self.map_size_cm // (self.resolution * 2))
        st_pose[:, 2] = 90. - (st_pose[:, 2])

        rot_mat, trans_mat = get_grid(st_pose, agent_view.size(),
                                      self.device)
        rotated = F.grid_sample(agent_view, rot_mat, align_corners=True)
        translated = F.grid_sample(rotated, trans_mat, align_corners=True)

        maps2 = torch.cat((maps_last.unsqueeze(1), translated.unsqueeze(1)), 1)

        map_pred, _ = torch.max(maps2, 1)

        return fp_map_pred, map_pred, pose_pred, current_poses



class BEV_Map():
    def __init__(self, args):
        self.args = args

        self.num_scenes = self.args.num_processes
        nc = self.args.num_sem_categories + 4  # num channels
        self.device = self.args.device

        self.map_size = self.args.map_size
        self.global_width = self.args.global_width
        self.global_height = self.args.global_height
        self.local_width = self.args.local_width
        self.local_height = self.args.local_height

        self.mapping_module = Mapping(self.args).to(self.device)
        self.mapping_module.eval()


        # Initializing full and local map
        self.full_map = torch.zeros(self.num_scenes, nc, self.global_width, self.global_height).float().to(self.device)
        self.local_map = torch.zeros(self.num_scenes, nc, self.local_width,
                                self.local_height).float().to(self.device)

        # Initial full and local pose
        self.full_pose = torch.zeros(self.num_scenes, 3).float().to(self.device)
        self.local_pose = torch.zeros(self.num_scenes, 3).float().to(self.device)

        # Origin of local map
        self.origins = np.zeros((self.num_scenes, 3))

        # Local Map Boundaries
        self.local_map_boundary = np.zeros((self.num_scenes, 4)).astype(int)

        # Planner pose inputs has 7 dimensions
        # 1-3 store continuous global agent location
        # 4-7 store local map boundaries
        self.planner_pose_inputs = np.zeros((self.num_scenes, 7))

    def mapping(self, obs,rgbd, infos,viz,step):

        #update_traj(self.full_pose)

        rgbd = torch.tensor(rgbd, dtype=torch.float32, device=self.device)
        if rgbd.ndim == 3:
            rgbd = rgbd.unsqueeze(0)
        poses = torch.from_numpy(np.asarray(
            [infos['sensor_pose'] for env_idx
             in range(self.num_scenes)])
        ).float().to(self.device) # this variable is relative poses

        body_center_heights = torch.from_numpy(np.asarray(
            [infos['body_center_height'] for env_idx in range(self.num_scenes)])
        ).float().to(self.device)


        _, self.local_map, _, self.local_pose = \
            self.mapping_module(obs,rgbd, poses, self.local_map, self.local_pose, body_center_heights,viz,step)

        #print("self.local_pose",self.local_pose) #automatically moved to the center of local map

        local_pose = self.local_pose.cpu().numpy()
        #print("local_pose,self.origins",local_pose,self.origins)

        self.planner_pose_inputs[:, :3] = local_pose + self.origins # in global coordinate
        #print('self.planner_pose_inputs[:, :3]=',self.planner_pose_inputs[:, :3])

        self.local_map[:, 2, :, :].fill_(0.)  # Resetting current location channel
        for e in range(self.num_scenes):
            # r, c = locs[e, 1], locs[e, 0]
            self.local_row = int(local_pose[e, 1] * 100.0 / self.args.map_resolution)
            self.local_col = int(local_pose[e, 0] * 100.0 / self.args.map_resolution)
            self.local_map[e, 2:4, self.local_row - 2:self.local_row + 3, self.local_col - 2:self.local_col + 3] = 1.

        for e in range(self.num_scenes):
            # agent cell
            self.local_row = int(local_pose[e, 1] * 100.0 / self.args.map_resolution)
            self.local_col = int(local_pose[e, 0] * 100.0 / self.args.map_resolution)

            H, W = self.local_map.shape[-2], self.local_map.shape[-1]
            self.local_row = max(0, min(H - 1, self.local_row))
            self.local_col = max(0, min(W - 1, self.local_col))

            # mark current location (channels 2–4)
            r0, r1 = max(0, self.local_row - 2), min(H, self.local_row + 3)
            c0, c1 = max(0, self.local_col - 2), min(W, self.local_col + 3)
            self.local_map[e, 2:4, r0:r1, c0:c1] = 1.0

            # mark explored underfoot (channel 1)
            r = 5
            y0, y1 = max(0, self.local_row - r), min(H, self.local_row + r + 1)
            x0, x1 = max(0, self.local_col - r), min(W, self.local_col + r + 1)
            yy, xx = torch.meshgrid(
                torch.arange(y0, y1, device=self.local_map.device),
                torch.arange(x0, x1, device=self.local_map.device),
                indexing="ij",
            )
            disk = (yy - self.local_row)**2 + (xx - self.local_col)**2 <= r*r
            self.local_map[e, 1, y0:y1, x0:x1][disk] = 1.0

            # optional small wedge ahead
            theta = float(self.local_pose[e, 2].item()) * math.pi / 180.0
            ahead, half_w = 9, 5
            for t in range(1, ahead + 1):
                ry = int(round(self.local_row + t * math.sin(theta)))
                rx = int(round(self.local_col + t * math.cos(theta)))
                if 0 <= ry < H and 0 <= rx < W:
                    yy0, yy1 = max(0, ry - half_w), min(H, ry + half_w + 1)
                    xx0, xx1 = max(0, rx - half_w), min(W, rx + half_w + 1)
                    self.local_map[e, 1, yy0:yy1, xx0:xx1] = 1.0




    def move_local_map(self, env_idx=0):
        self.full_map[env_idx, :, self.local_map_boundary[env_idx, 0]:self.local_map_boundary[env_idx, 1], self.local_map_boundary[env_idx, 2]:self.local_map_boundary[env_idx, 3]] = \
            self.local_map[env_idx]
        self.full_pose[env_idx] = self.local_pose[env_idx] + \
            torch.from_numpy(self.origins[env_idx]).to(self.device).float()

        locs = self.full_pose[env_idx].cpu().numpy()
        r, c = locs[1], locs[0]
        loc_r, loc_c = [int(r * 100.0 / self.args.map_resolution),
                        int(c * 100.0 / self.args.map_resolution)]

        self.local_map_boundary[env_idx] = self.get_local_map_boundaries((loc_r, loc_c))

        self.planner_pose_inputs[env_idx, 3:] = self.local_map_boundary[env_idx]
        self.origins[env_idx] = [self.local_map_boundary[env_idx][2] * self.args.map_resolution / 100.0,
                        self.local_map_boundary[env_idx][0] * self.args.map_resolution / 100.0, 0.]

        self.local_map[env_idx] = self.full_map[env_idx, :,
                                self.local_map_boundary[env_idx, 0]:self.local_map_boundary[env_idx, 1],
                                self.local_map_boundary[env_idx, 2]:self.local_map_boundary[env_idx, 3]]
        self.local_pose[env_idx] = self.full_pose[env_idx] - \
            torch.from_numpy(self.origins[env_idx]).to(self.device).float()

    def get_local_map_boundaries(self, agent_loc):
        loc_r, loc_c = agent_loc

        if self.args.global_downscaling > 1:
            gx1, gy1 = loc_r - self.local_width // 2, loc_c - self.local_height // 2
            gx2, gy2 = gx1 + self.local_width, gy1 + self.local_height
            if gx1 < 0:
                gx1, gx2 = 0, self.local_width
            if gx2 > self.global_width:
                gx1, gx2 = self.global_width - self.local_width, self.global_width

            if gy1 < 0:
                gy1, gy2 = 0, self.local_height
            if gy2 > self.global_height:
                gy1, gy2 = self.global_height - self.local_height, self.global_height
        else:
            gx1, gx2, gy1, gy2 = 0, self.global_width, 0, self.global_height

        return [gx1, gx2, gy1, gy2]

    def init_map_and_pose(self):
        self.full_map.fill_(0.)
        self.full_pose.fill_(0.)
        self.full_pose[:, :2] = self.args.map_size_cm / 100.0 / 2.0

        locs = self.full_pose.cpu().numpy()
        self.planner_pose_inputs[:, :3] = locs
        for e in range(self.num_scenes):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [int(r * 100.0 / self.args.map_resolution),
                            int(c * 100.0 / self.args.map_resolution)]

            self.full_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

            self.local_map_boundary[e] = self.get_local_map_boundaries((loc_r, loc_c))

            self.planner_pose_inputs[e, 3:] = self.local_map_boundary[e]
            self.origins[e] = [self.local_map_boundary[e][2] * self.args.map_resolution / 100.0,
                          self.local_map_boundary[e][0] * self.args.map_resolution / 100.0, 0.]

        for e in range(self.num_scenes):
            self.local_map[e] = self.full_map[e, :,
                                    self.local_map_boundary[e, 0]:self.local_map_boundary[e, 1],
                                    self.local_map_boundary[e, 2]:self.local_map_boundary[e, 3]]
            self.local_pose[e] = self.full_pose[e] - \
                torch.from_numpy(self.origins[e]).to(self.device).float()

    def init_map_and_pose_for_env(self, env_idx=0):
        self.full_map[env_idx].fill_(0.)
        self.full_pose[env_idx].fill_(0.)
        self.full_pose[env_idx, :2] = self.args.map_size_cm / 100.0 / 2.0

        locs = self.full_pose[env_idx].cpu().numpy()
        self.planner_pose_inputs[env_idx, :3] = locs
        r, c = locs[1], locs[0]
        loc_r, loc_c = [int(r * 100.0 / self.args.map_resolution),
                        int(c * 100.0 / self.args.map_resolution)]

        self.full_map[env_idx, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

        self.local_map_boundary[env_idx] = self.get_local_map_boundaries((loc_r, loc_c))

        self.planner_pose_inputs[env_idx, 3:] = self.local_map_boundary[env_idx]
        self.origins[env_idx] = [self.local_map_boundary[env_idx][2] * self.args.map_resolution / 100.0,
                      self.local_map_boundary[env_idx][0] * self.args.map_resolution / 100.0, 0.]

        self.local_map[env_idx] = self.full_map[env_idx, :, self.local_map_boundary[env_idx, 0]:self.local_map_boundary[env_idx, 1], self.local_map_boundary[env_idx, 2]:self.local_map_boundary[env_idx, 3]]
        self.local_pose[env_idx] = self.full_pose[env_idx] - \
            torch.from_numpy(self.origins[env_idx]).to(self.device).float()

    def update_intrinsic_rew(self, viz,step,next_goal_maps,env_idx=0):
        self.full_map[env_idx, :, self.local_map_boundary[env_idx, 0]:self.local_map_boundary[env_idx, 1], self.local_map_boundary[env_idx, 2]:self.local_map_boundary[env_idx, 3]] = \
            self.local_map[env_idx]

        self.global_exploration_map=self.full_map[0,1,:,:]

        with torch.no_grad():
            occ = self.full_map[0, 0].detach().clamp(0, 1)   # 占用
            pos = self.full_map[0, 2].detach()               # 当前位置点
            traj = self.full_map[0, 3].detach()              # 轨迹

            traj_m = traj > 0
            pos_m  = pos  > 0
            base = occ
            rgb = torch.stack([base, base, base], dim=-1).clone()  # [H,W,3]

            #轨迹
            rgb[..., 0][traj_m] = 0.0
            rgb[..., 1][traj_m] = 1.0
            rgb[..., 2][traj_m] = 0.0

            # 当前位置（覆盖）
            rgb[..., 0][pos_m] = 1.0
            rgb[..., 1][pos_m] = 0.0
            rgb[..., 2][pos_m] = 0.0


            next_goal = torch.as_tensor(next_goal_maps, device=rgb.device).bool()

            # 5x5 可视化膨胀
            import torch.nn.functional as F
            next_goal_5x5 = F.max_pool2d(next_goal.float().unsqueeze(0).unsqueeze(0),
                             kernel_size=5, stride=1, padding=2) \
                    .squeeze(0).squeeze(0).bool()

            # 着色
            rgb[next_goal_5x5] = torch.tensor([1.0, 0.0, 1.0], device=rgb.device)


            self.global_map_position_trajectory = rgb


