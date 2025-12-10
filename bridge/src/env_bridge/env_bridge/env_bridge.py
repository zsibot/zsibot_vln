#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Imu, PointCloud2
from nav_msgs.msg import Odometry
from message_filters import Subscriber, ApproximateTimeSynchronizer
import zmq, json, cv2, numpy as np, math, io, argparse, time
import math
from sensor_msgs_py import point_cloud2 as pc2

image_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=2)
imu_qos   = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)
odom_qos  = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,    depth=50)

####################
# if you need all data sent from MATRiX
# note this might need finetune of parameters for syncronization
tranfer_all = False
default_tcp = "tcp://127.0.0.1:5999"

imu_data_raw_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)
livox_imu_qos    = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)
livox_lidar_qos  = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)

###################

def imu_to_vec(msg: Imu) -> np.ndarray:
    """Imu → 10-dim vector [ox,oy,oz,ow, wx,wy,wz, ax,ay,az]."""
    return np.array([
        msg.orientation.x,
        msg.orientation.y,
        msg.orientation.z,
        msg.orientation.w,
        msg.angular_velocity.x,
        msg.angular_velocity.y,
        msg.angular_velocity.z,
        msg.linear_acceleration.x,
        msg.linear_acceleration.y,
        msg.linear_acceleration.z,
    ], dtype=np.float32)


def lidar_to_xyz(msg: PointCloud2, max_points: int = 50000) -> np.ndarray:
    """
    PointCloud2 → (N,3) xyz array.
    """
    pts = []
    for i, p in enumerate(pc2.read_points(msg,
                                          field_names=("x", "y", "z"),
                                          skip_nans=True)):
        if max_points is not None and i >= max_points:
            break
        pts.append([p[0], p[1], p[2]])

    if not pts:
        return np.zeros((0, 3), dtype=np.float32)

    return np.asarray(pts, dtype=np.float32)

def yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    n = math.sqrt(x*x + y*y + z*z + w*w)
    if not math.isfinite(n) or n == 0.0:
        return 0.0
    inv = 1.0 / n
    x, y, z, w = x*inv, y*inv, z*inv, w*inv

    s = 2.0 * (w*z + x*y)
    c = 1.0 - 2.0 * (y*y + z*z)

    if not math.isfinite(s): s = 0.0
    if not math.isfinite(c): c = 1.0

    yaw = math.atan2(s, c)

    if yaw <= -math.pi:
        yaw += 2.0*math.pi
    elif yaw > math.pi:
        yaw -= 2.0*math.pi
    return yaw

def decode_rgb(msg: CompressedImage, out_h=640, out_w=480):

    arr = np.frombuffer(msg.data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError('RGB imdecode failed')

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]

    target_ar = out_h / out_w
    cur_ar    = H / W

    if cur_ar < target_ar:
        new_w = int(H / target_ar)
        x0 = (W - new_w) // 2
        x1 = x0 + new_w
        crop = rgb[:, x0:x1, :]
    else:
        new_h = int(W * target_ar)
        y0 = (H - new_h) // 2
        y1 = y0 + new_h
        crop = rgb[y0:y1, :, :]

    out = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA)

    return out

def decode_depth(msg: CompressedImage, out_h=640, out_w=480, match_ar=True):

    arr = np.frombuffer(msg.data, dtype=np.uint8)
    dep = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)

    if dep is None:
        raise RuntimeError('Depth imdecode failed')
    if dep.ndim == 3:
        dep = cv2.cvtColor(dep, cv2.COLOR_BGR2GRAY)

    H, W = dep.shape[:2]

    if match_ar:
        target_ar = out_h / out_w
        cur_ar = H / W
        if cur_ar < target_ar:
            new_w = int(H / target_ar)
            x0 = (W - new_w) // 2
            dep = dep[:, x0:x0+new_w]
        else:
            new_h = int(W * target_ar)
            y0 = (H - new_h) // 2
            dep = dep[y0:y0+new_h, :]

    dep = cv2.resize(dep, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

    offset = 0
    depth_scaled = (dep.astype(np.float32)- offset) / (199.0-offset)          # (H,W)
    depth_scaled = depth_scaled[..., None]

    return depth_scaled


def make_empty_obs(H=640, W=480):
    return {
        "rgb":   np.zeros((H, W, 3),  dtype=np.uint8),
        "depth": np.zeros((H, W, 1),  dtype=np.float32),
        "compass": np.array([0.0], dtype=np.float32),
        "gps":     np.array([0.0, 0.0], dtype=np.float32),
        "imu":     np.zeros(10,dtype=np.float32),
        "imu_data_raw": np.zeros(10, dtype=np.float32),
        "livox_imu":    np.zeros(10, dtype=np.float32),
        "livox_lidar":  np.zeros((0, 3), dtype=np.float32),
        "pose": {
            "position": np.zeros(3, dtype=np.float32),
            "rotation": np.array([0.0, 0.0, 0.0, 1.0],
                                dtype=np.float32)
        }
    }


class ObsAggregatorZMQ(Node):
    def __init__(self, push_endpoint: str):
        super().__init__('obs_aggregator_zmq')
        self.get_logger().info(f'Push to ZMQ: {push_endpoint}')

        ctx = zmq.Context.instance()
        self.sock = ctx.socket(zmq.PUSH)
        self.sock.setsockopt(zmq.CONFLATE, 1)
        self.sock.setsockopt(zmq.SNDHWM, 1)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(push_endpoint)

        self.sub_rgb   = Subscriber(self, CompressedImage, '/image_raw/compressed',       qos_profile=image_qos)
        self.sub_depth = Subscriber(self, CompressedImage, '/image_raw/compressed/depth', qos_profile=image_qos)

        self.sub_imu   = Subscriber(self, Imu,           '/imu',                          qos_profile=imu_qos)
        self.sub_odom  = Subscriber(self, Odometry,      '/odom/mujoco_odom',             qos_profile=odom_qos) #position in global coordinates, origin is where the dog wakes up.

        if not tranfer_all:
            self.sync = ApproximateTimeSynchronizer(
                [self.sub_rgb, self.sub_depth, self.sub_imu, self.sub_odom],
                queue_size=30, slop=0.08)
            self.sync.registerCallback(self.on_sync)

        else:
            self.sub_imu_data_raw = Subscriber(self, Imu, '/imu/data_raw', qos_profile=imu_data_raw_qos)
            self.sub_livox_imu    = Subscriber(self, Imu, '/livox/imu', qos_profile=livox_imu_qos)
            self.sub_livox_lidar  = Subscriber(self, PointCloud2, '/livox/lidar',qos_profile=livox_lidar_qos)

            self.sync = ApproximateTimeSynchronizer(
                [self.sub_rgb, self.sub_depth, self.sub_imu, self.sub_odom,self.sub_imu_data_raw,self.sub_livox_imu,self.sub_livox_lidar],
                queue_size=30, slop=0.08)
            self.sync.registerCallback(self.on_sync_all)

        self.sent = 0
        self.last_gps =np.zeros(2,dtype=np.float32)

    def on_sync(self, rgb_msg, depth_msg, imu_msg, odom_msg):

        try:
            rgb = decode_rgb(rgb_msg)
            depth = decode_depth(depth_msg)
        except Exception as e:
            self.get_logger().warn(f'decode fail: {e}')
            return

        qi = imu_msg.orientation
        yaw = yaw_from_quat(qi.x, qi.y, qi.z, qi.w)

        p = odom_msg.pose.pose.position
        qo = odom_msg.pose.pose.orientation
        pos = np.array([p.x, p.y, p.z], dtype=np.float32)
        quat = np.array([qo.x, qo.y, qo.z, qo.w], dtype=np.float32)

        H, W = rgb.shape[:2]
        obs = make_empty_obs(H, W)
        obs["rgb"]      = rgb
        obs["depth"]    = depth.astype(np.float32)
        obs["compass"][:] = np.array([yaw], dtype=np.float32)
        obs["gps"][:]     = pos[:2]
        obs["pose"]["position"][:] = pos
        obs["pose"]["rotation"][:] = quat

        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            rgb=obs["rgb"],
            depth=obs["depth"],
            compass=obs["compass"],
            gps=obs["gps"],
            pose_position=obs["pose"]["position"],
            pose_rotation=obs["pose"]["rotation"],
        )
        payload = buf.getvalue()

        try:
            self.sock.send(payload, flags=zmq.DONTWAIT)
            self.sent += 1
            if self.sent % 10 == 0:
                self.get_logger().info(f'sent obs: {self.sent}')
        except zmq.Again:
            pass

        self.last_gps=obs["gps"]


    def on_sync_all(self, rgb_msg, depth_msg, imu_msg, odom_msg,
                    imu_data_raw_msg, livox_imu_msg, livox_lidar_msg):

        try:
            rgb = decode_rgb(rgb_msg)
            depth = decode_depth(depth_msg)
        except Exception as e:
            self.get_logger().warn(f'decode fail: {e}')
            return

        qi = imu_msg.orientation
        yaw = yaw_from_quat(qi.x, qi.y, qi.z, qi.w)

        p = odom_msg.pose.pose.position
        qo = odom_msg.pose.pose.orientation
        pos = np.array([p.x, p.y, p.z], dtype=np.float32)
        quat = np.array([qo.x, qo.y, qo.z, qo.w], dtype=np.float32)

        H, W = rgb.shape[:2]
        obs = make_empty_obs(H, W)
        obs["rgb"]      = rgb
        obs["depth"]    = depth.astype(np.float32)
        obs["compass"][:] = np.array([yaw], dtype=np.float32)
        obs["gps"][:]     = pos[:2]
        obs["pose"]["position"][:] = pos
        obs["pose"]["rotation"][:] = quat

        obs["imu"][:]          = imu_to_vec(imu_msg)
        obs["imu_data_raw"][:] = imu_to_vec(imu_data_raw_msg)
        obs["livox_imu"][:]    = imu_to_vec(livox_imu_msg)
        obs["livox_lidar"]     = lidar_to_xyz(livox_lidar_msg)

        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            rgb=obs["rgb"],
            depth=obs["depth"],
            compass=obs["compass"],
            gps=obs["gps"],
            pose_position = obs["pose"]["position"],
            pose_rotation = obs["pose"]["rotation"],

            imu           = obs["imu"],
            imu_data_raw  = obs["imu_data_raw"],   # (10,)
            livox_imu     = obs["livox_imu"],      # (10,)
            livox_lidar   = obs["livox_lidar"],    # (N,3)
        )
        payload = buf.getvalue()

        try:
            self.sock.send(payload, flags=zmq.DONTWAIT)
            self.sent += 1
            if self.sent % 10 == 0:
                self.get_logger().info(f'sent obs: {self.sent}')
        except zmq.Again:
            pass

        self.last_gps = obs["gps"]

    def destroy_node(self):
        try:
            self.sock.close(0)
        finally:
            super().destroy_node()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--push', type=str, default=default_tcp, help='ZMQ push endpoint for packed obs')
    args = parser.parse_args()
    rclpy.init()
    node = ObsAggregatorZMQ(args.push)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
