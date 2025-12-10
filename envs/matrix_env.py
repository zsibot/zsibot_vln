import time
import zmq
import io
import numpy             as np
import quaternion        as nq
from   PIL                                          import Image
from   agents.zeroshot.unigoal.utils.fmm.pose_utils import get_rel_pose_change

tcp_in  = 'tcp://127.0.0.1:5999'
tcp_out = "tcp://127.0.0.1:5556"

#the world coordinates is placed at where the robot wakes up with x forward, y leftward and z upward

def make_empty_obs(H=640, W=480):
    return {
        "rgb"         : np.zeros((H, W, 3),  dtype=np.uint8),
        "depth"       : np.zeros((H, W, 1),  dtype=np.float32),
        "compass"     : np.array([0.0], dtype=np.float32),
        "gps"         : np.array([0.0, 0.0], dtype=np.float32),
        "imu"         : np.zeros(10,dtype=np.float32),
        "imu_data_raw": np.zeros(10, dtype=np.float32),
        "livox_imu"   : np.zeros(10, dtype=np.float32),
        "livox_lidar" : np.zeros((0, 3), dtype=np.float32),
        "pose": {
            "position": np.zeros(3, dtype=np.float32),
            "rotation": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        }
    }

def xyzw_to_wxyz(xyzw):
    x, y, z, w = map(float, xyzw)
    return nq.quaternion(w, x, y, z)

def _ang_diff_rad(a, b):
    """ rad a-b to (-pi, pi]"""
    d = a - b
    return (d + np.pi) % (2*np.pi) - np.pi

class MatrixEnv:
    def __init__(self, args):

        self.ctx                      = zmq.Context.instance()
        self.sock                     = self.ctx.socket(zmq.PULL)
        self.sock.setsockopt(zmq.CONFLATE, 1)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.bind(tcp_in)
        self.ctx_action               = zmq.Context.instance()
        self.sock_action              = self.ctx_action.socket(zmq.PUSH)
        self.sock_action.connect(tcp_out)

        self.obs_zsibot               = make_empty_obs()
        self.args                     = args
        self.info                     = {}
        self.timestep                 = None
        self.last_agent_location      = None
        self.last_pos_xyz             = None
        self.last_yaw_rad             = None

    def reset(self):

        self.get_obs()
        self.timestep            = 0
        self.last_pos_xyz        = self.obs_zsibot['pose']['position'].copy()
        self.last_yaw_rad        = self.obs_zsibot["compass"].copy()
        self.last_agent_location = self.get_agent_location()

        self.info['time']               = self.timestep
        self.info['sensor_pose']        = [0., 0., 0.]
        self.info['body_center_height'] = self.args.body_center_height
        self.info["goal_invalid"]       = False

        return self.obs_zsibot, self.info

    def step(self, action):

        action_extracted= action['action']

        if action_extracted   ==1: # forward
            VALUE = b"1"
        elif action_extracted == 2: # left
            VALUE = b"2"
        elif action_extracted == 3:  # right
            VALUE = b"3"
        elif action_extracted == 0 :
            VALUE = b"0"
            print("----------------------------------------------------------------------")
            print("----------------------------------------------------------------------")
            print("-------------find target, waiting, break by ctrl+C-------------------")

            time.sleep(10000000)
        elif action_extracted == 5:
            pass

        if self.timestep==0:
            time.sleep(1.0)
        else:
            if not self.args.debug_mode and action_extracted!=5:
                self.sock_action.send(VALUE)

        time.sleep(0.2)
        pos_thresh         = 0.001   #0.001
        yaw_thresh_deg     = 1. #1.0
        min_stable_samples = 2 #3
        poll_timeout_ms    = 50
        stable_count       = 0
        max_wait_s         = 100.0

        yaw_thresh = np.deg2rad(yaw_thresh_deg)
        deadline   = time.time() + max_wait_s

        poller = zmq.Poller()
        poller.register(self.sock, zmq.POLLIN)

        while True: # receive data when robot stops moving
            socks = dict(poller.poll(poll_timeout_ms))
            if socks.get(self.sock) == zmq.POLLIN:
                npz_bytes           = self.sock.recv()
                data = np.load(io.BytesIO(npz_bytes), allow_pickle=False)
                self.get_obs()

                cur_pos   = self.obs_zsibot['pose']['position']
                cur_yaw   = self.obs_zsibot['compass']

                pos_delta = np.linalg.norm(cur_pos[:2] - self.last_pos_xyz[:2])
                yaw_delta = abs(_ang_diff_rad(cur_yaw, self.last_yaw_rad))

                self.last_pos_xyz = cur_pos.copy()
                self.last_yaw_rad = cur_yaw.copy()

                if pos_delta < pos_thresh and yaw_delta < yaw_thresh:
                    stable_count += 1
                else:
                    stable_count = 0
                if stable_count >= min_stable_samples:
                    break

                if time.time() > deadline:
                    print("why so long time waiting for data comming, exit for debuging")
                    exit()

        agent_pose                      = self.get_agent_pose()
        self.info['body_center_height'] = agent_pose['position'][2]
        dx, dy, do                      = self.get_location_change()
        self.info['sensor_pose']        = [dx, dy, do]
        self.timestep                  += 1
        self.info['time']               = self.timestep

        done=False
        return self.obs_zsibot, done, self.info

    def get_obs(self):
        npz_bytes                = self.sock.recv()
        data                     = np.load(io.BytesIO(npz_bytes), allow_pickle=False)

        self.obs_zsibot['rgb']                 = data['rgb']
        self.obs_zsibot['depth']               = data['depth']
        self.obs_zsibot['gps']                 = data['gps']
        self.obs_zsibot["compass"]             = data['compass']
        self.obs_zsibot['pose']['position'][:] = data['pose_position']
        pose_rotation_temp                     = data['pose_rotation']
        self.obs_zsibot['pose']['rotation']    = xyzw_to_wxyz(pose_rotation_temp)

        #################
        #only when all data are transferred by bridging module
        #self.obs_zsibot["imu"]          = data["imu"]            # shape (10,)
        #self.obs_zsibot["imu_data_raw"] = data["imu_data_raw"]   # shape (10,)
        #self.obs_zsibot["livox_imu"]    = data["livox_imu"]      # shape (10,)
        #self.obs_zsibot["livox_lidar"]  = data["livox_lidar"]    # shape (N,3)
        #################

    def get_agent_pose(self):
        return self.obs_zsibot['pose']

    def get_agent_location(self):
        return self.obs_zsibot["gps"][0],self.obs_zsibot["gps"][1],self.obs_zsibot["compass"][0]

    def get_location_change(self):
        curr_agent_location      = self.get_agent_location()
        dx, dy, do               = get_rel_pose_change(curr_agent_location, self.last_agent_location)
        self.last_agent_location = curr_agent_location
        return dx, dy, do

    def seed(self, seed):
        pass

def construct_envs(args):
    env = MatrixEnv(args=args)
    return env
