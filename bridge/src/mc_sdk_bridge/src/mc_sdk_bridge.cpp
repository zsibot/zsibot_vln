#include "mc_sdk_bridge.h"

ZSIBOT_IMPL_TOPIC(SDKCmd, Ecal);
ZSIBOT_IMPL_TOPIC(SDKState, Ecal);

using namespace mc_sdk_bridge;

McSdkNode::McSdkNode() : rclcpp::Node("mc_sdk_bridge") {
  RCLCPP_INFO(this->get_logger(), "node created");

  publisher_ = this->create_publisher<HighLevelRobotState>("highlevel_robotstate", 10);

  subscriber_ = this->create_subscription<HighLevelCmd>(
      "/highlevel_cmd", 10,
      std::bind(&McSdkNode::cmdCallback, this, std::placeholders::_1));

  sdk_pub_ = std::make_unique<zsibot::rmw::Publisher<SDKCmd>>();
  sdk_sub_ = std::make_unique<zsibot::rmw::Subscriber<SDKState>>();

  init();
  startZmqListener_();
}

McSdkNode::~McSdkNode() { stopZmqListener_(); }

void McSdkNode::init() {
  bool ok_pub = sdk_pub_->init("sdk_cmd");
  if (!ok_pub) {
    RCLCPP_ERROR(this->get_logger(), "sdk_pub_ init failed");
  }

  bool ok_sub = sdk_sub_->init("sdk_robotstate");
  if (!ok_sub) {
    RCLCPP_ERROR(this->get_logger(), "sdk_sub_ init failed");
  }

  sdk_sub_->subscribe([this](const zsibot::rmw::MessagePtr<robot_sdk::pb::SDKRobotState> &msg) {
    mc_sdk_robot_state_.acc.set__x(msg->data.acc()[0]);
    mc_sdk_robot_state_.acc.set__y(msg->data.acc()[1]);
    mc_sdk_robot_state_.acc.set__z(msg->data.acc()[2]);

    mc_sdk_robot_state_.gyro.set__x(msg->data.gyro()[0]);
    mc_sdk_robot_state_.gyro.set__y(msg->data.gyro()[1]);
    mc_sdk_robot_state_.gyro.set__z(msg->data.gyro()[2]);

    mc_sdk_robot_state_.rpy.set__x(msg->data.rpy()[0]);
    mc_sdk_robot_state_.rpy.set__y(msg->data.rpy()[1]);
    mc_sdk_robot_state_.rpy.set__z(msg->data.rpy()[2]);

    mc_sdk_robot_state_.pos.set__x(msg->data.position()[0]);
    mc_sdk_robot_state_.pos.set__y(msg->data.position()[1]);
    mc_sdk_robot_state_.pos.set__z(msg->data.position()[2]);

    mc_sdk_robot_state_.vel.set__x(msg->data.reserved_float()[0]);
    mc_sdk_robot_state_.vel.set__y(msg->data.reserved_float()[1]);
    mc_sdk_robot_state_.vel.set__z(msg->data.reserved_float()[2]);

    mc_sdk_robot_state_.vel_world.set__x(msg->data.v_world()[0]);
    mc_sdk_robot_state_.vel_world.set__y(msg->data.v_world()[1]);
    mc_sdk_robot_state_.vel_world.set__z(msg->data.v_world()[2]);

    publisher_->publish(mc_sdk_robot_state_);
  });

  RCLCPP_INFO(this->get_logger(), "SDK wiring ready");
}

void McSdkNode::cmdCallback(const HighLevelCmd &msg) {
  RCLCPP_INFO(this->get_logger(), "Received cmd: %d", msg.control_mode);
  mc_sdk_cmd_.set_control_mode(msg.control_mode);
  mc_sdk_cmd_.set_motion_mode(msg.motion_mode);
  mc_sdk_cmd_.set_vx(msg.cmd_vel.x);
  mc_sdk_cmd_.set_vy(msg.cmd_vel.y);
  mc_sdk_cmd_.set_vz(msg.cmd_vel.z);
  mc_sdk_cmd_.set_yaw_rate(msg.cmd_angular.z);
  mc_sdk_cmd_.set_pitch_rate(msg.cmd_angular.y);
  mc_sdk_cmd_.set_roll_rate(msg.cmd_angular.x);
  sdk_pub_->publish(mc_sdk_cmd_);
}

// ---------------- ZMQ ----------------

void McSdkNode::startZmqListener_() {
  try {
    zmq_ctx_  = std::make_unique<zmq::context_t>(1);
    zmq_pull_ = std::make_unique<zmq::socket_t>(*zmq_ctx_, zmq::socket_type::pull);

    zmq_pull_->set(zmq::sockopt::rcvtimeo, 100);
    zmq_pull_->set(zmq::sockopt::linger, 0);
    zmq_pull_->bind(zmq_endpoint_);

    zmq_run_.store(true);
    zmq_thread_ = std::thread([this] { this->zmqLoop_(); });

    RCLCPP_INFO(this->get_logger(), "ZMQ PULL bound on %s", zmq_endpoint_.c_str());
  } catch (const std::exception &e) {
    RCLCPP_ERROR(this->get_logger(), "startZmqListener_ failed: %s", e.what());
  }
}

void McSdkNode::stopZmqListener_() {
  zmq_run_.store(false);
  try {
    if (zmq_thread_.joinable()) {
      zmq_thread_.join();
    }
    if (zmq_pull_) {
      zmq_pull_->close();
      zmq_pull_.reset();
    }
    if (zmq_ctx_) {
      zmq_ctx_->shutdown();
      zmq_ctx_.reset();
    }
  } catch (const std::exception &e) {
    RCLCPP_WARN(this->get_logger(), "stopZmqListener_ raised: %s", e.what());
  }
}



void McSdkNode::zmqLoop_() {
  while (rclcpp::ok() && zmq_run_.load()) {
    zmq::message_t m;
    if (!zmq_pull_->recv(m, zmq::recv_flags::none)) {
      continue;  // timeout
    }

    std::string s(static_cast<const char*>(m.data()), m.size());

    // Log exactly what we received (length + printable/hex)
    std::ostringstream hex;
    for (unsigned char c : s) {
      hex << std::hex << std::uppercase << std::setw(2) << std::setfill('0') << (int)c << ' ';
    }
    RCLCPP_INFO(this->get_logger(), "ZMQ raw: '%s' (len=%zu) hex=[%s]",
                s.c_str(), s.size(), hex.str().c_str());

    // Trim whitespace/newlines
    auto is_ws = [](unsigned char c){ return std::isspace(c); };
    s.erase(std::remove_if(s.begin(), s.end(), is_ws), s.end());

    if (s.empty()) {
      RCLCPP_WARN(this->get_logger(), "Empty command after trimming");
      continue;
    }

    const char cmd = s[0];  // robust to stray bytes like "2\n"
    auto run_for_forward = 1800ms;  //
    auto period_forward  = 10ms;    //
    auto run_for = 150ms;  //
    auto period  = 10ms;    //



    auto runTimed_forward = [&](auto fn, const char* label) {
      RCLCPP_INFO(this->get_logger(), "ZMQ: received '%c' -> %s at 100Hz", cmd, label);
      auto timer = this->create_wall_timer(
        period_forward, [this, fn]() { (this->*fn)(); }
      );
      std::this_thread::sleep_for(run_for_forward);
      timer->cancel();
      RCLCPP_INFO(this->get_logger(), "ZMQ: finished %s", label);
    };

    auto runTimed = [&](auto fn, const char* label) {
      RCLCPP_INFO(this->get_logger(), "ZMQ: received '%c' -> %s at 100Hz", cmd, label);
      auto timer = this->create_wall_timer(
        period, [this, fn]() { (this->*fn)(); }
      );
      std::this_thread::sleep_for(run_for);
      timer->cancel();
      RCLCPP_INFO(this->get_logger(), "ZMQ: finished %s", label);
    };

    switch (cmd) {
      case '1': runTimed_forward(&McSdkNode::publishForward_, "forward publishing"); break;
      case '2': runTimed(&McSdkNode::publishLeft_,    "left");               break;
      case '3': runTimed(&McSdkNode::publishRight_,   "right");              break;
      case '0': runTimed(&McSdkNode::publishStop_,    "stop");               break;
      default:
        RCLCPP_WARN(this->get_logger(), "Invalid action '%c' (trimmed from '%s')", cmd, s.c_str());
        break;
    }
  }
}


void McSdkNode::publishForward_() {
  mc_sdk_cmd_.set_control_mode(control_mode_vel_);
  mc_sdk_cmd_.set_motion_mode(motion_mode_walk_);
  mc_sdk_cmd_.set_vx(0.2);
  mc_sdk_cmd_.set_vy(0.0);
  mc_sdk_cmd_.set_vz(0.0);
  mc_sdk_cmd_.set_yaw_rate(0.0);
  mc_sdk_cmd_.set_pitch_rate(0.0);
  mc_sdk_cmd_.set_roll_rate(0.0);
  sdk_pub_->publish(mc_sdk_cmd_);
  std::cout << "Forward action"<<std::endl;
}

void McSdkNode::publishLeft_() {
  mc_sdk_cmd_.set_control_mode(control_mode_vel_);
  mc_sdk_cmd_.set_motion_mode(motion_mode_walk_);
  mc_sdk_cmd_.set_vx(0.0);
  mc_sdk_cmd_.set_vy(0.0);
  mc_sdk_cmd_.set_vz(0.0);
  mc_sdk_cmd_.set_yaw_rate(0.25);
  mc_sdk_cmd_.set_pitch_rate(0.0);
  mc_sdk_cmd_.set_roll_rate(0.0);
  sdk_pub_->publish(mc_sdk_cmd_);
  std::cout << "Left action"<<std::endl;
}

void McSdkNode::publishRight_() {
  mc_sdk_cmd_.set_control_mode(control_mode_vel_);
  mc_sdk_cmd_.set_motion_mode(motion_mode_walk_);
  mc_sdk_cmd_.set_vx(0.0);
  mc_sdk_cmd_.set_vy(0.0);
  mc_sdk_cmd_.set_vz(0.0);
  mc_sdk_cmd_.set_yaw_rate(-0.25);
  mc_sdk_cmd_.set_pitch_rate(0.0);
  mc_sdk_cmd_.set_roll_rate(0.0);
  sdk_pub_->publish(mc_sdk_cmd_);
  std::cout << "Right action"<<std::endl;
}


void McSdkNode::publishStop_() {
  mc_sdk_cmd_.set_control_mode(control_mode_vel_);
  mc_sdk_cmd_.set_motion_mode(motion_mode_walk_);
  mc_sdk_cmd_.set_vx(0.0);
  mc_sdk_cmd_.set_vy(0.0);
  mc_sdk_cmd_.set_vz(0.0);
  mc_sdk_cmd_.set_yaw_rate(0.0);
  mc_sdk_cmd_.set_pitch_rate(0.0);
  mc_sdk_cmd_.set_roll_rate(0.0);
  sdk_pub_->publish(mc_sdk_cmd_);
}


