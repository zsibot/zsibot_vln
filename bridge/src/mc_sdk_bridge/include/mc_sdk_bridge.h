#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include <zmq.hpp>  // cppzmq

#include "mc_sdk_msg/msg/high_level_cmd.hpp"
#include "mc_sdk_msg/msg/high_level_robot_state.hpp"
#include "rclcpp/rclcpp.hpp"
#include "robot_sdk.pb.h"
#include "zsibot/base.hpp"
#include "zsibot/rmw/ecal.hpp"

using namespace std::chrono_literals;
using mc_sdk_msg::msg::HighLevelCmd;
using mc_sdk_msg::msg::HighLevelRobotState;

ZSIBOT_TOPIC(robot_sdk::pb::SDKCmd, SDKCmd);
ZSIBOT_TOPIC(robot_sdk::pb::SDKRobotState, SDKState);

namespace mc_sdk_bridge {

    class McSdkNode : public rclcpp::Node {
    public:
      McSdkNode();
      ~McSdkNode() override;

    private:
      // SDK <-> ROS wiring
      void init();
      void cmdCallback(const HighLevelCmd &msg);

      // ---------- ZMQ ----------
      void startZmqListener_();   // bind & spawn thread
      void stopZmqListener_();    // stop thread & close
      void zmqLoop_();            // thread function
      void publishForward_();     // send forward command to SDK
      void publishLeft_();
      void publishRight_();
      void publishStop_();

      // ZMQ members
      std::unique_ptr<zmq::context_t> zmq_ctx_;
      std::unique_ptr<zmq::socket_t>  zmq_pull_;
      std::thread                     zmq_thread_;
      std::atomic<bool>               zmq_run_{false};

      // Config
      std::string zmq_endpoint_ = "tcp://127.0.0.1:5556";
      double      forward_vx_   = 0.3;
      int         control_mode_vel_  = 2;
      int         motion_mode_walk_  = 2;
      // -------------------------

      // ROS/SDK members
      rclcpp::Publisher<HighLevelRobotState>::SharedPtr publisher_;
      rclcpp::Subscription<HighLevelCmd>::SharedPtr     subscriber_;
      std::unique_ptr<zsibot::rmw::Subscriber<SDKState>> sdk_sub_;
      std::unique_ptr<zsibot::rmw::Publisher<SDKCmd>>    sdk_pub_;

      robot_sdk::pb::SDKCmd   mc_sdk_cmd_;
      HighLevelRobotState     mc_sdk_robot_state_;
    };

}

