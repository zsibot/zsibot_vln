from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # 启动节点1
        Node(
            package='mc_sdk_bridge',
            executable='mc_sdk_bridge',
            name='mc_sdk_bridge',
            output='screen',
            parameters=[{

            }]
        )
    ])
