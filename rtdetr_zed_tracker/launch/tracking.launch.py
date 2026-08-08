"""Launch depth fusion (+ optional overlay) as one graph.

Pipeline: RT-DETR -> whole-box depth density clustering -> world-frame tracking.
The image-space ByteTrack stage is gone; depth_fusion_node consumes RT-DETR
detections directly and identity is established downstream by buoy_mapper_node
from world geometry.

The ZED + RT-DETR pipeline must already be running (produces /detections_output,
/zed_node/depth/...). All nodes here are separate processes joining that pipeline,
so they need the UDP Fast DDS profile exported in the shell that runs this launch:

  export FASTRTPS_DEFAULT_PROFILES_FILE=$ISAAC_ROS_WS/src/rtdetr_zed_tracker/udp_only_profile.xml
  ros2 launch rtdetr_zed_tracker tracking.launch.py

Launch arguments:
  detections_topic     RT-DETR detection topic (default /detections_output)
  depth_topic          ZED depth topic
  depth_info_topic     ZED depth camera_info topic
  color_topic          ZED color image topic
  enable_overlay       true/false — RGB overlay for rqt_image_view (default false)
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PKG = 'rtdetr_zed_tracker'


def generate_launch_description():
    share = get_package_share_directory(PKG)
    params = os.path.join(share, 'config', 'fusion_params.yaml')
    labels = os.path.join(share, 'config', 'class_labels.yaml')

    detections = LaunchConfiguration('detections_topic')
    depth = LaunchConfiguration('depth_topic')
    depth_info = LaunchConfiguration('depth_info_topic')
    color = LaunchConfiguration('color_topic')
    enable_overlay = LaunchConfiguration('enable_overlay')

    args = [
        DeclareLaunchArgument('detections_topic', default_value='/detections_output'),
        DeclareLaunchArgument('depth_topic', default_value='/zed_node/depth/depth_registered'),
        DeclareLaunchArgument('depth_info_topic', default_value='/zed_node/depth/camera_info'),
        DeclareLaunchArgument('color_topic', default_value='/zed_node/left/image_rect_color'),
        DeclareLaunchArgument('enable_overlay', default_value='false'),
    ]

    # ------------------------------------------------------------------ nodes
    fusion = Node(
        package=PKG, executable='depth_fusion_node', name='depth_fusion_node', output='screen',
        parameters=[params, {'class_labels_file': labels}],
        remappings=[
            ('~/detections_input', detections),
            ('~/depth', depth),
            ('~/depth_camera_info', depth_info),
        ],
    )

    overlay = Node(
        package=PKG, executable='overlay_node', name='overlay_node', output='screen',
        condition=IfCondition(enable_overlay),
        remappings=[('~/image', color), ('~/tracks_2d', detections)],
    )

    return LaunchDescription(args + [fusion, overlay])
