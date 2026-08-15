"""Launch depth fusion (+ optional enrichment stages) as one graph.

Pipeline: RT-DETR -> whole-box depth density clustering -> world-frame tracking.
The image-space ByteTrack stage is gone; depth_fusion_node consumes RT-DETR
detections directly and identity is established downstream by buoy_mapper_node
from world geometry.

Two OPTIONAL enrichment stages can be spliced in ahead of the fusion node. Each
is off by default and each is a plain topic-in/topic-out filter, so the graph
degrades to the bare pipeline when both are off:

    /detections_output
       └─(enable_class_remap)→ class_remap_node
            └─(enable_color_refinement)→ color_classification_node
                 └→ depth_fusion_node

Whichever stage is disabled is simply skipped -- the next stage's input topic is
computed below, so no other node needs to know which stages are running. With
both off, depth_fusion_node subscribes to the raw detections topic exactly as it
would if this file had never heard of either stage.

Everything on this graph speaks the RT-DETR wire format, where a detection's
``class_id`` is the NUMERIC class index as a string ("4"), not a name. Both
optional stages preserve that: class_remap_node maps index->index, and
color_classification_node is given class_labels_file so it can decide in name
space but write the index back out. depth_fusion_node still does int(class_id).

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
  enable_class_remap   true/false — collapse RT-DETR's 7 trained classes to a
                        generic "buoy", no retraining (default false; see class_remap.yaml)
  class_remap_file     YAML index->index overrides (default packaged config/class_remap.yaml)
  enable_color_refinement  true/false — YCrCb color re-check on buoy detections (default false)
  color_ranges_file    YAML color range overrides (default packaged config/color_ranges.yaml)
  color_vote_key       id|grid|none — how per-detection colour votes are keyed (default grid)
  color_vote_cell_px   grid cell size in px for color_vote_key=grid (default 64)

The intended pairing is enable_class_remap:=true enable_color_refinement:=true --
every trained class collapses to a shape-only "buoy" and YCrCb decides red vs
green, instead of trusting the colour RT-DETR was trained to predict.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

PKG = 'rtdetr_zed_tracker'

# Outputs of the two optional stages. Each is '<node name>/<relative topic>', so
# they must track the `name=` given to the matching Node below.
REMAP_OUT = '/class_remap_node/detections_remapped'
COLOR_OUT = '/color_classification_node/detections_color_refined'


def _if_enabled(flag, enabled_topic, disabled_topic):
    """``enabled_topic`` when the launch-argument ``flag`` is truthy, else ``disabled_topic``.

    Launch arguments are strings, never bools -- 'False', 'false' and '0' all have
    to read as off, hence the explicit lower()/tuple test rather than truthiness
    (a bare `if 'False'` is True in Python and would silently enable a stage the
    user asked to turn off).

    ``disabled_topic`` may itself be one of these expressions, which is what lets
    the stages chain: with the colour stage on and the remap stage off, the colour
    stage's own input resolves through to the raw detections topic.
    """
    return PythonExpression([
        "'", enabled_topic, "' if '", flag, "'.lower() in ('true', '1') else '", disabled_topic, "'",
    ])


def generate_launch_description():
    share = get_package_share_directory(PKG)
    params = os.path.join(share, 'config', 'fusion_params.yaml')
    labels = os.path.join(share, 'config', 'class_labels.yaml')
    default_color_ranges = os.path.join(share, 'config', 'color_ranges.yaml')
    default_class_remap = os.path.join(share, 'config', 'class_remap.yaml')

    detections = LaunchConfiguration('detections_topic')
    depth = LaunchConfiguration('depth_topic')
    depth_info = LaunchConfiguration('depth_info_topic')
    color = LaunchConfiguration('color_topic')
    enable_overlay = LaunchConfiguration('enable_overlay')
    enable_class_remap = LaunchConfiguration('enable_class_remap')
    class_remap_file = LaunchConfiguration('class_remap_file')
    enable_color_refinement = LaunchConfiguration('enable_color_refinement')
    color_ranges_file = LaunchConfiguration('color_ranges_file')
    color_vote_key = LaunchConfiguration('color_vote_key')
    color_vote_cell_px = LaunchConfiguration('color_vote_cell_px')
    color_min_confidence_overrides = LaunchConfiguration('color_min_confidence_overrides')

    args = [
        DeclareLaunchArgument('detections_topic', default_value='/detections_output'),
        DeclareLaunchArgument('depth_topic', default_value='/zed_node/depth/depth_registered'),
        DeclareLaunchArgument('depth_info_topic', default_value='/zed_node/depth/camera_info'),
        DeclareLaunchArgument('color_topic', default_value='/zed_node/left/image_rect_color'),
        DeclareLaunchArgument('enable_overlay', default_value='false'),
        DeclareLaunchArgument('enable_class_remap', default_value='false',
                              description='Collapse RT-DETR classes to a generic "buoy", '
                                          'no retraining (see class_remap.yaml)'),
        DeclareLaunchArgument('class_remap_file', default_value=default_class_remap,
                              description='YAML class-index remap for class_remap_node'),
        DeclareLaunchArgument('enable_color_refinement', default_value='false',
                              description='Enable YCrCb color re-check on buoy detections'),
        DeclareLaunchArgument('color_ranges_file', default_value=default_color_ranges,
                              description='YAML color ranges for color_classification_node'),
        DeclareLaunchArgument('color_vote_key', default_value='grid',
                              description='id|grid|none — how colour votes are keyed. There are '
                                          'no track ids on this graph any more, so grid (quantised '
                                          'box centre) is what keeps the vote working'),
        DeclareLaunchArgument('color_vote_cell_px', default_value='64.0',
                              description='Grid cell size in px when color_vote_key:=grid'),
        DeclareLaunchArgument('color_min_confidence_overrides', default_value='black:0.30',
                              description='Per-colour confidence bars, "colour:threshold" comma '
                                          'separated. black is raised above the global default '
                                          'because it sits at neutral chroma, where any dark '
                                          'washed-out patch can imitate it'),
    ]

    # ------------------------------------------------------------------ topic chain
    # Each stage reads whatever the stage before it produced, falling through to
    # the raw detections topic for every stage that is switched off.
    after_remap = _if_enabled(enable_class_remap, REMAP_OUT, detections)
    after_color = _if_enabled(enable_color_refinement, COLOR_OUT, after_remap)

    # ------------------------------------------------------------------ nodes
    # Optional stage 1: collapse RT-DETR's trained classes (see class_remap.yaml)
    # to a single generic shape. Index in, index out.
    class_remap = Node(
        package=PKG, executable='class_remap_node', name='class_remap_node', output='screen',
        condition=IfCondition(enable_class_remap),
        parameters=[{'class_remap_file': class_remap_file}],
        remappings=[('~/detections_input', detections)],
    )

    # Optional stage 2: re-check each buoy detection's colour against a
    # lighting-robust YCrCb threshold (see rtdetr_zed_tracker/color_classifier.py)
    # and correct the class index when the vote disagrees with the detector.
    #
    # class_labels_file is what keeps this honest on the wire: the node decides in
    # name space ('red_buoy'), but reads and writes the numeric index everyone else
    # on this graph speaks. Without it the node would compare '4' against
    # {'buoy', 'red_buoy', ...}, match nothing, and silently refine nothing.
    #
    # color_vote_key:=grid because LabelVote needs a key and there are no track ids
    # here any more -- see the node's docstring for what the grid key does and does
    # not buy you.
    color_classification = Node(
        package=PKG, executable='color_classification_node', name='color_classification_node',
        output='screen',
        condition=IfCondition(enable_color_refinement),
        parameters=[{
            'color_ranges_file': color_ranges_file,
            'class_labels_file': labels,
            'vote_key': color_vote_key,
            'vote_cell_px': color_vote_cell_px,
            'min_confidence_overrides': color_min_confidence_overrides,
        }],
        remappings=[
            ('~/image', color),
            ('~/detections_input', after_remap),
        ],
    )

    fusion = Node(
        package=PKG, executable='depth_fusion_node', name='depth_fusion_node', output='screen',
        parameters=[params, {'class_labels_file': labels}],
        remappings=[
            ('~/detections_input', after_color),
            ('~/depth', depth),
            ('~/depth_camera_info', depth_info),
        ],
    )

    # Draws the SAME detections the fusion node consumes, so a colour correction is
    # visible on screen rather than something you have to take on faith from a log
    # line. class_labels_file turns the numeric index back into a readable name.
    overlay = Node(
        package=PKG, executable='overlay_node', name='overlay_node', output='screen',
        condition=IfCondition(enable_overlay),
        parameters=[{'class_labels_file': labels}],
        remappings=[('~/image', color), ('~/tracks_2d', after_color)],
    )

    return LaunchDescription(args + [class_remap, color_classification, fusion, overlay])
