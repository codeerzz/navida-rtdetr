import os
from glob import glob

from setuptools import setup

package_name = 'rtdetr_zed_tracker'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml') + glob('config/*.rviz')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='metehan',
    maintainer_email='a_metehan-tr@hotmail.com',
    description='RT-DETR + ByteTrack + ZED depth 3D object tracking.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tracker_node = rtdetr_zed_tracker.tracker_node:main',
            'overlay_node = rtdetr_zed_tracker.overlay_node:main',
            'depth_fusion_node = rtdetr_zed_tracker.depth_fusion_node:main',
            'viewer_node = rtdetr_zed_tracker.viewer_node:main',
            'yolo_world_node = rtdetr_zed_tracker.yolo_world_node:main',
        ],
    },
)
