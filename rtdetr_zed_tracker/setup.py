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
    description='RT-DETR + whole-box ZED depth density clustering -> 3D object positions.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'depth_fusion_node = rtdetr_zed_tracker.depth_fusion_node:main',
            'overlay_node = rtdetr_zed_tracker.overlay_node:main',
            'color_classification_node = rtdetr_zed_tracker.color_classification_node:main',
            'class_remap_node = rtdetr_zed_tracker.class_remap_node:main',
        ],
    },
)
