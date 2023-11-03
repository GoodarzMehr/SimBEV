# MIT License
#
# Copyright (C) 2023 Goodarz Mehr
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.

"""
Sensor manager module that collects data from all of the sensors on a vehicle
and writes them file.
"""

import numpy as np


class SensorManager:
    '''
    BEV manager class that manages the fusion of camera and lidar data using
    BEVFusion to create a probabilistic classification map (PCM).

    Attributes:
        camera_list: List of RGB cameras.
        lidar_list: List of lidars.
        semantic_camera_list: List of semantic cameras.
        camera_name_list: List of RGB camera names.

    Methods:
        add_camera: add RGB camera to the BEV manager.
        add_lidar: add lidar to the BEV manager.
        add_semantic_camera: add semantic camera to the BEV manager.
        prepare_camera: prepare RGB camera names for rendering camera images.
        create_lidar_visualizer: create lidar visualizer.
        render: render camera images and lidar point clouds.
        get_ground_truth: get ground truth PCM.
        destroy: destroy all sensors.
    '''

    def __init__(self):
        self.camera_list = []
        self.lidar_list = []
        self.semantic_camera_list = []

        self.camera_name_list = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                                 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

    def add_camera(self, camera):
        self.camera_list.append(camera)

    def add_lidar(self, lidar):
        self.lidar_list.append(lidar)

    def add_semantic_camera(self, semantic_camera):
        self.semantic_camera_list.append(semantic_camera)
    
    def create_lidar_visualizer(self):
        for lidar in self.lidar_list:
            lidar.create_visualizer()
    
    def render(self):
        for camera, window_name in zip(self.camera_list, self.camera_name_list):
            camera.render(window_name)

        for semantic_camera in self.semantic_camera_list:
            semantic_camera.render()

        for lidar in self.lidar_list:
            lidar.render()
    
    def save(self, scene, frame):
        for camera, camera_name in zip(self.camera_list, self.camera_name_list):
            camera.save(camera_name, scene, frame)

        for lidar in self.lidar_list:
            lidar.save(scene, frame)

        for semantic_camera in self.semantic_camera_list:
            semantic_camera.save(scene, frame)
    
    def destroy(self):
        for camera in self.camera_list:
            camera.destroy()

        for semantic_camera in self.semantic_camera_list:
            semantic_camera.destroy()

        for lidar in self.lidar_list:
            lidar.destroy()