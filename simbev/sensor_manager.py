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
and renders them or writes them to file.
"""

import cv2
import time

from scipy.spatial.transform import Rotation as R


class CustomTimer:
    '''
    Timer class that uses a performance counter if available, otherwise time
    in seconds.

    Attributes:
        timer: timer.

    Methods:
        time: return time.
    '''
    
    def __init__(self):
        try:
            self.timer = time.perf_counter
        except AttributeError:
            self.timer = time.time

    def time(self):
        return self.timer()


class SensorManager:
    '''
    Sensor manager class that manages data collection.

    Attributes:
        vehicle: SensorManager's vehicle.
        camera_list: list of RGB cameras.
        lidar_list: list of lidars.
        semantic_camera_list: list of semantic cameras.
        timer: CustomTimer timer.
        camera_name_list: list of RGB camera names.
        data: information about the saved data.

    Methods:
        add_camera: add RGB camera to the sensor manager.
        add_lidar: add lidar to the sensor manager.
        add_semantic_camera: add semantic camera to the sensor manager.
        render: render camera images and lidar point clouds.
        save: save camera images and lidar point clouds.
        destroy: destroy all sensors.
    '''

    def __init__(self, vehicle):
        self.vehicle = vehicle
        
        self.camera_list = []
        self.lidar_list = []
        self.semantic_camera_list = []

        self.timer = CustomTimer()

        self.camera_name_list = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                                 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
        
        self.data = []

    def add_camera(self, camera):
        self.camera_list.append(camera)

    def add_lidar(self, lidar):
        self.lidar_list.append(lidar)

    def add_semantic_camera(self, semantic_camera):
        self.semantic_camera_list.append(semantic_camera)
    
    def render(self):
        for camera, window_name in zip(self.camera_list, self.camera_name_list):
            camera.render(window_name)

        for lidar in self.lidar_list:
            lidar.render()

        for semantic_camera in self.semantic_camera_list:
            semantic_camera.render()
    
    def save(self, path, scene, frame):
        for camera, camera_name in zip(self.camera_list, self.camera_name_list):
            camera.save(camera_name, path, scene, frame)

        for lidar in self.lidar_list:
            lidar.save(path, scene, frame)

        for semantic_camera in self.semantic_camera_list:
            semantic_camera.save(path, scene, frame)

        scene_data = {}

        ego_transform = self.vehicle.get_transform()

        scene_data['ego2global_translation'] = [ego_transform.location.x,
                                                -ego_transform.location.y,
                                                ego_transform.location.z]
        scene_data['ego2global_rotation'] = R.from_euler('xyz',
                                                         [ego_transform.rotation.roll,
                                                          -ego_transform.rotation.pitch,
                                                          -ego_transform.rotation.yaw],
                                                          degrees=True).as_quat().tolist()
        
        scene_data['timestamp'] = round(self.timer.time() * 10e6)

        scene_data['LIDAR_TOP'] = f'{path}/carla/sweeps/LIDAR_TOP/SimBEV-scene-{scene:03d}-frame-{frame:03d}-LIDAR_TOP.pcd.bin'

        for camera_name in self.camera_name_list:
            scene_data[camera_name] = f'{path}/carla/sweeps/{camera_name}/SimBEV-scene-{scene:03d}-frame-{frame:03d}-{camera_name}.jpg'

        scene_data['ground_truth'] = f'{path}/carla/ground-truth/SimBEV-scene-{scene:03d}-frame-{frame:03d}-GT.bin'

        self.data.append(scene_data)
    
    def destroy(self):
        for camera in self.camera_list:
            camera.destroy()

        for lidar in self.lidar_list:
            lidar.destroy()

        for semantic_camera in self.semantic_camera_list:
            semantic_camera.destroy()

        cv2.destroyAllWindows()
