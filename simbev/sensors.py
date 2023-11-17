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
Perception sensors for data collection.
"""

import cv2
import time
import carla

import numpy as np
import open3d as o3d

from matplotlib import colormaps as cm

VIRIDIS = np.array(cm.get_cmap('viridis').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])


class RGBCamera:
    '''
    RGB camera class that manages the creation and data acquisition of RGB
    cameras.

    Attributes:
        world: CARLA simulation world.
        sensor_manager: SensorManager the camera belongs to.
        width: image width in pixels.
        height: image height in pixels.
        options: dictionary of camera options.
        image: RGB image data.
        rgb_camera: CARLA RGB camera.

    Methods:
        _process_rgb_image: callback function for processing RGB image data.
        render: render RGB image.
        save: save RGB image to file.
        destroy: destroy RGB camera.
    '''

    def __init__(self, world, sensor_manager, transform, attached, width, height, options):
        self.world = world
        self.sensor_manager = sensor_manager
        self.width = width
        self.height = height
        self.options = options
        self.image = None

        self.sensor_manager.add_camera(self)

        rgb_camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        
        rgb_camera_bp.set_attribute('image_size_x', str(self.width))
        rgb_camera_bp.set_attribute('image_size_y', str(self.height))

        for key in options:
            rgb_camera_bp.set_attribute(key, options[key])

        self.rgb_camera = self.world.spawn_actor(rgb_camera_bp, transform, attach_to=attached)

        self.rgb_camera.listen(self._process_rgb_image)

    def _process_rgb_image(self, image):
        image.convert(carla.ColorConverter.Raw)

        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))

        # Remove alpha channel.
        self.image = array[:, :, :3]
    
    def render(self, window_name='RGB Image'):
        cv2.imshow(window_name, cv2.resize(self.image, (self.width // 4, self.height // 4)))
        cv2.waitKey(1)

    def save(self, camera_name, path, scene, frame):
        cv2.imwrite(f'{path}/carla/sweeps/{camera_name}/SimBEV-scene-{scene}-frame-{frame}-{camera_name}.jpg',
                    self.image)
    
    def destroy(self):
        self.rgb_camera.destroy()


class SemanticCamera:
    '''
    Semantic camera class that manages the creation and data acquisition of
    semantic segmentation cameras.

    Attributes:
        world: CARLA simulation world.
        sensor_manager: SensorManager the camera belongs to.
        width: image width in pixels.
        height: image height in pixels.
        options: dictionary of camera options.
        image: semantically segmented image data.
        semantic_camera: CARLA semantic segmentation camera.

    Methods:
        _process_semantic_image: callback function for processing semantically
            segmented image data.
        render: render semantically segmented image.
        save: save semantically segmented image to file.
        destroy: destroy semantic segmentation camera.
    '''

    def __init__(self, world, sensor_manager, transform, attached, width, height, options):
        self.world = world
        self.sensor_manager = sensor_manager
        self.width = width
        self.height = height
        self.options = options
        self.image = None

        self.sensor_manager.add_semantic_camera(self)

        semantic_camera_bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        
        semantic_camera_bp.set_attribute('image_size_x', str(self.width))
        semantic_camera_bp.set_attribute('image_size_y', str(self.height))

        for key in options:
            semantic_camera_bp.set_attribute(key, options[key])

        self.semantic_camera = self.world.spawn_actor(semantic_camera_bp, transform, attach_to=attached)

        self.semantic_camera.listen(self._process_semantic_image)

    def _process_semantic_image(self, image):
        image.convert(carla.ColorConverter.CityScapesPalette)

        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))

        # Remove alpha channel.
        self.image = array[:, :, :3]

    def render(self, window_name='Segmented BEV Image'):
        cv2.imshow(window_name, self.image)
        cv2.waitKey(1)
    
    def save(self, path, scene, frame):
        road_mask = np.logical_or(self.image[:, :, 2] == 128, self.image[:, :, 2] == 157)
        car_mask = self.image[:, :, 0] == 142
        truck_mask = np.logical_and(self.image[:, :, 0] == 70, self.image[:, :, 1] == 0)
        pedestrian_mask = self.image[:, :, 1] == 20

        ground_truth = np.array([road_mask, car_mask, truck_mask, pedestrian_mask])

        with open(f'{path}/carla/ground-truth/SimBEV-scene-{scene}-frame-{frame}-GT.bin', 'wb') as f:
            np.save(f, ground_truth)
    
    def destroy(self):
        self.semantic_camera.destroy()


class Lidar:
    '''
    Lidar class that manages the creation and data acquisition of lidars.

    Attributes:
        world: CARLA simulation world.
        sensor_manager: SensorManager the lidar belongs to.
        channels: number of lidar channels (beams).
        range: maximum range of lidar.
        options: dictionary of lidar options.
        frame: lidar frame counter.
        point_list: Open3D PointCloud used for point cloud visualization.
        lidar: CARLA lidar.
        data: point cloud data.
        points: lidar cloud points.
        visualizer: Open3D Visualizer for point cloud visualization.

    Methods:
        _process_point_cloud: callback function for processing lidar point
            cloud data.
        _create_visualizer: creates Open3D visualizer.
        render: render point cloud data using Open3D.
        save: save point cloud data to file.
        destroy: destroy lidar.
    '''

    def __init__(self, world, sensor_manager, transform, attached, channels, range, options):
        self.world = world
        self.sensor_manager = sensor_manager
        self.channels = channels
        self.range = range
        self.options = options
        self.frame = 0
        self.point_list = o3d.geometry.PointCloud()

        self.sensor_manager.add_lidar(self)

        lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        
        lidar_bp.set_attribute('channels', str(self.channels))
        lidar_bp.set_attribute('range', str(self.range))

        for key in options:
            lidar_bp.set_attribute(key, options[key])

        self.lidar = self.world.spawn_actor(lidar_bp, transform, attach_to=attached)

        self.lidar.listen(self._process_point_cloud)

    def _process_point_cloud(self, point_cloud):
        # Point cloud data contains x, y, z, and intensity values.
        self.data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')))
        self.data = np.reshape(self.data, (int(self.data.shape[0] / 4), 4))

        # Select the x, y, and z values and flip y data because CARLA uses a
        # left-handed coordinate system.
        self.points = self.data[:, :-1]
        self.points[:, 1] = -self.points[:, 1]

    def _create_visualizer(self, window_name='Lidar Point Cloud', width=1024, height=1024):
        self.visualizer = o3d.visualization.Visualizer()

        self.visualizer.create_window(window_name=window_name, width=width, height=height, left=0, top=0)
        self.visualizer.get_render_option().point_size = 1.0
        self.visualizer.get_render_option().background_color = [0.05, 0.05, 0.05]
        self.visualizer.get_render_option().show_coordinate_frame = True

        self.visualizer.add_geometry(self.point_list)
    
    def render(self):
        if self.frame == 0:
            self._create_visualizer()

        # Generate point cloud colors based on intensity values.
        intensity = self.data[:, -1]
        intensity_log = 1.0 + np.log(intensity) / 0.4
        intensity_color = np.c_[
            np.interp(intensity_log, VID_RANGE, VIRIDIS[:, 0]),
            np.interp(intensity_log, VID_RANGE, VIRIDIS[:, 1]),
            np.interp(intensity_log, VID_RANGE, VIRIDIS[:, 2])]
        
        self.point_list.points = o3d.utility.Vector3dVector(self.points)
        self.point_list.colors = o3d.utility.Vector3dVector(intensity_color)
        
        if self.frame == 2:
            self.visualizer.add_geometry(self.point_list)
        
        self.visualizer.update_geometry(self.point_list)
        self.visualizer.poll_events()
        self.visualizer.update_renderer()
        
        self.frame += 1
        
        time.sleep(0.005)
    
    def save(self, path, scene, frame):
        with open(f'{path}/carla/sweeps/LIDAR_TOP/SimBEV-scene-{scene}-frame-{frame}-LIDAR_TOP.pcd.bin', 'wb') as f:
            np.save(f, self.points)
    
    def destroy(self):
        self.lidar.destroy()
        
        try:
            self.visualizer.destroy_window()
        except AttributeError:
            print('No lidar visualizer to destroy.')
            