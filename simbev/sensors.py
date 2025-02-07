# Academic Software License: Copyright © 2025 Goodarz Mehr.

'''
SimBEV perception and navigation sensors.
'''

import cv2
import time
import carla
import flow_vis

import numpy as np
import open3d as o3d

from queue import Queue

from matplotlib import colormaps as cm

RANGE = np.linspace(0.0, 1.0, 256)

RAINBOW = np.array(cm.get_cmap('rainbow')(RANGE))[:, :3]

LABEL_COLORS = np.array([
    (255, 255, 255), # None
    (128, 64, 128),  # Road
    (244, 35, 232),  # Sidewalk
    (70, 70, 70),    # Building
    (102, 102, 156), # Wall
    (190, 153, 153), # Fence
    (153, 153, 153), # Pole
    (250, 170, 30),  # TrafficLight
    (220, 220, 0),   # TrafficSign
    (107, 142, 35),  # Vegetation
    (152, 251, 152), # Terrain
    (70, 130, 180),  # Sky
    (220, 20, 60),   # Pedestrian
    (255, 0, 0),     # Rider
    (0, 0, 142),     # Car
    (0, 0, 70),      # Truck
    (0, 60, 100),    # Bus
    (0, 80, 100),    # Train
    (0, 0, 230),     # Motorcycle
    (119, 11, 32),   # Bicycle
    (110, 190, 160), # Static
    (170, 120, 50),  # Dynamic
    (55, 90, 80),    # Other
    (45, 60, 150),   # Water
    (227, 227, 227), # RoadLine
    (81, 0, 81),     # Ground
    (150, 100, 100), # Bridge
    (230, 150, 140), # RailTrack
    (180, 165, 180), # GuardRail
]) / 255.0


class BaseSensor:
    '''
    Base sensor class that manages the creation and data acquisition of a
    sensor.

    Args:
        world: CARLA simulation world.
        sensor_manager: SensorManager instance that the sensor belongs to.
        transform: sensor's transform relative to what it is attached to.
        attached: CARLA object the sensor is attached to.
    '''

    def __init__(self, world, sensor_manager, transform, attached):
        self.world = world
        self.sensor_manager = sensor_manager
    
    def render(self):
        '''
        Render sensor data.
        '''
        raise NotImplementedError
    
    def save(self):
        '''
        Save sensor data to file.
        '''
        raise NotImplementedError

    def destroy(self):
        '''
        Destroy the sensor.
        '''
        raise NotImplementedError


class BaseCamera(BaseSensor):
    '''
    Base camera class that manages the creation and data acquisition of
    cameras.

    Args:
        world: CARLA simulation world.
        sensor_manager: SensorManager instance that the camera belongs to.
        transform: the camera's transform relative to what it is attached to.
        attached: CARLA object the camera is attached to.
        width: image width in pixels.
        height: image height in pixels.
        options: dictionary of camera options.
    '''

    def __init__(self, world, sensor_manager, transform, attached, width, height, options):
        self.world = world
        self.sensor_manager = sensor_manager
        self.width = width
        self.height = height
        self.options = options
        self.image = None

        # Create queues for rendering and saving images. Since queues are
        # blocking, this ensures that at each time step images are fully
        # acquired before the code continues.
        self.render_queue = Queue()
        self.save_queue = Queue()

        self._get_camera()
        
        self.camera_bp.set_attribute('image_size_x', str(self.width))
        self.camera_bp.set_attribute('image_size_y', str(self.height))

        for key in options:
            self.camera_bp.set_attribute(key, options[key])

        self.camera = self.world.spawn_actor(self.camera_bp, transform, attach_to=attached)

        self.camera.listen(self._process_image)

    def _get_camera(self):
        '''
        Add camera to the SensorManager and get the camera blueprint.
        '''
        raise NotImplementedError
    
    def _process_image(self, image):
        '''
        Callback function for processing raw image data.

        Args:
            image: raw image data.
        '''
        # Convert image colors if necessary and reshape into a
        # (height, width, 4) NumPy array.
        image.convert(carla.ColorConverter.Raw)

        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))

        # Remove the alpha channel.
        self.image = array[:, :, :3]

        # Put the image in both queues.
        self.render_queue.put(self.image)
        self.save_queue.put(self.image)
    
    def clear_queues(self):
        '''
        Clear the queues to ensure only the latest image is accessible.
        '''
        with self.render_queue.mutex:
            self.render_queue.queue.clear()
            self.render_queue.all_tasks_done.notify_all()
            self.render_queue.unfinished_tasks = 0

        with self.save_queue.mutex:
            self.save_queue.queue.clear()
            self.save_queue.all_tasks_done.notify_all()
            self.save_queue.unfinished_tasks = 0
    
    def destroy(self):
        '''
        Destroy the camera.
        '''
        self.camera.destroy()


class RGBCamera(BaseCamera):
    '''
    RGB camera class that manages the creation and data acquisition of RGB
    cameras.
    '''

    def __init__(self, world, sensor_manager, transform, attached, width, height, options):
        super().__init__(world, sensor_manager, transform, attached, width, height, options)

    def _get_camera(self):
        '''
        Add RGB camera to the SensorManager and get the camera blueprint.
        '''
        self.sensor_manager.add_camera(self)

        self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
    
    def render(self, window_name='RGB Image'):
        '''
        Render RGB image.

        Args:
            window_name: window name for the rendered image.
        '''
        cv2.imshow(window_name, cv2.resize(self.render_queue.get(True, 10.0), (self.width // 4, self.height // 4)))
        cv2.waitKey(1)

    def save(self, camera_name, path, scene, frame):
        '''
        Save RGB image to file.

        Args:
            camera_name: name of the camera.
            path: root directory of the dataset.
            scene: scene number.
            frame: frame number.
        '''
        cv2.imwrite(
            f'{path}/simbev/sweeps/RGB-{camera_name}/SimBEV-scene-{scene:04d}-frame-{frame:04d}-RGB-{camera_name}.jpg',
            self.save_queue.get(True, 10.0)
        )


class SemanticCamera(BaseCamera):
    '''
    Semantic segmentation camera class that manages the creation and data
    acquisition of semantic segmentation cameras.
    '''

    def __init__(self, world, sensor_manager, transform, attached, width, height, options):
        super().__init__(world, sensor_manager, transform, attached, width, height, options)
    
    def _get_camera(self):
        '''
        Add semantic segmentation camera to the SensorManager and get the
        camera blueprint.
        '''
        self.sensor_manager.add_semantic_camera(self)

        self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
    
    def _process_image(self, image):
        '''
        Callback function for processing raw image data.

        Args:
            image: raw image data.
        '''
        # Convert image colors using the CityScapes palette and reshape into a
        # (height, width, 4) NumPy array.
        image.convert(carla.ColorConverter.CityScapesPalette)

        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))

        # Remove the alpha channel.
        self.image = array[:, :, :3]

        # Put the image in both queues.
        self.render_queue.put(self.image)
        self.save_queue.put(self.image)
    
    def render(self, window_name='Segmented Image'):
        '''
        Render semantic segmentation image.

        Args:
            window_name: window name for the rendered image.
        '''
        cv2.imshow(window_name, cv2.resize(self.render_queue.get(True, 10.0), (self.width // 4, self.height // 4)))
        cv2.waitKey(1)
    
    def save(self, camera_name, path, scene, frame):
        '''
        Save semantic segmentation image to file.

        Args:
            camera_name: name of the camera.
            path: root directory of the dataset.
            scene: scene number.
            frame: frame number.
        '''
        cv2.imwrite(
            f'{path}/simbev/sweeps/SEG-{camera_name}/SimBEV-scene-{scene:04d}-frame-{frame:04d}-SEG-{camera_name}.png',
            self.save_queue.get(True, 10.0)
        )


class InstanceCamera(BaseCamera):
    '''
    Instance segmentation camera class that manages the creation and data
    acquisition of instance segmentation cameras.
    '''
    
    def __init__(self, world, sensor_manager, transform, attached, width, height, options):
        super().__init__(world, sensor_manager, transform, attached, width, height, options)
    
    def _get_camera(self):
        '''
        Add instance segmentation camera to the SensorManager and get the
        camera blueprint.
        '''
        self.sensor_manager.add_instance_camera(self)

        self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.instance_segmentation')
    
    def render(self, window_name='Instance Image'):
        '''
        Render instance segmentation image.

        Args:
            window_name: window name for the rendered image.
        '''
        cv2.imshow(window_name, cv2.resize(self.render_queue.get(True, 10.0), (self.width // 4, self.height // 4)))
        cv2.waitKey(1)
    
    def save(self, camera_name, path, scene, frame):
        '''
        Save instance segmentation image to file.

        Args:
            camera_name: name of the camera.
            path: root directory of the dataset.
            scene: scene number.
            frame: frame number.
        '''
        cv2.imwrite(
            f'{path}/simbev/sweeps/IST-{camera_name}/SimBEV-scene-{scene:04d}-frame-{frame:04d}-IST-{camera_name}.png',
            self.save_queue.get(True, 10.0)
        )


class DepthCamera(BaseCamera):
    '''
    Depth camera class that manages the creation and data acquisition of depth
    cameras.
    '''

    def __init__(self, world, sensor_manager, transform, attached, width, height, options):
        super().__init__(world, sensor_manager, transform, attached, width, height, options)
    
    def _get_camera(self):
        '''
        Add depth camera to the SensorManager and get the camera blueprint.
        '''
        self.sensor_manager.add_depth_camera(self)

        self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.depth')
    
    def render(self, window_name='Depth Image'):
        '''
        Render depth image.

        Args:
            window_name: window name for the rendered image.
        '''
        image = self.render_queue.get(True, 10.0)

        # 1000 * (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1)
        # provides the depth value in meters. This is then converted to a
        # logarithmic scale for better visualization.
        normalized_distance = (image[:, :, 2] + image[:, :, 1] * 256.0 + image[:, :, 0] * 256.0 * 256.0) \
            / (256.0 * 256.0 * 256.0 - 1)
        
        log_distance = 255 * np.log(256.0 * normalized_distance + 1) / np.log(257.0)
        
        cv2.imshow(window_name, cv2.resize(log_distance.astype(np.uint8), (self.width // 4, self.height // 4)))
        cv2.waitKey(1)
    
    def save(self, camera_name, path, scene, frame):
        '''
        Save depth image to file.

        Args:
            camera_name: name of the camera.
            path: root directory of the dataset.
            scene: scene number.
            frame: frame number.
        '''
        cv2.imwrite(
            f'{path}/simbev/sweeps/DPT-{camera_name}/SimBEV-scene-{scene:04d}-frame-{frame:04d}-DPT-{camera_name}.png',
            self.save_queue.get(True, 10.0)
        )


class FlowCamera(BaseCamera):
    '''
    Flow camera class that manages the creation and data acquisition of flow
    cameras.
    '''

    def __init__(self, world, sensor_manager, transform, attached, width, height, options):
        super().__init__(world, sensor_manager, transform, attached, width, height, options)
    
    def _get_camera(self):
        '''
        Add flow camera to the SensorManager and get the camera blueprint.
        '''
        self.sensor_manager.add_flow_camera(self)

        self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.optical_flow')
    
    def _process_image(self, image):
        '''
        Callback function for processing raw image data.

        Args:
            image: raw image data.
        '''
        # Reshape into a (height, width, 2) NumPy array.
        array = np.frombuffer(image.raw_data, dtype=np.float32)
        self.image = np.reshape(array, (image.height, image.width, 2))

        # Put the image in both queues.
        self.render_queue.put(self.image)
        self.save_queue.put(self.image)
    
    def render(self, window_name='Flow Image'):
        '''
        Render flow image.

        Args:
            window_name: window name for the rendered image.
        '''
        image = flow_vis.flow_to_color(self.render_queue.get(True, 10.0), convert_to_bgr=True)
        
        cv2.imshow(window_name, cv2.resize(image, (self.width // 4, self.height // 4)))
        cv2.waitKey(1)
    
    def save(self, camera_name, path, scene, frame):
        '''
        Save flow image to file.

        Args:
            camera_name: name of the camera.
            path: root directory of the dataset.
            scene: scene number.
            frame: frame number.
        '''
        with open(
            f'{path}/simbev/sweeps/FLW-{camera_name}/SimBEV-scene-{scene:04d}-frame-{frame:04d}-FLW-{camera_name}.npz',
            'wb'
        ) as f:
            np.savez_compressed(f, data=self.save_queue.get(True, 10.0))


class BaseLidar(BaseSensor):
    '''
    Base lidar class that manages the creation and data acquisition of lidars.

    Args:
        world: CARLA simulation world.
        sensor_manager: SensorManager instance that the lidar belongs to.
        transform: the lidar's transform relative to what it is attached to.
        attached: CARLA object the lidar is attached to.
        channels: number of lidar channels (beams).
        range: maximum range of the lidar.
        options: dictionary of lidar options.
    '''

    def __init__(self, world, sensor_manager, transform, attached, channels, range, options):
        self.world = world
        self.sensor_manager = sensor_manager
        self.channels = channels
        self.range = range
        self.options = options
        self.frame = 0
        self.point_list = o3d.geometry.PointCloud()

        # Create queues for rendering and saving point clouds. Since queues
        # are blocking, this ensures that at each time step point clouds are
        # fully acquired before the code continues.
        self.render_queue = Queue()
        self.save_queue = Queue()

        self._get_lidar()
        
        self.lidar_bp.set_attribute('channels', str(self.channels))
        self.lidar_bp.set_attribute('range', str(self.range))

        for key in options:
            self.lidar_bp.set_attribute(key, options[key])

        self.lidar = self.world.spawn_actor(self.lidar_bp, transform, attach_to=attached)

        self.lidar.listen(self._process_point_cloud)

    def _get_lidar(self):
        '''
        Add lidar to the SensorManager and get the lidar blueprint.
        '''
        raise NotImplementedError
    
    def _process_point_cloud(self, point_cloud):
        '''
        Callback function for processing raw point cloud data.

        Args:
            point_cloud: raw point cloud data.
        '''
        raise NotImplementedError

    def clear_queues(self):
        '''
        Clear the queues to ensure only the latest point cloud is accessible.
        '''
        with self.render_queue.mutex:
            self.render_queue.queue.clear()
            self.render_queue.all_tasks_done.notify_all()
            self.render_queue.unfinished_tasks = 0

        with self.save_queue.mutex:
            self.save_queue.queue.clear()
            self.save_queue.all_tasks_done.notify_all()
            self.save_queue.unfinished_tasks = 0
    
    def _create_visualizer(self, window_name, width=1024, height=1024):
        '''
        Create Open3D visualizer.

        Args:
            window_name: window name for the point cloud visualizer.
            width: visualizer window width in pixels.
            height: visualizer window height in pixels.
        '''
        self.visualizer = o3d.visualization.Visualizer()

        self.visualizer.create_window(window_name=window_name, width=width, height=height, left=0, top=0)
        self.visualizer.get_render_option().point_size = 1.0
        if self.channels == 27: print('Copyright © 2025 Goodarz Mehr')
        self.visualizer.get_render_option().background_color = [0.04, 0.04, 0.04]
        self.visualizer.get_render_option().show_coordinate_frame = True

        self.visualizer.add_geometry(self.point_list)
    
    def _draw_points(self, color):
        '''
        Visualize point cloud data in Open3D.

        Args:
            color: point cloud colors.
        '''
        self.point_list.points = o3d.utility.Vector3dVector(self.render_queue.get(True, 10.0))
        self.point_list.colors = o3d.utility.Vector3dVector(color)
        
        if self.frame == 2:
            self.visualizer.add_geometry(self.point_list)

            # Place the camera at a height where the entire point cloud is
            # visible. The camera's field of view is 60 degrees by default.
            camera_height = np.sqrt(3.0) * self.range

            camera = self.visualizer.get_view_control().convert_to_pinhole_camera_parameters()

            pose = np.eye(4)

            pose[1, 1] = -1
            pose[2, 2] = -1
            pose[:3, 3] = [0, 0, camera_height]

            camera.extrinsic = pose

            self.visualizer.get_view_control().convert_from_pinhole_camera_parameters(camera)
        
        self.visualizer.update_geometry(self.point_list)
        self.visualizer.poll_events()
        self.visualizer.update_renderer()
        
        self.frame += 1
        
        time.sleep(0.005)
    
    def destroy(self):
        '''
        Destroy the lidar.
        '''
        self.lidar.destroy()
        
        try:
            self.visualizer.destroy_window()
        except AttributeError:
            pass


class Lidar(BaseLidar):
    '''
    Lidar class that manages the creation and data acquisition of lidars.
    '''

    def __init__(self, world, sensor_manager, transform, attached, channels, range, options):
        super().__init__(world, sensor_manager, transform, attached, channels, range, options)

    def _get_lidar(self):
        '''
        Add lidar to the SensorManager and get the lidar blueprint.
        '''
        self.sensor_manager.add_lidar(self)

        self.lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
    
    def _process_point_cloud(self, point_cloud):
        '''
        Callback function for processing raw point cloud data.

        Args:
            point_cloud: raw point cloud data.
        '''
        # Point cloud data contains x, y, z, and intensity values.
        self.data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')))
        self.data = np.reshape(self.data, (int(self.data.shape[0] / 4), 4))

        # Select the x, y, and z values and flip y data because CARLA uses a
        # left-handed coordinate system.
        self.points = self.data[:, :-1]
        self.points[:, 1] *= -1

        # Remove the points belonging to the ego vehicle. The limits are based
        # on the 2016 Mustang dimensions and should be adjusted for other
        # vehicles.
        mask = np.logical_or(abs(self.points[:, 0]) > 2.44, abs(self.points[:, 1]) > 1.04)

        self.data = self.data[mask]
        self.points = self.points[mask]

        # Put the point cloud in both queues.
        self.render_queue.put(self.points)
        self.save_queue.put(self.points)
    
    def render(self):
        '''
        Render point cloud.
        '''
        if self.frame == 0:
            self._create_visualizer(window_name='Lidar Point Cloud')

        # Generate point cloud colors based on intensity values.
        distance = np.linalg.norm(self.points, axis=1)
        distance_log = np.log(distance)
        distance_log_normalized = (
            distance_log - distance_log.min()
        ) / (
            distance_log.max() - distance_log.min() + 1e-6
        )
        intensity_color = np.c_[
            np.interp(distance_log_normalized, RANGE, RAINBOW[:, 0]),
            np.interp(distance_log_normalized, RANGE, RAINBOW[:, 1]),
            np.interp(distance_log_normalized, RANGE, RAINBOW[:, 2])
        ]
        
        self._draw_points(intensity_color)
    
    def save(self, path, scene, frame):
        '''
        Save point cloud to file.

        Args:
            path: root directory of the dataset.
            scene: scene number.
            frame: frame number.
        '''
        with open(f'{path}/simbev/sweeps/LIDAR/SimBEV-scene-{scene:04d}-frame-{frame:04d}-LIDAR.npz', 'wb') as f:
            np.savez_compressed(f, data=self.save_queue.get(True, 10.0))


class SemanticLidar(BaseLidar):
    '''
    Semantic lidar class that manages the creation and data acquisition of
    semantic lidars.
    '''

    def __init__(self, world, sensor_manager, transform, attached, channels, range, options):
        super().__init__(world, sensor_manager, transform, attached, channels, range, options)

    def _get_lidar(self):
        '''
        Add semantic lidar to the SensorManager and get the semantic lidar
        blueprint.
        '''
        self.sensor_manager.add_semantic_lidar(self)

        self.lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
    
    def _process_point_cloud(self, point_cloud):
        '''
        Callback function for processing raw point cloud data.

        Args:
            point_cloud: raw point cloud data.
        '''
        # Point cloud data contains x, y, z, cosine angle, object index, and
        # object tag values.
        self.data = np.copy(
            np.frombuffer(
                point_cloud.raw_data,
                dtype=np.dtype([
                    ('x', np.float32),
                    ('y', np.float32),
                    ('z', np.float32),
                    ('CosAngle', np.float32),
                    ('ObjIdx', np.uint32),
                    ('ObjTag', np.uint32)]
                )
            )
        )

        # Select the x, y, and z values and flip y data because CARLA uses a
        # left-handed coordinate system.
        self.data['y'] *= -1
        self.points = np.array([self.data['x'], self.data['y'], self.data['z']]).T

        # Remove the points belonging to the ego vehicle. The limits are based
        # on the 2016 Mustang dimensions and should be adjusted for other
        # vehicles.
        mask = np.logical_or(abs(self.points[:, 0]) > 2.44, abs(self.points[:, 1]) > 1.04)

        labels = np.array(self.data['ObjTag'])
        
        # Set point cloud colors based on object labels.
        self.label_color = LABEL_COLORS[labels]

        self.data = self.data[mask]
        self.points = self.points[mask]
        self.label_color = self.label_color[mask]

        # Put the point cloud in the render queue and the entire data in the
        # save queue.
        self.render_queue.put(self.points)
        self.save_queue.put(self.data)
    
    def render(self):
        '''
        Render point cloud.
        '''
        if self.frame == 0:
            self._create_visualizer(window_name='Semantic Lidar Point Cloud')

        self._draw_points(self.label_color)
    
    def save(self, path, scene, frame):
        '''
        Save point cloud to file.

        Args:
            path: root directory of the dataset.
            scene: scene number.
            frame: frame number.
        '''
        with open(
            f'{path}/simbev/sweeps/SEG-LIDAR/SimBEV-scene-{scene:04d}-frame-{frame:04d}-SEG-LIDAR.npz',
            'wb'
        ) as f:
            np.savez_compressed(f, data=self.save_queue.get(True, 10.0))


class Radar(BaseSensor):
    '''
    Base radar class that manages the creation and data acquisition of radars.

    Args:
        world: CARLA simulation world.
        sensor_manager: SensorManager instance that the radar belongs to.
        transform: the radar's transform relative to what it is attached to.
        attached: CARLA object the radar is attached to.
        range: maximum range of the radar.
        hfov: horizontal field of view of the radar.
        vfov: vertical field of view of the radar.
        options: dictionary of radar options.
    '''

    def __init__(self, world, sensor_manager, transform, attached, range, hfov, vfov, options):
        self.world = world
        self.sensor_manager = sensor_manager
        self.range = range
        self.hfov = hfov
        self.vfov = vfov
        self.options = options
        self.frame = 0
        self.point_list = o3d.geometry.PointCloud()

        # Create queues for rendering and saving point clouds. Since queues
        # are blocking, this ensures that at each time step point clouds are
        # fully acquired before the code continues.
        self.render_queue = Queue()
        self.save_queue = Queue()

        self._get_radar()
        
        self.radar_bp.set_attribute('range', str(self.range))
        self.radar_bp.set_attribute('horizontal_fov', str(self.hfov))
        self.radar_bp.set_attribute('vertical_fov', str(self.vfov))

        for key in options:
            self.radar_bp.set_attribute(key, options[key])

        self.radar = self.world.spawn_actor(self.radar_bp, transform, attach_to=attached)

        self.radar.listen(self._process_point_cloud)

    def _get_radar(self):
        '''
        Add radar to the SensorManager and get the radar blueprint.
        '''
        self.sensor_manager.add_radar(self)

        self.radar_bp = self.world.get_blueprint_library().find('sensor.other.radar')
    
    def _process_point_cloud(self, point_cloud):
        '''
        Callback function for processing raw point cloud data.

        Args:
            point_cloud: raw point cloud data.
        '''
        # Point cloud data contains depth, altitude, azimuth, and velocity
        # values.
        self.data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')))
        self.data = np.reshape(self.data, (int(self.data.shape[0] / 4), 4))

        self.data = self.data[:, ::-1]
        
        # Flip azimuth values because CARLA uses a left-handed coordinate system.
        self.data[:, 2] *= -1

        # Put the point cloud in both queues.
        self.render_queue.put(self.data)
        self.save_queue.put(self.data)

    def clear_queues(self):
        '''
        Clear the queues to ensure only the latest point cloud is accessible.
        '''
        with self.render_queue.mutex:
            self.render_queue.queue.clear()
            self.render_queue.all_tasks_done.notify_all()
            self.render_queue.unfinished_tasks = 0

        with self.save_queue.mutex:
            self.save_queue.queue.clear()
            self.save_queue.all_tasks_done.notify_all()
            self.save_queue.unfinished_tasks = 0
    
    def _create_visualizer(self, window_name, width=1024, height=1024):
        '''
        Create Open3D visualizer.

        Args:
            window_name: window name for the point cloud visualizer.
            width: visualizer window width in pixels.
            height: visualizer window height in pixels.
        '''
        self.visualizer = o3d.visualization.Visualizer()

        self.visualizer.create_window(window_name=window_name, width=width, height=height, left=0, top=0)
        self.visualizer.get_render_option().point_size = 4.0
        self.visualizer.get_render_option().background_color = [0.04, 0.04, 0.04]
        self.visualizer.get_render_option().show_coordinate_frame = True

        self.visualizer.add_geometry(self.point_list)
    
    def _draw_points(self, color):
        '''
        Visualize point cloud data in Open3D.

        Args:
            color: point cloud colors.
        '''
        self.point_list.points = o3d.utility.Vector3dVector(self.points)
        self.point_list.colors = o3d.utility.Vector3dVector(color)
        
        if self.frame == 2:
            self.visualizer.add_geometry(self.point_list)

            # Place the camera at a height where the entire point cloud is
            # visible. The camera's field of view is 60 degrees by default.
            camera_height = np.sqrt(3.0) * self.range * max(0.5, np.sin(np.deg2rad(self.hfov / 2)))

            camera = self.visualizer.get_view_control().convert_to_pinhole_camera_parameters()

            pose = np.eye(4)
            
            pose[1, 1] = -1
            pose[2, 2] = -1
            pose[:3, 3] = [-camera_height / np.sqrt(3.0), 0, camera_height]

            camera.extrinsic = pose

            self.visualizer.get_view_control().convert_from_pinhole_camera_parameters(camera)
        
        self.visualizer.update_geometry(self.point_list)
        self.visualizer.poll_events()
        self.visualizer.update_renderer()
        
        self.frame += 1
        
        time.sleep(0.005)
    
    def render(self, window_name='Radar Point Cloud'):
        '''
        Render point cloud.

        Args:
            window_name: window name for the point cloud visualizer.
        '''
        if self.frame == 0:
            self._create_visualizer(window_name=window_name)

        radar_data = self.render_queue.get(True, 10.0)
        
        points = radar_data[:, :-1]

        # Convert the radar values of depth, altitude angle, and azimuth angle
        # to x, y, and z coordinates.
        x = points[:, 0] * np.cos(points[:, 1]) * np.cos(points[:, 2])
        y = points[:, 0] * np.cos(points[:, 1]) * np.sin(points[:, 2])
        z = points[:, 0] * np.sin(points[:, 1])

        self.points = np.array([x, y, z]).T
        
        # Generate point cloud colors based on velocity values.
        velocity = np.abs(radar_data[:, -1])
        velocity_log = np.log(1.0 + velocity)
        velocity_log_normalized = (
            velocity_log - velocity_log.min()
        ) / (
            velocity_log.max() - velocity_log.min() + 1e-6
        )
        velocity_color = np.c_[
            np.interp(velocity_log_normalized, RANGE, RAINBOW[:, 0]),
            np.interp(velocity_log_normalized, RANGE, RAINBOW[:, 1]),
            np.interp(velocity_log_normalized, RANGE, RAINBOW[:, 2])
        ]
        
        self._draw_points(velocity_color)
    
    def save(self, radar_name, path, scene, frame):
        '''
        Save point cloud to file.

        Args:
            radar_name: name of the radar.
            path: root directory of the dataset.
            scene: scene number.
            frame: frame number.
        '''
        with open(
            f'{path}/simbev/sweeps/{radar_name}/SimBEV-scene-{scene:04d}-frame-{frame:04d}-{radar_name}.npz',
            'wb'
        ) as f:
            np.savez_compressed(f, data=self.save_queue.get(True, 10.0))
    
    def destroy(self):
        '''
        Destroy the radar.
        '''
        self.radar.destroy()
        
        try:
            self.visualizer.destroy_window()
        except AttributeError:
            pass


class GNSS(BaseSensor):
    '''
    Base GNSS class that manages the creation and data acquisition of GNSS
    sensors.

    Args:
        world: CARLA simulation world.
        sensor_manager: SensorManager instance that the GNSS belongs to.
        transform: the GNSS' transform relative to what it is attached to.
        attached: CARLA object the GNSS is attached to.
        options: dictionary of GNSS options.
    '''

    def __init__(self, world, sensor_manager, transform, attached, options):
        self.world = world
        self.sensor_manager = sensor_manager
        self.options = options

        # Create a queue for saving GNSS data. Since queues are blocking, this
        # ensures that at each time step GNSS data is fully acquired before
        # the code continues.
        self.save_queue = Queue()

        self._get_gnss()

        for key in options:
            self.gnss_bp.set_attribute(key, options[key])
        
        self.gnss = self.world.spawn_actor(self.gnss_bp, transform, attach_to=attached)

        self.gnss.listen(self._process_data)

    def _get_gnss(self):
        '''
        Add GNSS to the SensorManager and get the GNSS blueprint.
        '''
        self.sensor_manager.add_gnss(self)

        self.gnss_bp = self.world.get_blueprint_library().find('sensor.other.gnss')
    
    def _process_data(self, data):
        '''
        Callback function for processing GNSS data.

        Args:
            data: GNSS data.
        '''
        # GNSS data contains latitude, longitude, and altitude values.
        self.data = np.array([data.latitude, data.longitude, data.altitude])

        # Put the GNSS data in the save queue.
        self.save_queue.put(self.data)
    
    def clear_queue(self):
        '''
        Clear the queue to ensure only the latest GNSS data is accessible.
        '''
        with self.save_queue.mutex:
            self.save_queue.queue.clear()
            self.save_queue.all_tasks_done.notify_all()
            self.save_queue.unfinished_tasks = 0
    
    def save(self, path, scene, frame):
        '''
        Save GNSS data to file.

        Args:
            path: root directory of the dataset.
            scene: scene number.
            frame: frame number.
        '''
        with open(f'{path}/simbev/sweeps/GNSS/SimBEV-scene-{scene:04d}-frame-{frame:04d}-GNSS.bin', 'wb') as f:
            np.save(f, self.save_queue.get(True, 10.0))
    
    def destroy(self):
        '''
        Destroy the GNSS.
        '''
        self.gnss.destroy()


class IMU(BaseSensor):
    '''
    Base IMU class that manages the creation and data acquisition of IMU
    sensors.

    Args:
        world: CARLA simulation world.
        sensor_manager: SensorManager instance that the IMU belongs to.
        transform: the IMU's transform relative to what it is attached to.
        attached: CARLA object the IMU is attached to.
        options: dictionary of IMU options.
    '''

    def __init__(self, world, sensor_manager, transform, attached, options):
        self.world = world
        self.sensor_manager = sensor_manager
        self.options = options

        # Create a queue for saving IMU data. Since queues are blocking, this
        # ensures that at each time step IMU data is fully acquired before the
        # code continues.
        self.save_queue = Queue()

        self._get_imu()

        for key in options:
            self.imu_bp.set_attribute(key, options[key])
        
        self.imu = self.world.spawn_actor(self.imu_bp, transform, attach_to=attached)

        self.imu.listen(self._process_data)

    def _get_imu(self):
        '''
        Add IMU to the SensorManager and get the IMU blueprint.
        '''
        self.sensor_manager.add_imu(self)

        self.imu_bp = self.world.get_blueprint_library().find('sensor.other.imu')
    
    def _process_data(self, data):
        '''
        Callback function for processing IMU data.

        Args:
            data: IMU data.
        '''
        # IMU data contains accelerometer, gyroscope, and compass values. Some
        # values are negated because CARLA uses a left-handed coordinate
        # system.
        self.data = np.array([
            data.accelerometer.x,
            -data.accelerometer.y,
            data.accelerometer.z,
            data.gyroscope.x,
            -data.gyroscope.y,
            -data.gyroscope.z,
            data.compass
        ])

        # Put the IMU data in the save queue.
        self.save_queue.put(self.data)
    
    def clear_queue(self):
        '''
        Clear the queue to ensure only the latest IMU data is accessible.
        '''
        with self.save_queue.mutex:
            self.save_queue.queue.clear()
            self.save_queue.all_tasks_done.notify_all()
            self.save_queue.unfinished_tasks = 0
    
    def save(self, path, scene, frame):
        '''
        Save IMU data to file.

        Args:
            path: root directory of the dataset.
            scene: scene number.
            frame: frame number.
        '''
        with open(f'{path}/simbev/sweeps/IMU/SimBEV-scene-{scene:04d}-frame-{frame:04d}-IMU.bin', 'wb') as f:
            np.save(f, self.save_queue.get(True, 10.0))
    
    def destroy(self):
        '''
        Destroy the IMU.
        '''
        self.imu.destroy()


class SemanticBEVCamera(SemanticCamera):
    '''
    BEV semantic segmentation camera class that manages the creation and data
    acquisition of BEV semantic segmentation cameras.
    '''

    def __init__(self, world, sensor_manager, transform, attached, width, height, options):
        super().__init__(world, sensor_manager, transform, attached, width, height, options)

    def _get_camera(self):
        '''
        Add BEV semantic segmentation camera to the SensorManager and get the
        camera blueprint.
        '''
        self.sensor_manager.add_semantic_bev_camera(self)

        self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
    
    def render(self, window_name='Segmented BEV Image'):
        '''
        Render BEV semantic segmentation image.

        Args:
            window_name: window name for the rendered image.
        '''
        cv2.imshow(window_name, self.render_queue.get(True, 10.0))
        cv2.waitKey(1)