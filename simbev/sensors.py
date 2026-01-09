# Academic Software License: Copyright © 2026 Goodarz Mehr.

'''
SimBEV perception and navigation sensors.
'''

import cv2
import time
import carla

import numpy as np
import open3d as o3d

from queue import Queue

from matplotlib import colormaps as cm

try:
    from .utils import flow_to_color

except ImportError:
    from utils import flow_to_color

import pyvista as pv

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
    (110, 110, 110), # Rock
    (255, 165, 0),   # TrafficCone
    (200, 128, 128)  # Barrier
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
    def __init__(self, world: carla.World, sensor_manager, transform: carla.Transform, attached: carla.Actor):
        self._world = world
        self._sensor_manager = sensor_manager
    
    def render(self):
        '''Render sensor data.'''
        raise NotImplementedError
    
    def save(self):
        '''Save sensor data to file.'''
        raise NotImplementedError

    def destroy(self):
        '''Destroy the sensor.'''
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
    def __init__(
        self,
        world: carla.World,
        sensor_manager,
        transform: carla.Transform,
        attached: carla.Actor,
        width: int,
        height: int,
        options: dict
    ):
        self._world = world
        self._sensor_manager = sensor_manager
        self._width = width
        self._height = height
        self._options = options
        self._image = None

        # Create queues for rendering and saving images. Since queues are
        # blocking, this ensures that at each time step images are fully
        # acquired before the code continues.
        self._render_queue = Queue()
        self._save_queue = Queue()

        self._get_camera()
        
        self._camera_bp.set_attribute('image_size_x', str(self._width))
        self._camera_bp.set_attribute('image_size_y', str(self._height))

        for key in options:
            self._camera_bp.set_attribute(key, options[key])

        self._camera = self._world.spawn_actor(self._camera_bp, transform, attach_to=attached)

        self._camera.listen(self._process_image)

    def _get_camera(self):
        '''Add camera to the SensorManager and get the camera blueprint.'''
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
        self._image = array[:, :, :3]

        # Put the image in both queues.
        self._render_queue.put(self._image)
        self._save_queue.put(self._image)
    
    def clear_queues(self):
        '''Clear the queues to ensure only the latest image is accessible.'''
        with self._render_queue.mutex:
            self._render_queue.queue.clear()
            self._render_queue.unfinished_tasks = 0
            self._render_queue.all_tasks_done.notify_all()

        with self._save_queue.mutex:
            self._save_queue.queue.clear()
            self._save_queue.unfinished_tasks = 0
            self._save_queue.all_tasks_done.notify_all()
    
    def destroy(self):
        '''Destroy the camera.'''
        self._camera.destroy()


class RGBCamera(BaseCamera):
    '''
    RGB camera class that manages the creation and data acquisition of RGB
    cameras.
    '''
    def __init__(
        self,
        world: carla.World,
        sensor_manager,
        transform: carla.Transform,
        attached: carla.Actor,
        width: int,
        height: int,
        options: dict
    ):
        super().__init__(world, sensor_manager, transform, attached, width, height, options)

    def _get_camera(self):
        '''
        Add RGB camera to the SensorManager and get the camera blueprint.
        '''
        self._sensor_manager.add_sensor(self, 'rgb_camera')

        self._camera_bp = self._world.get_blueprint_library().find('sensor.camera.rgb')
    
    def render(self, window_name: str = 'RGB Image'):
        '''
        Render RGB image.

        Args:
            window_name: window name for the rendered image.
        '''
        cv2.imshow(window_name, cv2.resize(self._render_queue.get(True, 10.0), (self._width // 4, self._height // 4)))
        cv2.waitKey(1)

    def save(self, camera_name: str, path: str, scene: int, frame: int):
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
            self._save_queue.get(True, 10.0),
            [cv2.IMWRITE_JPEG_QUALITY, 80]
        )


class SemanticCamera(BaseCamera):
    '''
    Semantic segmentation camera class that manages the creation and data
    acquisition of semantic segmentation cameras.
    '''
    def __init__(
        self,
        world: carla.World,
        sensor_manager,
        transform: carla.Transform,
        attached: carla.Actor,
        width: int,
        height: int,
        options: dict
    ):
        super().__init__(world, sensor_manager, transform, attached, width, height, options)
    
    def _get_camera(self):
        '''
        Add semantic segmentation camera to the SensorManager and get the
        camera blueprint.
        '''
        self._sensor_manager.add_sensor(self, 'semantic_camera')

        self._camera_bp = self._world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
    
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
        self._image = array[:, :, :3]

        # Put the image in both queues.
        self._render_queue.put(self._image)
        self._save_queue.put(self._image)
    
    def render(self, window_name: str = 'Segmented Image'):
        '''
        Render semantic segmentation image.

        Args:
            window_name: window name for the rendered image.
        '''
        cv2.imshow(window_name, cv2.resize(self._render_queue.get(True, 10.0), (self._width // 4, self._height // 4)))
        cv2.waitKey(1)
    
    def save(self, camera_name: str, path: str, scene: int, frame: int):
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
            self._save_queue.get(True, 10.0)
        )


class InstanceCamera(BaseCamera):
    '''
    Instance segmentation camera class that manages the creation and data
    acquisition of instance segmentation cameras.
    '''
    def __init__(
        self,
        world: carla.World,
        sensor_manager,
        transform: carla.Transform,
        attached: carla.Actor,
        width: int,
        height: int,
        options: dict
    ):
        super().__init__(world, sensor_manager, transform, attached, width, height, options)
    
    def _get_camera(self):
        '''
        Add instance segmentation camera to the SensorManager and get the
        camera blueprint.
        '''
        self._sensor_manager.add_sensor(self, 'instance_camera')

        self._camera_bp = self._world.get_blueprint_library().find('sensor.camera.instance_segmentation')
    
    def render(self, window_name: str = 'Instance Image'):
        '''
        Render instance segmentation image.

        Args:
            window_name: window name for the rendered image.
        '''
        cv2.imshow(window_name, cv2.resize(self._render_queue.get(True, 10.0), (self._width // 4, self._height // 4)))
        cv2.waitKey(1)
    
    def save(self, camera_name: str, path: str, scene: int, frame: int):
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
            self._save_queue.get(True, 10.0)
        )


class DepthCamera(BaseCamera):
    '''
    Depth camera class that manages the creation and data acquisition of depth
    cameras.
    '''
    def __init__(
        self,
        world: carla.World,
        sensor_manager,
        transform: carla.Transform,
        attached: carla.Actor,
        width: int,
        height: int,
        options: dict
    ):
        super().__init__(world, sensor_manager, transform, attached, width, height, options)
    
    def _get_camera(self):
        '''
        Add depth camera to the SensorManager and get the camera blueprint.
        '''
        self._sensor_manager.add_sensor(self, 'depth_camera')

        self._camera_bp = self._world.get_blueprint_library().find('sensor.camera.depth')
    
    def render(self, window_name: str = 'Depth Image'):
        '''
        Render depth image.

        Args:
            window_name: window name for the rendered image.
        '''
        image = self._render_queue.get(True, 10.0)

        # 1000 * (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1)
        # provides the depth value in meters. This is then converted to a
        # logarithmic scale for better visualization.
        normalized_distance = (image[:, :, 2] + image[:, :, 1] * 256.0 + image[:, :, 0] * 256.0 * 256.0) \
            / (256.0 * 256.0 * 256.0 - 1)
        
        log_distance = 255 * np.log(256.0 * normalized_distance + 1) / np.log(257.0)
        
        cv2.imshow(window_name, cv2.resize(log_distance.astype(np.uint8), (self._width // 4, self._height // 4)))
        cv2.waitKey(1)
    
    def save(self, camera_name: str, path: str, scene: int, frame: int):
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
            self._save_queue.get(True, 10.0)
        )


class FlowCamera(BaseCamera):
    '''
    Flow camera class that manages the creation and data acquisition of flow
    cameras.
    '''
    def __init__(
        self,
        world: carla.World,
        sensor_manager,
        transform: carla.Transform,
        attached: carla.Actor,
        width: int,
        height: int,
        options: dict
    ):
        super().__init__(world, sensor_manager, transform, attached, width, height, options)
    
    def _get_camera(self):
        '''
        Add flow camera to the SensorManager and get the camera blueprint.
        '''
        self._sensor_manager.add_sensor(self, 'flow_camera')

        self._camera_bp = self._world.get_blueprint_library().find('sensor.camera.optical_flow')
    
    def _process_image(self, image):
        '''
        Callback function for processing raw image data.

        Args:
            image: raw image data.
        '''
        # Reshape into a (height, width, 2) NumPy array.
        array = np.frombuffer(image.raw_data, dtype=np.float32)
        self._image = np.reshape(array, (image.height, image.width, 2))

        # Put the image in both queues.
        self._render_queue.put(self._image)
        self._save_queue.put(self._image)
    
    def render(self, window_name: str = 'Flow Image'):
        '''
        Render flow image.

        Args:
            window_name: window name for the rendered image.
        '''
        image = flow_to_color(self._render_queue.get(True, 10.0))
        
        cv2.imshow(window_name, cv2.resize(image, (self._width // 4, self._height // 4)))
        cv2.waitKey(1)
    
    def save(self, camera_name: str, path: str, scene: int, frame: int):
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
            np.savez_compressed(f, data=self._save_queue.get(True, 10.0))


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
    def __init__(
        self,
        world: carla.World,
        sensor_manager,
        transform: carla.Transform,
        attached: carla.Actor,
        channels: int,
        range: float,
        options: dict
    ):
        self._world = world
        self._sensor_manager = sensor_manager
        self._channels = channels
        self._range = range
        self._options = options
        self._frame = 0
        self._point_list = o3d.geometry.PointCloud()

        # Create queues for rendering and saving point clouds. Since queues
        # are blocking, this ensures that at each time step point clouds are
        # fully acquired before the code continues.
        self._render_queue = Queue()
        self._save_queue = Queue()

        self._get_lidar()
        
        self._lidar_bp.set_attribute('channels', str(self._channels))
        self._lidar_bp.set_attribute('range', str(self._range))

        for key in options:
            self._lidar_bp.set_attribute(key, options[key])

        self._lidar = self._world.spawn_actor(self._lidar_bp, transform, attach_to=attached)

        self._lidar.listen(self._process_point_cloud)

    def _get_lidar(self):
        '''Add lidar to the SensorManager and get the lidar blueprint.'''
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
        with self._render_queue.mutex:
            self._render_queue.queue.clear()
            self._render_queue.unfinished_tasks = 0
            self._render_queue.all_tasks_done.notify_all()

        with self._save_queue.mutex:
            self._save_queue.queue.clear()
            self._save_queue.unfinished_tasks = 0
            self._save_queue.all_tasks_done.notify_all()
    
    def _create_visualizer(self, window_name: str, width: int = 1024, height: int = 1024):
        '''
        Create Open3D visualizer.

        Args:
            window_name: window name for the point cloud visualizer.
            width: visualizer window width in pixels.
            height: visualizer window height in pixels.
        '''
        self._visualizer = o3d.visualization.Visualizer()

        self._visualizer.create_window(window_name=window_name, width=width, height=height, left=0, top=0)
        
        self._visualizer.get_render_option().point_size = 1.0
        if self._channels == 27: print('Copyright © 2026 Goodarz Mehr')
        self._visualizer.get_render_option().background_color = [0.04, 0.04, 0.04]
        self._visualizer.get_render_option().show_coordinate_frame = True

        self._visualizer.add_geometry(self._point_list)
    
    def _draw_points(self, color: np.ndarray):
        '''
        Visualize point cloud data in Open3D.

        Args:
            color: point cloud colors.
        '''
        self._point_list.points = o3d.utility.Vector3dVector(self._render_queue.get(True, 10.0))
        self._point_list.colors = o3d.utility.Vector3dVector(color)
        
        if self._frame == 2:
            self._visualizer.add_geometry(self._point_list)

            # Place the camera at a height where the entire point cloud is
            # visible. The camera's field of view is 60 degrees by default.
            camera_height = np.sqrt(3.0) * self._range

            camera = self._visualizer.get_view_control().convert_to_pinhole_camera_parameters()

            pose = np.eye(4)

            pose[1, 1] = -1
            pose[2, 2] = -1
            pose[:3, 3] = [0, 0, camera_height]

            camera.extrinsic = pose

            self._visualizer.get_view_control().convert_from_pinhole_camera_parameters(camera)
        
        self._visualizer.update_geometry(self._point_list)
        self._visualizer.poll_events()
        self._visualizer.update_renderer()
        
        self._frame += 1
        
        time.sleep(0.005)
    
    def destroy(self):
        '''Destroy the lidar.'''
        self._lidar.destroy()
        
        try:
            self._visualizer.destroy_window()
        except AttributeError:
            pass


class Lidar(BaseLidar):
    '''
    Lidar class that manages the creation and data acquisition of lidars.
    '''
    def __init__(
        self,
        world: carla.World,
        sensor_manager,
        transform: carla.Transform,
        attached: carla.Actor,
        channels: int,
        range: float,
        options: dict
    ):
        super().__init__(world, sensor_manager, transform, attached, channels, range, options)

    def _get_lidar(self):
        '''Add lidar to the SensorManager and get the lidar blueprint.'''
        self._sensor_manager.add_sensor(self, 'lidar')

        self._lidar_bp = self._world.get_blueprint_library().find('sensor.lidar.ray_cast')
    
    def _process_point_cloud(self, point_cloud):
        '''
        Callback function for processing raw point cloud data.

        Args:
            point_cloud: raw point cloud data.
        '''
        # Point cloud data contains x, y, z, and intensity values.
        self._data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')))
        self._data = np.reshape(self._data, (int(self._data.shape[0] / 4), 4))

        # Select the x, y, and z values and flip y data because CARLA uses a
        # left-handed coordinate system.
        self._points = self._data[:, :-1]
        self._points[:, 1] *= -1

        # Remove the points belonging to the ego vehicle. The limits are based
        # on the 2016 Mustang dimensions and should be adjusted for other
        # vehicles.
        mask = np.logical_or(abs(self._points[:, 0]) > 2.44, abs(self._points[:, 1]) > 1.04)

        self._data = self._data[mask]
        self._points = self._points[mask]

        # Put the point cloud in both queues.
        self._render_queue.put(self._points)
        self._save_queue.put(self._points)
    
    def render(self):
        '''Render point cloud.'''
        if self._frame == 0:
            self._create_visualizer(window_name='Lidar Point Cloud')

        # Generate point cloud colors based on intensity values.
        distance = np.linalg.norm(self._points, axis=1)
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
    
    def save(self, path: str, scene: int, frame: int):
        '''
        Save point cloud to file.

        Args:
            path: root directory of the dataset.
            scene: scene number.
            frame: frame number.
        '''
        with open(f'{path}/simbev/sweeps/LIDAR/SimBEV-scene-{scene:04d}-frame-{frame:04d}-LIDAR.npz', 'wb') as f:
            np.savez_compressed(f, data=self._save_queue.get(True, 10.0))


class SemanticLidar(BaseLidar):
    '''
    Semantic lidar class that manages the creation and data acquisition of
    semantic lidars.
    '''
    def __init__(
        self,
        world: carla.World,
        sensor_manager,
        transform: carla.Transform,
        attached: carla.Actor,
        channels: int,
        range: float,
        options: dict
    ):
        super().__init__(world, sensor_manager, transform, attached, channels, range, options)

    def _get_lidar(self):
        '''
        Add semantic lidar to the SensorManager and get the semantic lidar
        blueprint.
        '''
        self._sensor_manager.add_sensor(self, 'semantic_lidar')

        self._lidar_bp = self._world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
    
    def _process_point_cloud(self, point_cloud):
        '''
        Callback function for processing raw point cloud data.

        Args:
            point_cloud: raw point cloud data.
        '''
        # Point cloud data contains x, y, z, cosine angle, object index, and
        # object tag values.
        self._data = np.copy(
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
        self._data['y'] *= -1
        self._points = np.array([self._data['x'], self._data['y'], self._data['z']]).T

        # Remove the points belonging to the ego vehicle. The limits are based
        # on the 2016 Mustang dimensions and should be adjusted for other
        # vehicles.
        mask = np.logical_or(abs(self._points[:, 0]) > 2.44, abs(self._points[:, 1]) > 1.04)

        labels = np.array(self._data['ObjTag'])
        
        # Set point cloud colors based on object labels.
        self._label_color = LABEL_COLORS[labels]

        self._data = self._data[mask]
        self._points = self._points[mask]
        self._label_color = self._label_color[mask]

        # Put the point cloud in the render queue and the entire data in the
        # save queue.
        self._render_queue.put(self._points)
        self._save_queue.put(self._data)
    
    def render(self):
        '''Render point cloud.'''
        if self._frame == 0:
            self._create_visualizer(window_name='Semantic Lidar Point Cloud')

        self._draw_points(self._label_color)
    
    def save(self, path: str, scene: int, frame: int):
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
            np.savez_compressed(f, data=self._save_queue.get(True, 10.0))


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

    def __init__(
        self,
        world: carla.World,
        sensor_manager,
        transform: carla.Transform,
        attached: carla.Actor,
        range: float,
        hfov: float,
        vfov: float,
        options: dict
    ):
        self._world = world
        self._sensor_manager = sensor_manager
        self._range = range
        self._hfov = hfov
        self._vfov = vfov
        self._options = options
        self._frame = 0
        self._point_list = o3d.geometry.PointCloud()

        # Create queues for rendering and saving point clouds. Since queues
        # are blocking, this ensures that at each time step point clouds are
        # fully acquired before the code continues.
        self._render_queue = Queue()
        self._save_queue = Queue()

        self._get_radar()
        
        self._radar_bp.set_attribute('range', str(self._range))
        self._radar_bp.set_attribute('horizontal_fov', str(self._hfov))
        self._radar_bp.set_attribute('vertical_fov', str(self._vfov))

        for key in options:
            self._radar_bp.set_attribute(key, options[key])

        self._radar = self._world.spawn_actor(self._radar_bp, transform, attach_to=attached)

        self._radar.listen(self._process_point_cloud)

    def _get_radar(self):
        '''Add radar to the SensorManager and get the radar blueprint.'''
        self._sensor_manager.add_sensor(self, 'radar')

        self._radar_bp = self._world.get_blueprint_library().find('sensor.other.radar')
    
    def _process_point_cloud(self, point_cloud):
        '''
        Callback function for processing raw point cloud data.

        Args:
            point_cloud: raw point cloud data.
        '''
        # Point cloud data contains depth, altitude, azimuth, and velocity
        # values.
        self._data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')))
        self._data = np.reshape(self._data, (int(self._data.shape[0] / 4), 4))

        self._data = self._data[:, ::-1]
        
        # Flip azimuth values because CARLA uses a left-handed coordinate system.
        self._data[:, 2] *= -1

        # Put the point cloud in both queues.
        self._render_queue.put(self._data)
        self._save_queue.put(self._data)

    def clear_queues(self):
        '''
        Clear the queues to ensure only the latest point cloud is accessible.
        '''
        with self._render_queue.mutex:
            self._render_queue.queue.clear()
            self._render_queue.unfinished_tasks = 0
            self._render_queue.all_tasks_done.notify_all()

        with self._save_queue.mutex:
            self._save_queue.queue.clear()
            self._save_queue.unfinished_tasks = 0
            self._save_queue.all_tasks_done.notify_all()
    
    def _create_visualizer(self, window_name: str, width: int = 1024, height: int = 1024):
        '''
        Create Open3D visualizer.

        Args:
            window_name: window name for the point cloud visualizer.
            width: visualizer window width in pixels.
            height: visualizer window height in pixels.
        '''
        self._visualizer = o3d.visualization.Visualizer()

        self._visualizer.create_window(window_name=window_name, width=width, height=height, left=0, top=0)
        
        self._visualizer.get_render_option().point_size = 4.0
        self._visualizer.get_render_option().background_color = [0.04, 0.04, 0.04]
        self._visualizer.get_render_option().show_coordinate_frame = True

        self._visualizer.add_geometry(self._point_list)
    
    def _draw_points(self, color: np.ndarray):
        '''
        Visualize point cloud data in Open3D.

        Args:
            color: point cloud colors.
        '''
        self._point_list.points = o3d.utility.Vector3dVector(self._points)
        self._point_list.colors = o3d.utility.Vector3dVector(color)
        
        if self._frame == 2:
            self._visualizer.add_geometry(self._point_list)

            # Place the camera at a height where the entire point cloud is
            # visible. The camera's field of view is 60 degrees by default.
            camera_height = np.sqrt(3.0) * self._range * max(0.5, np.sin(np.deg2rad(self._hfov / 2)))

            camera = self._visualizer.get_view_control().convert_to_pinhole_camera_parameters()

            pose = np.eye(4)
            
            pose[1, 1] = -1
            pose[2, 2] = -1
            pose[:3, 3] = [-camera_height / np.sqrt(3.0), 0, camera_height]

            camera.extrinsic = pose

            self._visualizer.get_view_control().convert_from_pinhole_camera_parameters(camera)
        
        self._visualizer.update_geometry(self._point_list)
        self._visualizer.poll_events()
        self._visualizer.update_renderer()
        
        self._frame += 1
        
        time.sleep(0.005)
    
    def render(self, window_name: str = 'Radar Point Cloud'):
        '''
        Render point cloud.

        Args:
            window_name: window name for the point cloud visualizer.
        '''
        if self._frame == 0:
            self._create_visualizer(window_name=window_name)

        radar_data = self._render_queue.get(True, 10.0)
        
        points = radar_data[:, :-1]

        # Convert the radar values of depth, altitude angle, and azimuth angle
        # to x, y, and z coordinates.
        x = points[:, 0] * np.cos(points[:, 1]) * np.cos(points[:, 2])
        y = points[:, 0] * np.cos(points[:, 1]) * np.sin(points[:, 2])
        z = points[:, 0] * np.sin(points[:, 1])

        self._points = np.array([x, y, z]).T
        
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
    
    def save(self, radar_name: str, path: str, scene: int, frame: int):
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
            np.savez_compressed(f, data=self._save_queue.get(True, 10.0))
    
    def destroy(self):
        '''Destroy the radar.'''
        self._radar.destroy()
        
        try:
            self._visualizer.destroy_window()
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
    def __init__(
        self,
        world: carla.World,
        sensor_manager,
        transform: carla.Transform,
        attached: carla.Actor,
        options: dict
    ):
        self._world = world
        self._sensor_manager = sensor_manager
        self._options = options

        # Create a queue for saving GNSS data. Since queues are blocking, this
        # ensures that at each time step GNSS data is fully acquired before
        # the code continues.
        self._save_queue = Queue()

        self._get_gnss()

        for key in options:
            self._gnss_bp.set_attribute(key, options[key])
        
        self._gnss = self._world.spawn_actor(self._gnss_bp, transform, attach_to=attached)

        self._gnss.listen(self._process_data)

    def _get_gnss(self):
        '''Add GNSS to the SensorManager and get the GNSS blueprint.'''
        self._sensor_manager.add_sensor(self, 'gnss')

        self._gnss_bp = self._world.get_blueprint_library().find('sensor.other.gnss')
    
    def _process_data(self, data):
        '''
        Callback function for processing GNSS data.

        Args:
            data: GNSS data.
        '''
        # GNSS data contains latitude, longitude, and altitude values.
        self._data = np.array([data.latitude, data.longitude, data.altitude])

        # Put the GNSS data in the save queue.
        self._save_queue.put(self._data)
    
    def clear_queues(self):
        '''
        Clear the queue to ensure only the latest GNSS data is accessible.
        '''
        with self._save_queue.mutex:
            self._save_queue.queue.clear()
            self._save_queue.unfinished_tasks = 0
            self._save_queue.all_tasks_done.notify_all()
    
    def save(self, path: str, scene: int, frame: int):
        '''
        Save GNSS data to file.

        Args:
            path: root directory of the dataset.
            scene: scene number.
            frame: frame number.
        '''
        with open(f'{path}/simbev/sweeps/GNSS/SimBEV-scene-{scene:04d}-frame-{frame:04d}-GNSS.bin', 'wb') as f:
            np.save(f, self._save_queue.get(True, 10.0))
    
    def destroy(self):
        '''Destroy the GNSS.'''
        self._gnss.destroy()


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
    def __init__(
        self,
        world: carla.World,
        sensor_manager,
        transform: carla.Transform,
        attached: carla.Actor,
        options: dict
    ):
        self._world = world
        self._sensor_manager = sensor_manager
        self._options = options

        # Create a queue for saving IMU data. Since queues are blocking, this
        # ensures that at each time step IMU data is fully acquired before the
        # code continues.
        self._save_queue = Queue()

        self._get_imu()

        for key in options:
            self._imu_bp.set_attribute(key, options[key])
        
        self._imu = self._world.spawn_actor(self._imu_bp, transform, attach_to=attached)

        self._imu.listen(self._process_data)

    def _get_imu(self):
        '''Add IMU to the SensorManager and get the IMU blueprint.'''
        self._sensor_manager.add_sensor(self, 'imu')

        self._imu_bp = self._world.get_blueprint_library().find('sensor.other.imu')
    
    def _process_data(self, data):
        '''
        Callback function for processing IMU data.

        Args:
            data: IMU data.
        '''
        # IMU data contains accelerometer, gyroscope, and compass values. Some
        # values are negated because CARLA uses a left-handed coordinate
        # system.
        self._data = np.array([
            data.accelerometer.x,
            -data.accelerometer.y,
            data.accelerometer.z,
            data.gyroscope.x,
            -data.gyroscope.y,
            -data.gyroscope.z,
            data.compass
        ])

        # Put the IMU data in the save queue.
        self._save_queue.put(self._data)
    
    def clear_queues(self):
        '''
        Clear the queue to ensure only the latest IMU data is accessible.
        '''
        with self._save_queue.mutex:
            self._save_queue.queue.clear()
            self._save_queue.unfinished_tasks = 0
            self._save_queue.all_tasks_done.notify_all()
    
    def save(self, path: str, scene: int, frame: int):
        '''
        Save IMU data to file.

        Args:
            path: root directory of the dataset.
            scene: scene number.
            frame: frame number.
        '''
        with open(f'{path}/simbev/sweeps/IMU/SimBEV-scene-{scene:04d}-frame-{frame:04d}-IMU.bin', 'wb') as f:
            np.save(f, self._save_queue.get(True, 10.0))
    
    def destroy(self):
        '''Destroy the IMU.'''
        self._imu.destroy()


class SemanticBEVCamera(SemanticCamera):
    '''
    BEV semantic segmentation camera class that manages the creation and data
    acquisition of BEV semantic segmentation cameras.
    '''
    def __init__(
        self,
        world: carla.World,
        sensor_manager,
        transform: carla.Transform,
        attached: carla.Actor,
        width: int,
        height: int,
        options: dict
    ):
        super().__init__(world, sensor_manager, transform, attached, width, height, options)

    def _get_camera(self):
        '''
        Add BEV semantic segmentation camera to the SensorManager and get the
        camera blueprint.
        '''
        self._sensor_manager.add_sensor(self, 'semantic_bev_camera')

        self._camera_bp = self._world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
    
    def get_save_queue(self):
        '''Get the save queue.'''
        return self._save_queue
    
    def render(self, window_name: str = 'Segmented BEV Image'):
        '''
        Render BEV semantic segmentation image.

        Args:
            window_name: window name for the rendered image.
        '''
        cv2.imshow(window_name, self._render_queue.get(True, 10.0))
        cv2.waitKey(1)

class VoxelDetector(BaseSensor):
    '''
    Voxel detector class that manages the creation and data acquisition of
    voxel detectors.

    Args:
        world: CARLA simulation world.
        sensor_manager: SensorManager instance that the voxel detector belongs
            to.
        transform: the voxel detector's transform relative to what it is
            attached to.
        attached: CARLA object the voxel detector is attached to.
        range: maximum range of the detection area.
        voxel_size: size of each voxel.
        upper_limit: upper limit of the detection area.
        lower_limit: lower limit of the detection area.
        options: dictionary of voxel detector options.
    '''
    def __init__(
        self,
        world: carla.World,
        sensor_manager,
        transform: carla.Transform,
        attached: carla.Actor,
        range: float,
        voxel_size: float,
        upper_limit: float,
        lower_limit: float,
        options: dict
    ):
        self._world = world
        self._sensor_manager = sensor_manager
        self._range = range
        self._voxel_size = voxel_size
        self._upper_limit = upper_limit
        self._lower_limit = lower_limit
        self._options = options
        self._frame = 0
        self._voxel_grid = None

        # Calculate voxel grid dimensions.
        self._dim_x = int(round(2 * self._range / self._voxel_size))
        self._dim_y = int(round(2 * self._range / self._voxel_size))
        self._dim_z = int(round((self._upper_limit - self._lower_limit) / self._voxel_size))

        # Create queues for rendering and saving voxel grids. Since queues are
        # blocking, this ensures that at each time step voxel grids are fully
        # acquired before the code continues.
        self._render_queue = Queue()
        self._save_queue = Queue()

        self._get_sensor()
        
        self._voxel_detector_bp.set_attribute('range', str(self._range))
        self._voxel_detector_bp.set_attribute('upper_limit', str(self._upper_limit))
        self._voxel_detector_bp.set_attribute('lower_limit', str(self._lower_limit))
        self._voxel_detector_bp.set_attribute('voxel_size', str(self._voxel_size))

        for key in options:
            self._voxel_detector_bp.set_attribute(key, options[key])

        self._sensor = self._world.spawn_actor(self._voxel_detector_bp, transform, attach_to=attached)

        self._sensor.listen(self._process_voxel_data)

    def _get_sensor(self):
        '''
        Add voxel detector to the SensorManager and get the sensor blueprint.
        '''
        self._sensor_manager.add_sensor(self, 'voxel_detector')

        self._voxel_detector_bp = self._world.get_blueprint_library().find('sensor.other.voxel_detection')
    
    def _process_voxel_data(self, voxel_data):
        '''
        Callback function for processing raw voxel data from the sensor.

        Args:
            voxel_data: raw voxel detection data.
        '''
        # Convert voxel data into a NumPy array and reshape into a 3D voxel
        # grid.
        voxel_array = np.frombuffer(voxel_data.raw_data, dtype=np.uint8)
        
        voxel_grid = voxel_array.reshape((self._dim_x, self._dim_y, self._dim_z))

        voxel_grid = np.flip(voxel_grid, axis=1)

        # Put the voxel grid in both queues.
        self._render_queue.put(voxel_grid)
        self._save_queue.put(voxel_grid)

    def clear_queues(self):
        '''
        Clear the queues to ensure only the latest voxel grid is accessible.
        '''
        with self._render_queue.mutex:
            self._render_queue.queue.clear()
            self._render_queue.unfinished_tasks = 0
            self._render_queue.all_tasks_done.notify_all()

        with self._save_queue.mutex:
            self._save_queue.queue.clear()
            self._save_queue.unfinished_tasks = 0
            self._save_queue.all_tasks_done.notify_all()
    
    def _create_visualizer(self, window_name: str, width: int = 1024, height: int = 1024):
        '''
        Create Open3D visualizer.

        Args:
            window_name: window name for the visualizer.
            width: visualizer window width in pixels.
            height: visualizer window height in pixels.
        '''
        self._visualizer = o3d.visualization.Visualizer()

        self._visualizer.create_window(window_name=window_name, width=width, height=height, left=0, top=0)
        
        self._visualizer.get_render_option().background_color = [0.04, 0.04, 0.04]
        self._visualizer.get_render_option().show_coordinate_frame = True
    
    def _create_voxel_grid(self, voxel_data):
        '''
        Create a PyVista ImageData (UniformGrid) from the voxel data.
        
        Args:
            voxel_data: voxel array with class labels.
        
        Returns:
            grid: voxel grid as a PyVista ImageData object.
        '''
        grid = pv.ImageData()
        
        # Set dimensions (nodes = cells + 1 for each axis).
        grid.dimensions = np.array(voxel_data.shape) + 1
        
        grid.spacing = (self._voxel_size, self._voxel_size, self._voxel_size)
        grid.origin = (-self._range, -self._range, self._lower_limit)
        
        # Add labels as cell data (flattened in Fortran order to match
        # X-fastest indexing)
        grid.cell_data['labels'] = voxel_data.flatten(order='F')
        
        return grid
    
    def render(self, window_name: str = 'Voxel Detector'):
        '''
        Render the voxel grid as colored cubes.
        
        Args:
            window_name: window name for the voxel grid visualizer.
        '''
        if self._frame == 0:
            self._plotter = pv.Plotter(window_size=[1024, 1024], title=window_name)
            
            self._plotter.set_background([0.04, 0.04, 0.04])
            self._plotter.add_axes()
            
            camera_height = 4.0 * self._range
            
            self._plotter.camera_position = [(0, 0, camera_height), (0, 0, 0), (0, 1, 0)]
            
            self._plotter.show(interactive_update=True, auto_close=False)

        voxel_data = self._render_queue.get(True, 10.0)
        
        grid = self._create_voxel_grid(voxel_data)

        occupied = grid.threshold(0.5, scalars='labels')
        
        if occupied.n_cells > 0:
            labels = occupied.cell_data['labels'].astype(int)
            
            colors = LABEL_COLORS[labels]
            
            occupied.cell_data['colors'] = colors
                
            self._plotter.add_mesh(
                occupied, 
                name='voxels',
                scalars='colors', 
                rgb=True, 
                show_scalar_bar=False,
                lighting=True,
                ambient=0.7,
                diffuse=0.6,
                specular=0.0,
                smooth_shading=False,
                reset_camera=False,
            )
        else:
            self._plotter.remove_actor('voxels')
        
        self._plotter.update()
        
        self._frame += 1
        
        time.sleep(0.001)
    
    def save(self, path: str, scene: int, frame: int):
        '''
        Save voxel grid to file.

        Args:
            path: root directory of the dataset.
            scene: scene number.
            frame: frame number.
        '''
        with open(f'{path}/simbev/sweeps/VOXEL-GRID/SimBEV-scene-{scene:04d}-frame-{frame:04d}-VOXEL-GRID.npz', 'wb') as f:
            np.savez_compressed(f, data=self._save_queue.get(True, 10.0))
    
    def destroy(self):
        '''Destroy the voxel detector.'''
        self._sensor.destroy()
        
        try:
            self._plotter.close()
        except AttributeError:
            pass