# Academic Software License: Copyright © 2025 Goodarz Mehr.

'''
Module that collects data from all sensors on a vehicle and
renders or saves them.
'''

import cv2

import numpy as np

from concurrent.futures import ThreadPoolExecutor

from scipy.spatial.transform import Rotation as R

try:
    from .utils import CustomTimer

except ImportError:
    from utils import CustomTimer

class SensorManager:
    '''
    Sensor Manager class that manages data collection.

    Args:
        config: SimBEV configuration.
        vehicle: vehicle the Sensor Manager belongs to.
    '''

    def __init__(self, config, vehicle):
        self._config = config
        self._vehicle = vehicle

        self.sensor_list = {
            'rgb_camera': [],
            'semantic_camera': [],
            'instance_camera': [],
            'depth_camera': [],
            'flow_camera': [],
            'lidar': [],
            'semantic_lidar': [],
            'radar': [],
            'gnss': [],
            'imu': [],
            'semantic_bev_camera': [],
            'voxel_detector': []
        }
        
        self._name_list = {
            'camera': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
            'radar': ['RAD_LEFT', 'RAD_FRONT', 'RAD_RIGHT', 'RAD_BACK'],
            'bev_camera': ['TOP_VIEW', 'BOTTOM_VIEW'],
            'voxel_detector': ['3D Ground Truth']
        }

        self._camera_type_abbrevs = {'rgb': 'RGB', 'semantic': 'SEG', 'instance': 'IST', 'depth': 'DPT', 'flow': 'FLW'}
        
        self._other_sensor_abbrevs = {
            'lidar': 'LIDAR',
            'semantic_lidar': 'SEG-LIDAR',
            'gnss': 'GNSS',
            'imu': 'IMU',
            'voxel_detector': 'VOXEL-GRID'
        }

        self._timer = CustomTimer()
        
        self._data = []

        self._io_executor = ThreadPoolExecutor(
            max_workers=self._config['max_io_workers'],
            thread_name_prefix='sensor_io'
        )

        self._io_futures = []

    def get_data(self):
        '''
        Get the data.

        Returns:
            data: list of dictionaries containing information about the
            collected data.
        '''
        return self._data
    
    def add_sensor(self, sensor, sensor_type):
        '''
        Add sensor to the list of sensors.

        Args:
            sensor: sensor to add to the list of sensors.
            sensor_type: type of the sensor.
        '''
        self.sensor_list[sensor_type].append(sensor)
    
    def clear_queues(self):
        '''Clear sensor queues.'''
        for key in self.sensor_list:
            for sensor in self.sensor_list[key]:
                sensor.clear_queues()
    
    def render(self):
        '''Render sensor data.'''
        for type, abbrev in self._camera_type_abbrevs.items():
            for camera, window_name in zip(self.sensor_list[f'{type}_camera'], self._name_list['camera']):
                camera.render(f'{abbrev}-' + window_name)
        
        for type in ['lidar', 'semantic_lidar']:
            for sensor in self.sensor_list[type]:
                sensor.render()
        
        for radar, window_name in zip(self.sensor_list['radar'], self._name_list['radar']):
            radar.render(window_name)
        
        for voxel_detector, window_name in zip(self.sensor_list['voxel_detector'], self._name_list['voxel_detector']):
            voxel_detector.render(window_name)
        
        if self._config['render_bev_camera_images']:
            for semantic_bev_camera, window_name in zip(
                self.sensor_list['semantic_bev_camera'],
                self._name_list['bev_camera']
            ):
                semantic_bev_camera.render(window_name)
    
    def save(self, path, scene, frame):
        '''
        Save sensor data.

        Args:
            path: root directory of the dataset.
            scene: scene number.
            frame: frame number.
        '''
        # Submit all I/O operations asynchronously.
        for key in self.sensor_list:
            if key in ['rgb_camera', 'semantic_camera', 'instance_camera', 'depth_camera', 'flow_camera']:
                for camera, camera_name in zip(self.sensor_list[key], self._name_list['camera']):
                    self._io_futures.append(self._io_executor.submit(camera.save, camera_name, path, scene, frame))
            elif key in ['radar']:
                for radar, radar_name in zip(self.sensor_list[key], self._name_list['radar']):
                    self._io_futures.append(self._io_executor.submit(radar.save, radar_name, path, scene, frame))
            elif key in ['lidar', 'semantic_lidar', 'gnss', 'imu', 'voxel_detector']:
                for sensor in self.sensor_list[key]:
                    self._io_futures.append(self._io_executor.submit(sensor.save, path, scene, frame))
        
        scene_data = {}

        ego_transform = self._vehicle.get_transform()

        scene_data['ego2global_translation'] = [ego_transform.location.x,
                                                -ego_transform.location.y,
                                                ego_transform.location.z]
        scene_data['ego2global_rotation'] = np.roll(
            R.from_euler(
                'xyz',
                [ego_transform.rotation.roll, -ego_transform.rotation.pitch, -ego_transform.rotation.yaw],
                degrees=True).as_quat(), 1
        ).tolist()
        
        scene_data['timestamp'] = round(self._timer.time() * 10e6)

        for camera_name in self._name_list['camera']:
            for type, abbrev in self._camera_type_abbrevs.items():
                if self._config[f'use_{type}_camera']:
                    scene_data[f'{abbrev}-{camera_name}'] = f'{path}/simbev/sweeps/{abbrev}-{camera_name}' \
                    f'/SimBEV-scene-{scene:04d}-frame-{frame:04d}-{abbrev}-{camera_name}.' + \
                    ('jpg' if type == 'rgb' else 'png' if type in ['semantic', 'instance', 'depth'] else 'npz')

        if self._config['use_radar']:
            for radar_name in self._name_list['radar']:
                scene_data[f'{radar_name}'] = f'{path}/simbev/sweeps/{radar_name}' \
                f'/SimBEV-scene-{scene:04d}-frame-{frame:04d}-{radar_name}.npz'
        
        for type, abbrev in self._other_sensor_abbrevs.items():
            if self._config[f'use_{type}']:
                scene_data[f'{abbrev}'] = f'{path}/simbev/sweeps/{abbrev}' \
                f'/SimBEV-scene-{scene:04d}-frame-{frame:04d}-{abbrev}.' + \
                ('npz' if type in ['lidar', 'semantic_lidar', 'voxel_detector'] else 'bin')
        
        scene_data['GT_SEG'] = f'{path}/simbev/ground-truth/seg/SimBEV-scene-{scene:04d}-frame-{frame:04d}-GT_SEG.npz'
        scene_data['GT_SEG_VIZ'] = f'{path}/simbev/ground-truth/seg_viz' \
        f'/SimBEV-scene-{scene:04d}-frame-{frame:04d}-GT_SEG_VIZ.jpg'
        scene_data['GT_DET'] = f'{path}/simbev/ground-truth/det/SimBEV-scene-{scene:04d}-frame-{frame:04d}-GT_DET.bin'
        scene_data['HD_MAP'] = f'{path}/simbev/ground-truth/hd_map' \
            f'/SimBEV-scene-{scene:04d}-frame-{frame:04d}-HD_MAP.json'

        scene_data['scene'] = scene
        scene_data['frame'] = frame

        self._data.append(scene_data)
    
    def reset(self):
        '''Reset scenario data.'''
        self.wait_for_saves()

        self._data = []
    
    def wait_for_saves(self):
        '''Wait for all pending I/O operations to complete.'''
        for future in self._io_futures:
            future.result()

        self._io_futures.clear()
    
    def destroy(self):
        '''Destroy the sensors.'''
        self.wait_for_saves()  # Ensure all saves are complete
        
        self._io_executor.shutdown(wait=True)
        
        for key in self.sensor_list:
            for sensor in self.sensor_list[key]:
                sensor.destroy()

        cv2.destroyAllWindows()
