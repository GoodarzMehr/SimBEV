# Academic Software License: Copyright © 2025 Goodarz Mehr.

'''
Sensor Manager module that collects data from all sensors on a vehicle and
renders or saves them.
'''

import cv2

import numpy as np

from utils import CustomTimer

from concurrent.futures import ThreadPoolExecutor

from scipy.spatial.transform import Rotation as R

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
            'semantic_bev_camera': []
        }
        
        self._name_list = {
            'camera': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                       'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
            'radar': ['RAD_LEFT', 'RAD_FRONT', 'RAD_RIGHT', 'RAD_BACK'],
            'bev_camera': ['TOP_VIEW', 'BOTTOM_VIEW']
        }

        self._timer = CustomTimer()
        
        self._data = []

        self._io_executor = ThreadPoolExecutor(max_workers=48, thread_name_prefix="sensor_io")
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
        for camera, window_name in zip(self.sensor_list['rgb_camera'], self._name_list['camera']):
            camera.render(window_name)

        for semantic_camera, window_name in zip(self.sensor_list['semantic_camera'], self._name_list['camera']):
            semantic_camera.render('SEG ' + window_name)

        for instance_camera, window_name in zip(self.sensor_list['instance_camera'], self._name_list['camera']):
            instance_camera.render('IST ' + window_name)

        for depth_camera, window_name in zip(self.sensor_list['depth_camera'], self._name_list['camera']):
            depth_camera.render('DPT ' + window_name)

        for flow_camera, window_name in zip(self.sensor_list['flow_camera'], self._name_list['camera']):
            flow_camera.render('FLW ' + window_name)
        
        for lidar in self.sensor_list['lidar']:
            lidar.render()

        for semantic_lidar in self.sensor_list['semantic_lidar']:
            semantic_lidar.render()
        
        for radar, window_name in zip(self.sensor_list['radar'], self._name_list['radar']):
            radar.render(window_name)

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
        # for key in self.sensor_list:
        #     if key in ['rgb_camera', 'semantic_camera', 'instance_camera', 'depth_camera', 'flow_camera']:
        #         for camera, camera_name in zip(self.sensor_list[key], self._name_list['camera']):
        #             camera.save(camera_name, path, scene, frame)
        #     elif key in ['radar']:
        #         for radar, radar_name in zip(self.sensor_list[key], self._name_list['radar']):
        #             radar.save(radar_name, path, scene, frame)
        #     elif key in ['lidar', 'semantic_lidar', 'gnss', 'imu']:
        #         for sensor in self.sensor_list[key]:
        #             sensor.save(path, scene, frame)

        # Submit all I/O operations asynchronously
        for key in self.sensor_list:
            if key in ['rgb_camera', 'semantic_camera', 'instance_camera', 'depth_camera', 'flow_camera']:
                for camera, camera_name in zip(self.sensor_list[key], self._name_list['camera']):
                    self._io_futures.append(
                        self._io_executor.submit(camera.save, camera_name, path, scene, frame)
                    )
            elif key in ['radar']:
                for radar, radar_name in zip(self.sensor_list[key], self._name_list['radar']):
                    self._io_futures.append(
                        self._io_executor.submit(radar.save, radar_name, path, scene, frame)
                    )
            elif key in ['lidar', 'semantic_lidar', 'gnss', 'imu']:
                for sensor in self.sensor_list[key]:
                    self._io_futures.append(
                        self._io_executor.submit(sensor.save, path, scene, frame)
                    )
        
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
            if self._config['use_rgb_camera']:
                scene_data[f'RGB-{camera_name}'] = f'{path}/simbev/sweeps/RGB-{camera_name}' \
                f'/SimBEV-scene-{scene:04d}-frame-{frame:04d}-RGB-{camera_name}.jpg'
            if self._config['use_semantic_camera']:
                scene_data[f'SEG-{camera_name}'] = f'{path}/simbev/sweeps/SEG-{camera_name}' \
                f'/SimBEV-scene-{scene:04d}-frame-{frame:04d}-SEG-{camera_name}.png'
            if self._config['use_instance_camera']:
                scene_data[f'IST-{camera_name}'] = f'{path}/simbev/sweeps/IST-{camera_name}' \
                f'/SimBEV-scene-{scene:04d}-frame-{frame:04d}-IST-{camera_name}.png'
            if self._config['use_depth_camera']:
                scene_data[f'DPT-{camera_name}'] = f'{path}/simbev/sweeps/DPT-{camera_name}' \
                f'/SimBEV-scene-{scene:04d}-frame-{frame:04d}-DPT-{camera_name}.png'
            if self._config['use_flow_camera']:
                scene_data[f'FLW-{camera_name}'] = f'{path}/simbev/sweeps/FLW-{camera_name}' \
                f'/SimBEV-scene-{scene:04d}-frame-{frame:04d}-FLW-{camera_name}.npz'

        if self._config['use_lidar']:
            scene_data['LIDAR'] = f'{path}/simbev/sweeps/LIDAR/SimBEV-scene-{scene:04d}-frame-{frame:04d}-LIDAR.npz'
        if self._config['use_semantic_lidar']:
            scene_data['SEG-LIDAR'] = f'{path}/simbev/sweeps/SEG-LIDAR' \
            f'/SimBEV-scene-{scene:04d}-frame-{frame:04d}-SEG-LIDAR.npz'

        if self._config['use_radar']:
            for radar_name in self._name_list['radar']:
                scene_data[f'{radar_name}'] = f'{path}/simbev/sweeps/{radar_name}' \
                f'/SimBEV-scene-{scene:04d}-frame-{frame:04d}-{radar_name}.npz'

        if self._config['use_gnss']:
            scene_data['GNSS'] = f'{path}/simbev/sweeps/GNSS/SimBEV-scene-{scene:04d}-frame-{frame:04d}-GNSS.bin'
        
        if self._config['use_imu']:
            scene_data['IMU'] = f'{path}/simbev/sweeps/IMU/SimBEV-scene-{scene:04d}-frame-{frame:04d}-IMU.bin'
        
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
        self._data = []
    
    def wait_for_saves(self):
        '''Wait for all pending I/O to complete.'''
        for future in self._io_futures:
            future.result()  # This will raise any exceptions
        self._io_futures.clear()
    
    def destroy(self):
        '''Destroy the sensors.'''
        self.wait_for_saves()  # Ensure all saves complete
        self._io_executor.shutdown(wait=True)
        for key in self.sensor_list:
            for sensor in self.sensor_list[key]:
                sensor.destroy()

        cv2.destroyAllWindows()
