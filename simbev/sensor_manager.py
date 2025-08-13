# Academic Software License: Copyright © 2025 Goodarz Mehr.

'''
Sensor manager module that collects data from all sensors on a vehicle and
renders or saves them.
'''

import cv2

import numpy as np

from utils import CustomTimer

from scipy.spatial.transform import Rotation as R

class SensorManager:
    '''
    Sensor Manager class that manages data collection.

    Args:
        core: CarlaCore instance that manages the simulation.
        vehicle: vehicle the sensor manager belongs to.
    '''

    def __init__(self, core, vehicle):
        self.core = core
        self.vehicle = vehicle
        
        self.camera_list = []
        self.semantic_camera_list = []
        self.instance_camera_list = []
        self.depth_camera_list = []
        self.flow_camera_list = []
        
        self.lidar_list = []
        self.semantic_lidar_list = []

        self.radar_list = []

        self.gnss_list = []

        self.imu_list = []
        
        self.semantic_bev_camera_list = []

        self.timer = CustomTimer()

        self.camera_name_list = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                                 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
        
        self.radar_name_list = ['RAD_LEFT', 'RAD_FRONT', 'RAD_RIGHT', 'RAD_BACK']
        
        self.bev_camera_name_list = ['TOP_VIEW', 'BOTTOM_VIEW']
        
        self.data = []

    def add_camera(self, camera):
        '''
        Add camera to the list of cameras.

        Args:
            camera: camera to add to the list of cameras.
        '''
        self.camera_list.append(camera)

    def add_semantic_camera(self, semantic_camera):
        '''
        Add semantic segmentation camera to the list of cameras.

        Args:
            semantic_camera: semantic segmentation camera to add to the list
                of cameras.
        '''
        self.semantic_camera_list.append(semantic_camera)

    def add_instance_camera(self, instance_camera):
        '''
        Add instance segmentation camera to the list of cameras.

        Args:
            instance_camera: instance segmentation camera to add to the list
                of cameras.
        '''
        self.instance_camera_list.append(instance_camera)
    
    def add_depth_camera(self, depth_camera):
        '''
        Add depth camera to the list of cameras.

        Args:
            depth_camera: depth camera to add to the list of cameras.
        '''
        self.depth_camera_list.append(depth_camera)
    
    def add_flow_camera(self, flow_camera):
        '''
        Add optical flow camera to the list of cameras.

        Args:
            flow_camera: optical flow camera to add to the list of cameras.
        '''
        self.flow_camera_list.append(flow_camera)
    
    def add_lidar(self, lidar):
        '''
        Add lidar to the list of lidars.

        Args:
            lidar: lidar to add to the list of lidars.
        '''
        self.lidar_list.append(lidar)
    
    def add_semantic_lidar(self, semantic_lidar):
        '''
        Add semantic lidar to the list of semantic lidars.

        Args:
            semantic_lidar: semantic lidar to add to the list of semantic
                lidars.
        '''
        self.semantic_lidar_list.append(semantic_lidar)
    
    def add_radar(self, radar):
        '''
        Add radar to the list of radars.

        Args:
            radar: radar to add to the list of radars.
        '''
        self.radar_list.append(radar)
    
    def add_gnss(self, gnss):
        '''
        Add GNSS sensor to the list of GNSS sensors.

        Args:
            gnss: GNSS sensor to add to the list of GNSS sensors.
        '''
        self.gnss_list.append(gnss)
    
    def add_imu(self, imu):
        '''
        Add IMU sensor to the list of IMU sensors.

        Args:
            imu: IMU sensor to add to the list of IMU sensors.
        '''
        self.imu_list.append(imu)
    
    def add_semantic_bev_camera(self, semantic_bev_camera):
        '''
        Add semantic segmentation camera to the list of BEV cameras.

        Args:
            semantic_bev_camera: semantic segmentation camera to add to the
                list of BEV cameras.
        '''
        self.semantic_bev_camera_list.append(semantic_bev_camera)
    
    def clear_queues(self):
        '''
        Clear sensor queues.
        '''
        for camera in self.camera_list:
            camera.clear_queues()

        for semantic_camera in self.semantic_camera_list:
            semantic_camera.clear_queues()
        
        for instance_camera in self.instance_camera_list:
            instance_camera.clear_queues()
        
        for depth_camera in self.depth_camera_list:
            depth_camera.clear_queues()
        
        for flow_camera in self.flow_camera_list:
            flow_camera.clear_queues()
        
        for lidar in self.lidar_list:
            lidar.clear_queues()

        for semantic_lidar in self.semantic_lidar_list:
            semantic_lidar.clear_queues()
        
        for radar in self.radar_list:
            radar.clear_queues()
        
        for gnss in self.gnss_list:
            gnss.clear_queue()
        
        for imu in self.imu_list:
            imu.clear_queue()
        
        for semantic_bev_camera in self.semantic_bev_camera_list:
            semantic_bev_camera.clear_queues()
    
    def render(self):
        '''
        Render sensor data.
        '''
        for camera, window_name in zip(self.camera_list, self.camera_name_list):
            camera.render(window_name)

        for semantic_camera, window_name in zip(self.semantic_camera_list, self.camera_name_list):
            semantic_camera.render('SEG ' + window_name)
        
        for instance_camera, window_name in zip(self.instance_camera_list, self.camera_name_list):
            instance_camera.render('IST ' + window_name)
        
        for depth_camera, window_name in zip(self.depth_camera_list, self.camera_name_list):
            depth_camera.render('DPT ' + window_name)
        
        for flow_camera, window_name in zip(self.flow_camera_list, self.camera_name_list):
            flow_camera.render('FLW ' + window_name)
        
        for lidar in self.lidar_list:
            lidar.render()

        for semantic_lidar in self.semantic_lidar_list:
            semantic_lidar.render()
        
        for radar, window_name in zip(self.radar_list, self.radar_name_list):
            radar.render(window_name)
        
        for semantic_bev_camera, window_name in zip(self.semantic_bev_camera_list, self.bev_camera_name_list):
            semantic_bev_camera.render(window_name)
    
    def save(self, path, scene, frame):
        '''
        Save sensor data.

        Args:
            path: root directory of the dataset.
            scene: scene number.
            frame: frame number.
        '''
        for camera, camera_name in zip(self.camera_list, self.camera_name_list):
            camera.save(camera_name, path, scene, frame)

        for semantic_camera, camera_name in zip(self.semantic_camera_list, self.camera_name_list):
            semantic_camera.save(camera_name, path, scene, frame)
        
        for instance_camera, camera_name in zip(self.instance_camera_list, self.camera_name_list):
            instance_camera.save(camera_name, path, scene, frame)
        
        for depth_camera, camera_name in zip(self.depth_camera_list, self.camera_name_list):
            depth_camera.save(camera_name, path, scene, frame)
        
        for flow_camera, camera_name in zip(self.flow_camera_list, self.camera_name_list):
            flow_camera.save(camera_name, path, scene, frame)
        
        for lidar in self.lidar_list:
            lidar.save(path, scene, frame)

        for semantic_lidar in self.semantic_lidar_list:
            semantic_lidar.save(path, scene, frame)
        
        for radar, radar_name in zip(self.radar_list, self.radar_name_list):
            radar.save(radar_name, path, scene, frame)
        
        for gnss in self.gnss_list:
            gnss.save(path, scene, frame)
        
        for imu in self.imu_list:
            imu.save(path, scene, frame)
        
        scene_data = {}

        ego_transform = self.vehicle.get_transform()

        scene_data['ego2global_translation'] = [ego_transform.location.x,
                                                -ego_transform.location.y,
                                                ego_transform.location.z]
        scene_data['ego2global_rotation'] = np.roll(
            R.from_euler(
                'xyz',
                [ego_transform.rotation.roll, -ego_transform.rotation.pitch, -ego_transform.rotation.yaw],
                degrees=True).as_quat(), 1
        ).tolist()
        
        scene_data['timestamp'] = round(self.timer.time() * 10e6)

        for camera_name in self.camera_name_list:
            if self.core.config['use_rgb_camera']:
                scene_data[f'RGB-{camera_name}'] = f'{path}/simbev/sweeps/RGB-{camera_name}' \
                f'/SimBEV-scene-{scene:04d}-frame-{frame:04d}-RGB-{camera_name}.jpg'
            if self.core.config['use_semantic_camera']:
                scene_data[f'SEG-{camera_name}'] = f'{path}/simbev/sweeps/SEG-{camera_name}' \
                f'/SimBEV-scene-{scene:04d}-frame-{frame:04d}-SEG-{camera_name}.png'
            if self.core.config['use_instance_camera']:
                scene_data[f'IST-{camera_name}'] = f'{path}/simbev/sweeps/IST-{camera_name}' \
                f'/SimBEV-scene-{scene:04d}-frame-{frame:04d}-IST-{camera_name}.png'
            if self.core.config['use_depth_camera']:
                scene_data[f'DPT-{camera_name}'] = f'{path}/simbev/sweeps/DPT-{camera_name}' \
                f'/SimBEV-scene-{scene:04d}-frame-{frame:04d}-DPT-{camera_name}.png'
            if self.core.config['use_flow_camera']:
                scene_data[f'FLW-{camera_name}'] = f'{path}/simbev/sweeps/FLW-{camera_name}' \
                f'/SimBEV-scene-{scene:04d}-frame-{frame:04d}-FLW-{camera_name}.npz'

        if self.core.config['use_lidar']:
            scene_data['LIDAR'] = f'{path}/simbev/sweeps/LIDAR/SimBEV-scene-{scene:04d}-frame-{frame:04d}-LIDAR.npz'
        if self.core.config['use_semantic_lidar']:
            scene_data['SEG-LIDAR'] = f'{path}/simbev/sweeps/SEG-LIDAR' \
            f'/SimBEV-scene-{scene:04d}-frame-{frame:04d}-SEG-LIDAR.npz'
        
        if self.core.config['use_radar']:
            for radar_name in self.radar_name_list:
                scene_data[f'{radar_name}'] = f'{path}/simbev/sweeps/{radar_name}' \
                f'/SimBEV-scene-{scene:04d}-frame-{frame:04d}-{radar_name}.npz'
        
        if self.core.config['use_gnss']:
            scene_data['GNSS'] = f'{path}/simbev/sweeps/GNSS/SimBEV-scene-{scene:04d}-frame-{frame:04d}-GNSS.bin'
        
        if self.core.config['use_imu']:
            scene_data['IMU'] = f'{path}/simbev/sweeps/IMU/SimBEV-scene-{scene:04d}-frame-{frame:04d}-IMU.bin'
        
        scene_data['GT_SEG'] = f'{path}/simbev/ground-truth/seg/SimBEV-scene-{scene:04d}-frame-{frame:04d}-GT_SEG.npz'
        scene_data['GT_SEG_VIZ'] = f'{path}/simbev/ground-truth/seg_viz' \
        f'/SimBEV-scene-{scene:04d}-frame-{frame:04d}-GT_SEG_VIZ.jpg'
        scene_data['GT_DET'] = f'{path}/simbev/ground-truth/det/SimBEV-scene-{scene:04d}-frame-{frame:04d}-GT_DET.bin'
        scene_data['HD_MAP'] = f'{path}/simbev/ground-truth/hd_map/SimBEV-scene-{scene:04d}-frame-{frame:04d}-HD_MAP.json'

        scene_data['scene'] = scene
        scene_data['frame'] = frame

        self.data.append(scene_data)
    
    def reset(self):
        '''
        Reset scenario data.
        '''
        self.data = []
    
    def destroy(self):
        '''
        Destroy sensors.
        '''
        for camera in self.camera_list:
            camera.destroy()

        for semantic_camera in self.semantic_camera_list:
            semantic_camera.destroy()
        
        for instance_camera in self.instance_camera_list:
            instance_camera.destroy()
        
        for depth_camera in self.depth_camera_list:
            depth_camera.destroy()
        
        for flow_camera in self.flow_camera_list:
            flow_camera.destroy()
        
        for lidar in self.lidar_list:
            lidar.destroy()

        for semantic_lidar in self.semantic_lidar_list:
            semantic_lidar.destroy()
        
        for radar in self.radar_list:
            radar.destroy()
        
        for gnss in self.gnss_list:
            gnss.destroy()
        
        for imu in self.imu_list:
            imu.destroy()
        
        for semantic_bev_camera in self.semantic_bev_camera_list:
            semantic_bev_camera.destroy()

        cv2.destroyAllWindows()
