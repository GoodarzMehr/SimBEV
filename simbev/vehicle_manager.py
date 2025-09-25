# Academic Software License: Copyright © 2025 Goodarz Mehr.

import carla
import random
import logging

import numpy as np

from sensors import *

from utils import kill_all_servers

from sensor_manager import SensorManager
from ground_truth_manager import GTManager


logger = logging.getLogger(__name__)


class VehicleManager:
    def __init__(self, config, world, traffic_manager, map_name):
        self.config = config
        self.world = world
        self.traffic_manager = traffic_manager
        self.map_name = map_name
    
    def get_ground_truth_manager(self):
        return self.ground_truth_manager
    
    def get_sensor_manager(self):
        return self.sensor_manager
    
    def spawn_vehicle(self, bp, spawn_points, tm_port):
        '''
        Spawn the ego vehicle and its sensors.
        '''
        try:
            scene_info = {}
            
            bev_fov = 2 * np.rad2deg(np.arctan(self.config['bev_dim'] * self.config['bev_res'] / 2000))

            self.config['bev_properties'] = {'fov': str(bev_fov)}
            
            # Instantiate the vehicle.
            logger.debug('Spawning the ego vehicle...')
            self.vehicle = None

            while self.vehicle is None:
                self.vehicle = self.world.try_spawn_actor(bp, random.choice(spawn_points))
                # self.vehicle = self.world.try_spawn_actor(bp, spawn_points[0])

            self.vehicle.set_autopilot(True, tm_port)
            self.vehicle.set_simulate_physics(self.config['simulate_physics'])
            self.vehicle.show_debug_telemetry(self.config['show_debug_telemetry'])

            self.traffic_manager.update_vehicle_lights(self.vehicle, True)

            logger.debug('Ego vehicle spawned.')

            # Set the percentage of time the ego vehicle ignores traffic
            # lights, traffic signs, other vehicles, and walkers.
            logger.debug('Configuring ego vehicle behavior...')

            self.traffic_manager.ignore_lights_percentage(self.vehicle, self.config['ignore_lights_percentage'])
            self.traffic_manager.ignore_signs_percentage(self.vehicle, self.config['ignore_signs_percentage'])
            self.traffic_manager.ignore_vehicles_percentage(self.vehicle, self.config['ignore_vehicles_percentage'])
            self.traffic_manager.ignore_walkers_percentage(self.vehicle, self.config['ignore_walkers_percentage'])

            # Determine whether the ego vehicle is reckless (ignores all
            # traffic rules).
            scene_info['reckless_ego'] = False

            if self.config['reckless_ego']:
                p = self.config['reckless_ego_percentage'] / 100.0
                
                if np.random.choice(2, p=[1 - p, p]):
                    logger.warning('Ego vehicle is reckless!')

                    self.traffic_manager.ignore_lights_percentage(self.vehicle, 100.0)
                    self.traffic_manager.ignore_signs_percentage(self.vehicle, 100.0)
                    self.traffic_manager.ignore_vehicles_percentage(self.vehicle, 100.0)
                    self.traffic_manager.ignore_walkers_percentage(self.vehicle, 100.0)

                    scene_info['reckless_ego'] = True

            if 'speed_difference' not in self.config:
                self.traffic_manager.vehicle_percentage_speed_difference(self.vehicle, random.uniform(-40.0, 20.0))
            
            if 'distance_to_leading' not in self.config:
                self.traffic_manager.distance_to_leading_vehicle(self.vehicle, random.gauss(4.2, 1.0))
            
            logger.debug('Ego vehicle behavior configured.')

            # Instantiate the Sensor Manager.
            logger.debug('Creating sensors...')

            self.sensor_manager = SensorManager(self.config, self.vehicle)

            # Set up camera locations.
            camera_location_front_left = carla.Transform(
                carla.Location(x=0.4, y=-0.4, z=1.6),
                carla.Rotation(yaw=-55.0)
            )
            camera_location_front = carla.Transform(
                carla.Location(x=0.6, y=0.0, z=1.6),
                carla.Rotation(yaw=0.0)
            )
            camera_location_front_right = carla.Transform(
                carla.Location(x=0.4, y=0.4, z=1.6),
                carla.Rotation(yaw=55.0)
            )
            camera_location_back_left = carla.Transform(
                carla.Location(x=0.0, y=-0.4, z=1.6),
                carla.Rotation(yaw=-110)
            )
            camera_location_back = carla.Transform(
                carla.Location(x=-1.0, y=0.0, z=1.6),
                carla.Rotation(yaw=180.0)
            )
            camera_location_back_right = carla.Transform(
                carla.Location(x=0.0, y=0.4, z=1.6),
                carla.Rotation(yaw=110.0)
            )

            camera_locations = [
                camera_location_front_left,
                camera_location_front,
                camera_location_front_right,
                camera_location_back_left,
                camera_location_back,
                camera_location_back_right
            ]

            # Create cameras.
            if self.config['use_rgb_camera']:
                for location in camera_locations:
                    RGBCamera(
                        self.world,
                        self.sensor_manager,
                        location,
                        self.vehicle,
                        self.config['camera_width'],
                        self.config['camera_height'],
                        self.config['rgb_camera_properties']
                    )
            
            if self.config['use_semantic_camera']:
                for location in camera_locations:
                    SemanticCamera(
                        self.world,
                        self.sensor_manager,
                        location,
                        self.vehicle,
                        self.config['camera_width'],
                        self.config['camera_height'],
                        self.config['semantic_camera_properties']
                    )
            
            if self.config['use_instance_camera']:
                for location in camera_locations:
                    InstanceCamera(
                        self.world,
                        self.sensor_manager,
                        location,
                        self.vehicle,
                        self.config['camera_width'],
                        self.config['camera_height'],
                        self.config['instance_camera_properties']
                    )
                
            
            if self.config['use_depth_camera']:
                for location in camera_locations:
                    DepthCamera(
                        self.world,
                        self.sensor_manager,
                        location,
                        self.vehicle,
                        self.config['camera_width'],
                        self.config['camera_height'],
                        self.config['depth_camera_properties']
                    )
            
            if self.config['use_flow_camera']:
                for location in camera_locations:
                    FlowCamera(
                        self.world,
                        self.sensor_manager,
                        location,
                        self.vehicle,
                        self.config['camera_width'],
                        self.config['camera_height'],
                        self.config['flow_camera_properties']
                    )
            
            # Create a lidar.
            if self.config['use_lidar']:
                Lidar(
                    self.world,
                    self.sensor_manager,
                    carla.Transform(carla.Location(x=0.0, y=0.0, z=1.8)),
                    self.vehicle,
                    self.config['lidar_channels'],
                    self.config['lidar_range'],
                    self.config['lidar_properties']
                )
            
            # Create a semantic lidar.
            if self.config['use_semantic_lidar']:
                SemanticLidar(
                    self.world,
                    self.sensor_manager,
                    carla.Transform(carla.Location(x=0.0, y=0.0, z=1.8), carla.Rotation(yaw=0.0)),
                    self.vehicle,
                    self.config['semantic_lidar_channels'],
                    self.config['semantic_lidar_range'],
                    self.config['semantic_lidar_properties']
                )
            
            radar_location_left = carla.Transform(carla.Location(x=0.0, y=-1.0, z=0.6), carla.Rotation(yaw=-90.0))
            radar_location_front = carla.Transform(carla.Location(x=2.4, y=0.0, z=0.6), carla.Rotation(yaw=0.0))
            radar_location_right = carla.Transform(carla.Location(x=0.0, y=1.0, z=0.6), carla.Rotation(yaw=90.0))
            radar_location_back = carla.Transform(carla.Location(x=-2.4, y=0.0, z=0.6), carla.Rotation(yaw=180.0))

            radar_locations = [
                radar_location_left,
                radar_location_front,
                radar_location_right,
                radar_location_back,
            ]
            
            # Create radars
            if self.config['use_radar']:
                for location in radar_locations:
                    Radar(
                        self.world,
                        self.sensor_manager,
                        location,
                        self.vehicle,
                        self.config['radar_range'],
                        self.config['radar_horizontal_fov'],
                        self.config['radar_vertical_fov'],
                        self.config['radar_properties']
                    )
            
            # Create a GNSS sensor.
            if self.config['use_gnss']:
                GNSS(
                    self.world,
                    self.sensor_manager,
                    carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0)),
                    self.vehicle,
                    self.config['gnss_properties']
                )
            
            # Create an IMU sensor.
            if self.config['use_imu']:
                IMU(
                    self.world,
                    self.sensor_manager,
                    carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0)),
                    self.vehicle,
                    self.config['imu_properties']
                )
            
            # Create BEV semantic cameras for obtaining the ground truth.
            SemanticBEVCamera(
                self.world,
                self.sensor_manager,
                carla.Transform(carla.Location(x=0.0, y=0.0, z=1000.0), carla.Rotation(pitch=-90)),
                self.vehicle,
                self.config['bev_dim'],
                self.config['bev_dim'],
                self.config['bev_properties']
            )
            SemanticBEVCamera(
                self.world,
                self.sensor_manager,
                carla.Transform(carla.Location(x=0.0, y=0.0, z=-1000.0), carla.Rotation(pitch=90)),
                self.vehicle,
                self.config['bev_dim'],
                self.config['bev_dim'],
                self.config['bev_properties']
            )
            
            self.world.tick()

            logger.debug('Sensors created.')
            logger.debug('Creating the Ground Truth Manager...')

            self.ground_truth_manager = GTManager(
                self.config,
                self.world,
                self.vehicle,
                self.sensor_manager,
                self.map_name
            )

            logger.debug('Ground Truth Manager created.')

            return scene_info

        except Exception as e:
            logger.error(f'Error while spawning the vehicle: {e}')

            kill_all_servers()

            time.sleep(3.0)

            raise Exception('Cannot spawn the vehicle. Good bye!')
    
    def move_vehicle(self, spawn_points, tm_port):
        '''
        Move the ego vehicle to a random spawn point and configure its
        behavior at the start of a new scene.
        '''
        try:
            scene_info = {}
            
            # Move the vehicle.
            logger.debug('Moving the ego vehicle...')

            self.vehicle.set_autopilot(False, tm_port)
            self.vehicle.enable_constant_velocity(carla.Vector3D(0.0, 0.0, 0.0))

            self.world.tick()

            time.sleep(1.0)

            self.vehicle.set_transform(random.choice(spawn_points))
            # self.vehicle.set_transform(spawn_points[0])
            self.vehicle.disable_constant_velocity()
            self.vehicle.set_autopilot(True, tm_port)

            self.sensor_manager.reset()

            self.world.tick()

            time.sleep(1.0)

            logger.debug('Ego vehicle moved.')

            # Determine whether the ego vehicle is reckless (ignores all
            # traffic rules).
            logger.debug('Configuring ego vehicle behavior...')

            if self.config['reckless_ego']:
                p = self.config['reckless_ego_percentage'] / 100.0
                
                if np.random.choice(2, p=[1 - p, p]):
                    logger.warning('The ego vehicle is reckless!')

                    self.traffic_manager.ignore_lights_percentage(self.vehicle, 100.0)
                    self.traffic_manager.ignore_signs_percentage(self.vehicle, 100.0)
                    self.traffic_manager.ignore_vehicles_percentage(self.vehicle, 100.0)
                    self.traffic_manager.ignore_walkers_percentage(self.vehicle, 100.0)

                    scene_info['reckless_ego'] = True    
                else:
                    self.traffic_manager.ignore_lights_percentage(
                        self.vehicle,
                        self.config['ignore_lights_percentage']
                    )
                    self.traffic_manager.ignore_signs_percentage(
                        self.vehicle,
                        self.config['ignore_signs_percentage']
                    )
                    self.traffic_manager.ignore_vehicles_percentage(
                        self.vehicle,
                        self.config['ignore_vehicles_percentage']
                    )
                    self.traffic_manager.ignore_walkers_percentage(
                        self.vehicle,
                        self.config['ignore_walkers_percentage']
                    )

                    scene_info['reckless_ego'] = False
            
            if 'speed_difference' not in self.config:
                self.traffic_manager.vehicle_percentage_speed_difference(self.vehicle, random.uniform(-40.0, 20.0))
            
            if 'distance_to_leading' not in self.config:
                self.traffic_manager.distance_to_leading_vehicle(self.vehicle, random.gauss(4.2, 1.0))

            self.world.tick()

            logger.debug('Ego vehicle behavior configured.')

            return scene_info

        except Exception as e:
            logger.error(f'Error while moving the vehicle: {e}')

            kill_all_servers()

            time.sleep(3.0)

            raise Exception('Cannot move the vehicle. Good bye!')
    
    def destroy_vehicle(self):
        '''
        Destroy the Sensor Manager and the vehicle.
        '''
        logger.debug('Destroying the Sensor Manager...')

        self.sensor_manager.destroy()

        logger.debug('Sensor Manager destroyed.')
        logger.debug('Destroying the vehicle...')

        self.vehicle.destroy()

        logger.debug('Vehicle destroyed.')
        logger.debug('Shutting down the Traffic Manager...')

        self.traffic_manager.shut_down()

        logger.debug('Traffic Manager shut down.')