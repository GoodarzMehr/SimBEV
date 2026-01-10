# Academic Software License: Copyright © 2026 Goodarz Mehr.

'''
Module that spawns and manages the ego vehicle and its behavior.
'''

import carla
import random
import logging

import numpy as np

try:
    from .sensors import *
    
    from .utils import kill_all_servers

    from .sensor_manager import SensorManager
    from .ground_truth_manager import GTManager

except ImportError:
    from sensors import *
    
    from utils import kill_all_servers

    from sensor_manager import SensorManager
    from ground_truth_manager import GTManager


logger = logging.getLogger(__name__)


class VehicleManager:
    '''
    The Vehicle Manager spawns and manages the ego vehicle and its behavior.

    Args:
        config: dictionary of configuration parameters.
        world: CARLA world.
        traffic_manager: CARLA traffic manager.
        map_name: name of the CARLA map.
    '''
    def __init__(self, config: dict, world: carla.World, traffic_manager: carla.TrafficManager, map_name: str):
        self._config = config
        self._world = world
        self._traffic_manager = traffic_manager
        self._map_name = map_name
    
    def get_sensor_manager(self):
        '''Get the Sensor Manager.'''
        return self._sensor_manager
    
    def get_ground_truth_manager(self):
        '''Get the Ground Truth Manager.'''
        return self._ground_truth_manager
    
    def _spawn_sensors(self):
        '''Spawn the sensors attached to the ego vehicle.'''
        logger.debug('Creating the Sensor Manager...')

        self._sensor_manager = SensorManager(self._config, self.vehicle)

        logger.debug('Sensor Manager created.')
        logger.debug('Creating the sensors...')
        
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

        # Create the cameras.
        if self._config['use_rgb_camera']:
            for location in camera_locations:
                RGBCamera(
                    self._world,
                    self._sensor_manager,
                    location,
                    self.vehicle,
                    self._config['camera_width'],
                    self._config['camera_height'],
                    self._config['rgb_camera_properties']
                )
        
        if self._config['use_semantic_camera']:
            for location in camera_locations:
                SemanticCamera(
                    self._world,
                    self._sensor_manager,
                    location,
                    self.vehicle,
                    self._config['camera_width'],
                    self._config['camera_height'],
                    self._config['semantic_camera_properties']
                )
        
        if self._config['use_instance_camera']:
            for location in camera_locations:
                InstanceCamera(
                    self._world,
                    self._sensor_manager,
                    location,
                    self.vehicle,
                    self._config['camera_width'],
                    self._config['camera_height'],
                    self._config['instance_camera_properties']
                )
            
        
        if self._config['use_depth_camera']:
            for location in camera_locations:
                DepthCamera(
                    self._world,
                    self._sensor_manager,
                    location,
                    self.vehicle,
                    self._config['camera_width'],
                    self._config['camera_height'],
                    self._config['depth_camera_properties']
                )
        
        if self._config['use_flow_camera']:
            for location in camera_locations:
                FlowCamera(
                    self._world,
                    self._sensor_manager,
                    location,
                    self.vehicle,
                    self._config['camera_width'],
                    self._config['camera_height'],
                    self._config['flow_camera_properties']
                )
        
        # Create a lidar.
        if self._config['use_lidar']:
            Lidar(
                self._world,
                self._sensor_manager,
                carla.Transform(carla.Location(x=0.0, y=0.0, z=1.8)),
                self.vehicle,
                self._config['lidar_channels'],
                self._config['lidar_range'],
                self._config['lidar_properties']
            )
        
        # Create a semantic lidar.
        if self._config['use_semantic_lidar']:
            SemanticLidar(
                self._world,
                self._sensor_manager,
                carla.Transform(carla.Location(x=0.0, y=0.0, z=1.8), carla.Rotation(yaw=0.0)),
                self.vehicle,
                self._config['semantic_lidar_channels'],
                self._config['semantic_lidar_range'],
                self._config['semantic_lidar_properties']
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
        
        # Create the radars
        if self._config['use_radar']:
            for location in radar_locations:
                Radar(
                    self._world,
                    self._sensor_manager,
                    location,
                    self.vehicle,
                    self._config['radar_range'],
                    self._config['radar_horizontal_fov'],
                    self._config['radar_vertical_fov'],
                    self._config['radar_properties']
                )
        
        # Create a GNSS sensor.
        if self._config['use_gnss']:
            GNSS(
                self._world,
                self._sensor_manager,
                carla.Transform(),
                self.vehicle,
                self._config['gnss_properties']
            )
        
        # Create an IMU sensor.
        if self._config['use_imu']:
            IMU(
                self._world,
                self._sensor_manager,
                carla.Transform(),
                self.vehicle,
                self._config['imu_properties']
            )
        
        # Create BEV semantic cameras for obtaining the ground truth.
        bev_location_above = carla.Transform(
            carla.Location(x=0.0, y=0.0, z=self._config['bev_camera_height']),
            carla.Rotation(pitch=-90)
        )
        bev_location_below = carla.Transform(
            carla.Location(x=0.0, y=0.0, z=-self._config['bev_camera_height']),
            carla.Rotation(pitch=90)
        )

        bev_locations = [bev_location_above, bev_location_below]
        
        for location in bev_locations:
            SemanticBEVCamera(
                self._world,
                self._sensor_manager,
                location,
                self.vehicle,
                self._config['bev_dim'],
                self._config['bev_dim'],
                self._config['bev_properties']
            )
        
        # Create voxel detector for obtaining the 3D ground truth.
        if self._config['use_voxel_detector']:
            VoxelDetector(
                self._world,
                self._sensor_manager,
                carla.Transform(carla.Location(x=0.0, y=0.0, z=0.02)),
                self.vehicle,
                self._config['voxel_detector_range'],
                self._config['voxel_size'],
                self._config['voxel_detector_upper_limit'],
                self._config['voxel_detector_lower_limit'],
                self._config['voxel_detector_properties']
            )
        
        self._world.tick()

        logger.debug('Sensors created.')
    
    def spawn_vehicle(self, bp: carla.ActorBlueprint, spawn_points: list[carla.Waypoint], tm_port: int) -> dict:
        '''
        Spawn the ego vehicle and its sensors.
        
        Args:
            bp: ego vehicle blueprint.
            spawn_points: list of available spawn points.
            tm_port: Traffic Manager port.
        '''
        try:
            scene_info = {}
            
            bev_fov = 2 * np.rad2deg(
                np.arctan(self._config['bev_dim'] * self._config['bev_res'] / (2 * self._config['bev_camera_height'])))
            

            self._config['bev_properties'] = {'fov': str(bev_fov)}
            
            # Instantiate the vehicle.
            logger.debug('Spawning the ego vehicle...')
            self.vehicle = None

            spawn_point = random.choice(spawn_points)

            self._world.get_spectator().set_transform(
                carla.Transform(
                    spawn_point.location + carla.Location(z=40.0),
                    carla.Rotation(pitch=-90.0, yaw=spawn_point.rotation.yaw)
                )
            )

            for _ in range(100):
                self._world.tick()
            
            while self.vehicle is None:
                self.vehicle = self._world.try_spawn_actor(bp, spawn_point)

            self.vehicle.set_autopilot(True, tm_port)
            self.vehicle.set_simulate_physics(self._config['simulate_physics'])
            self.vehicle.show_debug_telemetry(self._config['show_debug_telemetry'])

            self._traffic_manager.update_vehicle_lights(self.vehicle, True)

            logger.debug('Ego vehicle spawned.')

            # Set the percentage of time the ego vehicle ignores traffic
            # lights, traffic signs, other vehicles, and walkers.
            logger.debug('Configuring ego vehicle behavior...')

            self._traffic_manager.ignore_lights_percentage(self.vehicle, self._config['ignore_lights_percentage'])
            self._traffic_manager.ignore_signs_percentage(self.vehicle, self._config['ignore_signs_percentage'])
            self._traffic_manager.ignore_vehicles_percentage(self.vehicle, self._config['ignore_vehicles_percentage'])
            self._traffic_manager.ignore_walkers_percentage(self.vehicle, self._config['ignore_walkers_percentage'])

            # Determine whether the ego vehicle is reckless (ignores all
            # traffic rules).
            scene_info['reckless_ego'] = False
            scene_info['distracted_ego'] = False

            p = self._config['reckless_ego_percentage'] / 100.0
            
            if np.random.choice(2, p=[1 - p, p]):
                logger.warning('The ego vehicle is reckless!')

                self._traffic_manager.ignore_lights_percentage(self.vehicle, 100.0)
                self._traffic_manager.ignore_signs_percentage(self.vehicle, 100.0)
                self._traffic_manager.ignore_vehicles_percentage(self.vehicle, 100.0)
                self._traffic_manager.ignore_walkers_percentage(self.vehicle, 100.0)

                scene_info['reckless_ego'] = True
            else:
                p = self._config['distracted_ego_percentage'] / 100.0
                
                if np.random.choice(2, p=[1 - p, p]):
                    logger.warning('The ego vehicle is distracted!')

                    self._traffic_manager.ignore_lights_percentage(self.vehicle, 100.0)
                    self._traffic_manager.ignore_signs_percentage(self.vehicle, 100.0)

                    scene_info['distracted_ego'] = True

            if 'speed_difference' not in self._config:
                self._traffic_manager.vehicle_percentage_speed_difference(self.vehicle, random.uniform(-40.0, 20.0))
            
            if 'distance_to_leading' not in self._config:
                self._traffic_manager.distance_to_leading_vehicle(self.vehicle, random.gauss(4.2, 1.0))
            
            logger.debug('Ego vehicle behavior configured.')

            self._spawn_sensors()

            # Instantiate the Ground Truth Manager.
            logger.debug('Creating the Ground Truth Manager...')

            self._ground_truth_manager = GTManager(
                self._config,
                self._world,
                self.vehicle,
                self._sensor_manager,
                self._map_name
            )

            logger.debug('Ground Truth Manager created.')

            self._world.tick()

            return scene_info

        except Exception as e:
            logger.error(f'Error while spawning the vehicle: {e}')

            kill_all_servers()

            time.sleep(3.0)

            raise Exception('Cannot spawn the vehicle. Good bye!')
    
    def move_vehicle(self, spawn_points: list[carla.Waypoint], tm_port: int) -> dict:
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

            self._world.tick()

            time.sleep(1.0)

            spawn_point = random.choice(spawn_points)

            self._world.get_spectator().set_transform(
                carla.Transform(
                    spawn_point.location + carla.Location(z=40.0),
                    carla.Rotation(pitch=-90.0, yaw=spawn_point.rotation.yaw)
                )
            )

            for _ in range(99):
                self._world.tick()

            self.vehicle.set_transform(spawn_point)
            self.vehicle.disable_constant_velocity()
            self.vehicle.set_autopilot(True, tm_port)

            # Reset the sensors.
            self._sensor_manager.reset()

            self._world.tick()

            time.sleep(1.0)

            logger.debug('Ego vehicle moved.')

            # Determine whether the ego vehicle is reckless (ignores all
            # traffic rules).
            logger.debug('Configuring ego vehicle behavior...')

            p = self._config['reckless_ego_percentage'] / 100.0
            
            if np.random.choice(2, p=[1 - p, p]):
                logger.warning('The ego vehicle is reckless!')

                self._traffic_manager.ignore_lights_percentage(self.vehicle, 100.0)
                self._traffic_manager.ignore_signs_percentage(self.vehicle, 100.0)
                self._traffic_manager.ignore_vehicles_percentage(self.vehicle, 100.0)
                self._traffic_manager.ignore_walkers_percentage(self.vehicle, 100.0)

                scene_info['reckless_ego'] = True    
            else:
                p = self._config['distracted_ego_percentage'] / 100.0

                if np.random.choice(2, p=[1 - p, p]):
                    logger.warning('The ego vehicle is distracted!')

                    self._traffic_manager.ignore_lights_percentage(self.vehicle, 100.0)
                    self._traffic_manager.ignore_signs_percentage(self.vehicle, 100.0)

                    scene_info['distracted_ego'] = True
                else:
                    self._traffic_manager.ignore_lights_percentage(
                        self.vehicle,
                        self._config['ignore_lights_percentage']
                    )
                    self._traffic_manager.ignore_signs_percentage(
                        self.vehicle,
                        self._config['ignore_signs_percentage']
                    )
                    self._traffic_manager.ignore_vehicles_percentage(
                        self.vehicle,
                        self._config['ignore_vehicles_percentage']
                    )
                    self._traffic_manager.ignore_walkers_percentage(
                        self.vehicle,
                        self._config['ignore_walkers_percentage']
                    )

                    scene_info['distracted_ego'] = False
                
                scene_info['reckless_ego'] = False
            
            if 'speed_difference' not in self._config:
                self._traffic_manager.vehicle_percentage_speed_difference(self.vehicle, random.uniform(-40.0, 20.0))
            
            if 'distance_to_leading' not in self._config:
                self._traffic_manager.distance_to_leading_vehicle(self.vehicle, random.gauss(4.2, 1.0))

            self._world.tick()

            logger.debug('Ego vehicle behavior configured.')

            return scene_info

        except Exception as e:
            logger.error(f'Error while moving the vehicle: {e}')

            kill_all_servers()

            time.sleep(3.0)

            raise Exception('Cannot move the vehicle. Good bye!')
    
    def find_vehicle(self):
        '''Find the vehicle in the world.'''
        try:
            logger.debug('Finding the ego vehicle in the world...')

            self._world.tick()

            actors = self._world.get_actors()

            for actor in actors:
                if 'role_name' in actor.attributes and actor.attributes['role_name'] == 'hero':
                    self.vehicle = actor

                    logger.debug('Ego vehicle found.')

                    self._spawn_sensors()

                    self._world.tick()

                    return
            
            raise Exception('Ego vehicle not found in the world.')

        except Exception as e:
            logger.error(f'Error while finding the vehicle: {e}')

            kill_all_servers()

            time.sleep(3.0)

            raise Exception('Cannot find the vehicle. Good bye!')
    
    def destroy_vehicle(self):
        '''Destroy the Sensor Manager and the vehicle.'''
        logger.debug('Destroying the Sensor Manager...')

        self._sensor_manager.destroy()

        logger.debug('Sensor Manager destroyed.')
        logger.debug('Destroying the ego vehicle...')

        self.vehicle.destroy()

        logger.debug('Ego vehicle destroyed.')
