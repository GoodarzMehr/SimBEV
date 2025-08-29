# Academic Software License: Copyright © 2025 Goodarz Mehr.

'''
Module that performs the core functions of CARLA, such as initializing the
server, connecting the client, setting up scenarios, spawning and destroying
actors, applying actions, and setting the spectator view.
'''

import os
import cv2
import time
import json
import torch
import carla
import random
import signal
import psutil
import logging
import subprocess

import numpy as np

from utils import *
from sensors import *

from sensor_manager import SensorManager

from skimage.morphology import binary_closing, binary_opening, binary_dilation

logger = logging.getLogger(__name__)

MAP_PALETTE = {
    'road': (196, 80, 196),
    'road_line': (160, 240, 40),
    'sidewalk': (240, 196, 240),
    'crosswalk': (240, 196, 196),
    'car': (0, 128, 240),
    'truck': (80, 240, 80),
    'bus': (0, 144, 0),
    'motorcycle': (240, 240, 0),
    'bicycle': (0, 240, 240),
    'rider': (240, 144, 0),
    'pedestrian': (240, 0, 0)
}

CITYSCAPE_PALETTE = {
    'road': (128, 64, 128),
    'road_line': (227, 227, 227),
    'sidewalk': (244, 35, 232),
    'crosswalk': (157, 234, 50),
    'car': (0, 0, 142),
    'truck': (0, 0, 70),
    'bus': (0, 60, 100),
    'motorcycle': (0, 0, 230),
    'bicycle': (119, 11, 32),
    'rider': (255, 0, 0),
    'pedestrian': (220, 20, 60)
}

TRAFFIC_SIGN = {
    'highwaySign': 'highway',
    'pole_': 'billboard',
    'DoNoEnter': 'do_not_enter',
    'Interchange': 'interchange',
    'NoTrunLeft': 'no_left_turn',
    'busNoPark': 'bus_stop_no_parking',
    'cleanPet': 'clean_up_after_pet',
    'minPark': 'parking_time_limit',
    'noBicyc': 'no_bicycles',
    'noPark': 'no_parking',
    'noPed': 'no_pedestrians',
    'noStand': 'no_standing',
    'onlyCrossw': 'cross_only_at_crosswalk',
    'photo': 'photo_enforced',
    'reserved': 'reserved_parking',
    'school': 'school_zone',
    'stopPed': 'stop_for_pedestrians',
    'tow': 'tow_away_zone',
    'yieldPed': 'yield_to_pedestrians',
    'passenger': 'passenger_cars_only',
    'DiamondSignal': 'accident_ahead',
    'wildCrossing': 'animal_crossing',
    'AnimalCrossing': 'animal_crossing',
    'LaneReduceL': 'lane_reduction',
    'LaneReduct': 'lane_reduction',
    'MichiganLeft': 'michigan_left',
    'NoTurns': 'no_turns',
    'noTurn': 'no_turns',
    'OneWay': 'one_way',
    'oneWay': 'one_way',
    'stopSign': 'stop',
    'STOP_': 'stop',
    '_Stop': 'stop',
    '_Yield': 'yield',
    '_yield_': 'yield',
    '_Flag': 'street_name',
    '_letter': 'street_name',
    'SpeedLimiter': 'speed_limit',
    'SpeedSign': 'speed_limit',
    '_SpeedLimit': 'speed_limit',
    'SpeedLimit20': 'speed_limit_20',
    'SpeedLimit25': 'speed_limit_25',
    'SpeedLimit30': 'speed_limit_30',
    'SpeedLimit40': 'speed_limit_40',
    'SpeedLimit50': 'speed_limit_50',
    'SpeedLimit55': 'speed_limit_55',
    'SpeedLimit60': 'speed_limit_60',
    'SpeedLimit70': 'speed_limit_70',
    'SpeedLimit75': 'speed_limit_75',
    'SpeedLimit80': 'speed_limit_80',
    'SpeedLimit90': 'speed_limit_90',
    'SpeedLimit100': 'speed_limit_100',
    'SpeedLimit110': 'speed_limit_110',
    'SpeedLimit120': 'speed_limit_120',
    'SpeedLimit20_15': 'speed_limit_20_min_15',
    'SpeedLimit25_15': 'speed_limit_25_min_15',
    'SpeedLimit30_15': 'speed_limit_30_min_15',
    'SpeedLimit50_40': 'speed_limit_50_min_40',
    'SpeedLimit55_40': 'speed_limit_55_min_40',
    'SpeedLimit75_45': 'speed_limit_75_min_45',
}

WEATHER_ATTRIBUTES = [
    'cloudiness',
    'precipitation',
    'precipitation_deposits',
    'wind_intensity',
    'sun_azimuth_angle',
    'sun_altitude_angle',
    'wetness',
    'fog_density',
    'fog_distance',
    'fog_falloff',
    'scattering_intensity',
    'mie_scattering_scale',
    'rayleigh_scattering_scale',
    'dust_storm'
]


def is_used(port):
    '''
    Check whether or not a port is used.

    Args:
        port: port number.
    
    Return:
        True if the port is being used, False otherwise.
    '''
    return port in [conn.laddr.port for conn in psutil.net_connections()]

def kill_all_servers():
    '''
    Kill all PIDs that start with CARLA.
    '''
    processes = [p for p in psutil.process_iter() if 'carla' in p.name().lower()]
    
    for process in processes:
        os.kill(process.pid, signal.SIGKILL)

class CarlaCore:
    '''
    CARLA Core class that performs the core functions of CARLA, such as
    initializing the server, connecting the client, spawning vehicles and
    sensors, starting and ending each scenario, and setting the spectator
    view.

    Args:
        config: dictionary of configuration parameters.
    '''
    def __init__(self, config = {}):
        self.config = config
        
        self.client = None
        self.world = None
        self.map = None

        self.door_status = [
            carla.VehicleDoor.FL,
            carla.VehicleDoor.FR,
            carla.VehicleDoor.RL,
            carla.VehicleDoor.RR,
            carla.VehicleDoor.All
        ]

        self.scene_info = {}
        self.scene_data = None

        self.scene_duration = 0.5

        self.init_server()
        self.connect_client()

    def __getstate__(self):
        logger.warning('No pickles for CARLA! Copyright © 2025 Goodarz Mehr')
    
    def init_server(self):
        '''
        Initialize CARLA server.
        '''
        # Start server on a random port.
        self.server_port = random.randint(15000, 32000)

        time.sleep(1.0)

        server_port_used = is_used(self.server_port)
        stream_port_used = is_used(self.server_port + 1)
        
        # Check if the server port is already being used, if so, add 2 to the
        # port number and check again.
        while server_port_used or stream_port_used:
            if server_port_used:
                logger.warning(f'Server port {self.server_port} is already being used.')
            if stream_port_used:
                logger.warning(f'Stream port {self.server_port + 1} is already being used.')

            self.server_port += 2
            
            server_port_used = is_used(self.server_port)
            stream_port_used = is_used(self.server_port + 1)

        # Create the CARLA server launch command.
        if self.config['show_display']:
            server_command = [
                f'{os.environ["CARLA_ROOT"]}/CarlaUE4.sh',
                '-nosound',
                '-windowed',
                f'-ResX={self.config["resolution_x"]}',
                f'-ResY={self.config["resolution_y"]}'
            ]
        else:
            server_command = [
                f'{os.environ["CARLA_ROOT"]}/CarlaUE4.sh',
                '-RenderOffScreen -nosound'
            ]

        server_command += [
            f'--carla-rpc-port={self.server_port}',
            f'-quality-level={self.config["quality_level"]}',
            f'-ini:[/Script/Engine.RendererSettings]:r.GraphicsAdapter={self.config["render_gpu"]}'
        ]

        server_command_text = ' '.join(map(str, server_command))
        
        logger.debug(server_command_text)
        
        server_process = subprocess.Popen(
            server_command_text,
            shell=True,
            preexec_fn=os.setsid,
            stdout=open(os.devnull, 'w')
        )
        
        time.sleep(20.0)
    
    def connect_client(self):
        '''
        Connect the client to the CARLA server.
        '''
        # Try connecting a client to the server.
        for i in range(self.config['retries_on_error']):
            try:
                logger.debug(f'Connecting to server on port {self.server_port}...')
                
                self.client = carla.Client(self.config['host'], self.server_port)
                
                self.client.set_timeout(self.config['timeout'])

                logger.debug('Connected to server.')

                return

            except Exception as e:
                logger.warning(f'Waiting for server to be ready: {e}, attempt {i + 1} of '
                               f'{self.config["retries_on_error"]}.')
                
                time.sleep(3.0)

        raise Exception('Cannot connect to CARLA server. Good bye!')
    
    def load_map(self, map_name):
        '''
        Load the desired map and apply the desired settings.

        Args:
            map_name: name of the map to load.
        '''
        logger.info(f'Loading {map_name}...')
        
        self.client.load_world(map_name)
        
        self.world = self.client.get_world()
        self.map = self.world.get_map()

        settings = self.world.get_settings()

        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.config['timestep']

        # If the selected map is Town12 or Town13 (large maps), limit tile
        # stream distance and actor active distance. If Town13 or Town15 is
        # selected, set max culling distance to 100.0, then revert back to 0.
        # This ensures faraway objects are rendered correctly.
        if map_name == 'Town12' or map_name == 'Town13':
            settings.tile_stream_distance = self.config['tile_stream_distance']
            settings.actor_active_distance = self.config['actor_active_distance']

        if map_name == 'Town13' or map_name == 'Town15':
            settings.max_culling_distance = 100.0

            self.world.apply_settings(settings)

            time.sleep(3.0)
        
        settings.max_culling_distance = 0.0

        self.world.apply_settings(settings)
        
        self.world.tick()

        logger.info(f'{map_name} loaded.')
    
        # Some objects obstruct the overhead or bottom-up view that is
        # necessary for collection of accurate ground truth data, so they are
        # removed from the map.
        logger.debug(f'Removing objects obstructing the overhead or bottom-up view from {map_name}...')

        if map_name == 'Town02':
            obstructing = [
                'Floor_',
                'Vh_Car_AudiA2_'
            ]
        elif map_name == 'Town03':
            obstructing = [
                'SM_GasStation01',
                'SM_Mansion02',
                'Sassafras_04_LOD27',
                'Custom_pine_beech_02_LOD1',
                'Veg_Tree_AcerSaccharum_v19',
                'Veg_Tree_AcerSaccharum_v20',
                'Japanese_Maple_01_LOD10',
                'Japanese_Maple_01_LOD11',
                'Japanese_Maple_01_LOD14',
                'SM_T03_RailTrain02',
                'BP_RepSpline5_Inst_0_0',
                'BP_RepSpline5_Inst_2_2',
                'BP_RepSpline6_',
                'Road_Road_Town03_1_'
            ]
        elif map_name == 'Town04':
            obstructing = [
                'SideWalkCube',
                'SM_GasStation01'
            ]
        elif map_name == 'Town05':
            obstructing = [
                'Plane',
                'SM_Awning117'
            ]
        elif map_name == 'Town07':
            obstructing = [
                'Cube'
            ]
        elif map_name == 'Town10HD':
            obstructing = [
                'SM_Tesla2',
                'SM_Tesla_2502',
                'SM_Mustang_prop2',
                'SM_Patrol2021Parked2',
                'SM_mercedescccParked2',
                'SM_LincolnMkz2017_prop',
                'Vh_Car_ToyotaPrius_NOrig',
                'InstancedFoliageActor_0_Inst_235_0',
                'InstancedFoliageActor_0_Inst_239_4',
                'InstancedFoliageActor_0_Inst_245_10',
                'InstancedFoliageActor_0_Inst_246_11',
                'InstancedFoliageActor_0_Inst_249_14',
                'InstancedFoliageActor_0_Inst_250_15',
                'InstancedFoliageActor_0_Inst_251_16',
                'InstancedFoliageActor_0_Inst_252_17',
                'InstancedFoliageActor_0_Inst_253_18',
                'InstancedFoliageActor_0_Inst_254_19',
                'InstancedFoliageActor_0_Inst_255_20',
                'InstancedFoliageActor_0_Inst_256_21',
                'InstancedFoliageActor_0_Inst_257_22',
                'InstancedFoliageActor_0_Inst_258_23',
                'InstancedFoliageActor_0_Inst_259_24',
                'InstancedFoliageActor_0_Inst_260_25',
                'InstancedFoliageActor_0_Inst_261_26',
                'InstancedFoliageActor_0_Inst_276_41',
                'InstancedFoliageActor_0_Inst_277_42'
            ]
        else:
            obstructing = []

        self.bad_crosswalks = [
            'Road_Crosswalk_Town03_59_',
            'Road_Crosswalk_Town04_28_',
            'Road_Crosswalk_Town04_29_',
            'Road_Crosswalk_Town04_30_',
            'Road_Crosswalk_Town07_5_',
            'Road_Crosswalk_Town07_8_',
            'Road_Crosswalk_Town07_9_'
        ]
        
        self.objects = self.world.get_environment_objects()

        to_remove = [obj.id for obj in self.objects if any(x in obj.name for x in obstructing)]

        self.world.enable_environment_objects(set(to_remove), False)

        self.world.tick()

        logger.debug(f'Objects obstructing the overhead or bottom-up view were removed from {map_name}.')
        
        # Generate waypoints.
        logger.debug('Generating waypoints...')

        self.map_name = map_name

        self.waypoints = self.map.generate_waypoints(self.config['waypoint_distance'])

        self.crosswalks = self.map.get_crosswalks()

        self.world.tick()

        logger.debug('Waypoints generated.')

        # Set up the Traffic Manager.
        logger.debug('Setting up Traffic Manager...')

        self.tm_port = self.server_port // 10 + self.server_port % 10

        while is_used(self.tm_port):
            logger.warning(f'Traffic Manager port {self.tm_port} is already being used. Checking the next one...')
            self.tm_port += 1
        
        self.traffic_manager = self.client.get_trafficmanager(self.tm_port)
        self.traffic_manager.set_synchronous_mode(True)

        logger.debug(f'Traffic Manager is connected to port {self.tm_port}.')

        # Get the ego vehicle blueprint and spawn points.
        logger.debug('Getting agent blueprint and spawn points...')

        self.bp = self.world.get_blueprint_library().filter(self.config['vehicle'])[0]

        self.bp.set_attribute('role_name', 'hero')
        self.bp.set_attribute('color', self.config['vehicle_color'])
        
        wp_all = self.map.generate_waypoints(self.config['spawn_point_separation_distance'])

        # Filter out spawn points that are in a junction or within 4 meters of
        # a junction.
        wp_mid = [wp for wp in wp_all if wp.next(4.0) != []]

        self.spawn_points = [
            wp.transform for wp in wp_mid if (wp.is_junction is False and wp.next(4.0)[0].is_junction is False)
        ]

        for sp in self.spawn_points:
            sp.location.z += 0.6

        self.spawn_points_backup = self.spawn_points
        
        if 'ego_vehicle_spawn_point' in self.config:
            if self.map_name in self.config['ego_vehicle_spawn_point']:
                spawn_point_list = self.config['ego_vehicle_spawn_point'][self.map_name]

                self.spawn_points = []

                for sp in spawn_point_list:
                    location = carla.Location(x=sp[0][0], y=sp[0][1], z=sp[0][2])
                    rotation = carla.Rotation(roll=sp[1][0], pitch=sp[1][1], yaw=sp[1][2])

                    self.spawn_points.append(carla.Transform(location, rotation))
        
        logger.debug(f'{len(self.spawn_points)} spawn points available.')

        logger.debug('Got agent blueprint and spawn points.')

        # Set data type, since calculations for larger maps require more
        # precision.
        if self.map_name in ['Town12', 'Town13', 'Town15']:
            self.dType = torch.double
        else:
            self.dType = torch.float
        
        # Get the Light Manager.
        logger.debug('Getting the Light Manager...')

        self.light_manager = self.world.get_lightmanager()

        self.street_light_intensity = self.light_manager.get_intensity(
            self.light_manager.get_all_lights(carla.LightGroup.Street)
        )

        logger.debug('Got Light Manager.')
    
    def spawn_vehicle(self):
        '''
        Spawn the ego vehicle and its sensors.
        '''
        try:
            self.scene_data = None
            self.scene_info = {}
            
            bev_fov = 2 * np.rad2deg(np.arctan(self.config['bev_dim'] * self.config['bev_res'] / 2000))

            self.config['bev_properties'] = {'fov': str(bev_fov)}
            
            # Instantiate the vehicle.
            logger.debug('Spawning the ego vehicle...')
            self.vehicle = None

            while self.vehicle is None:
                self.vehicle = self.world.try_spawn_actor(self.bp, random.choice(self.spawn_points))

            self.vehicle.set_autopilot(True, self.tm_port)
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
            self.scene_info['reckless_ego'] = False

            if self.config['reckless_ego']:
                p = self.config['reckless_ego_percentage'] / 100.0
                
                if np.random.choice(2, p=[1 - p, p]):
                    logger.warning('Ego vehicle is reckless!')

                    self.traffic_manager.ignore_lights_percentage(self.vehicle, 100.0)
                    self.traffic_manager.ignore_signs_percentage(self.vehicle, 100.0)
                    self.traffic_manager.ignore_vehicles_percentage(self.vehicle, 100.0)
                    self.traffic_manager.ignore_walkers_percentage(self.vehicle, 100.0)

                    self.scene_info['reckless_ego'] = True

            logger.debug('Ego vehicle behavior configured.')

            # Instantiate the Sensor Manager.
            logger.debug('Creating sensors...')

            self.sensor_manager = SensorManager(self, self.vehicle)

            # Set up camera locations.
            camera_location_front_left = carla.Transform(
                carla.Location(x=0.4, y=-0.4, z=1.6),
                carla.Rotation(yaw=-55.0)
            )
            camera_location_front = carla.Transform(carla.Location(x=0.6, y=0.0, z=1.6), carla.Rotation(yaw=0.0))
            camera_location_front_right = carla.Transform(
                carla.Location(x=0.4, y=0.4, z=1.6),
                carla.Rotation(yaw=55.0)
            )
            camera_location_back_left = carla.Transform(
                carla.Location(x=0.0, y=-0.4, z=1.6),
                carla.Rotation(yaw=-110)
            )
            camera_location_back = carla.Transform(carla.Location(x=-1.0, y=0.0, z=1.6), carla.Rotation(yaw=180.0))
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

        except Exception as e:
            logger.error(f'Error while spawning the vehicle: {e}')

            kill_all_servers()

            time.sleep(3.0)

            raise Exception('Cannot spawn the vehicle. Good bye!')
    
    def move_vehicle(self):
        '''
        Move the ego vehicle to a random spawn point and configure its
        behavior at the start of a new scene.
        '''
        try:
            self.scene_data = None
            self.scene_info = {}
            
            # Move the vehicle.
            logger.debug('Moving the ego vehicle...')

            self.vehicle.set_autopilot(False, self.tm_port)
            self.vehicle.enable_constant_velocity(carla.Vector3D(0.0, 0.0, 0.0))

            self.world.tick()

            time.sleep(1.0)

            self.vehicle.set_transform(random.choice(self.spawn_points))
            self.vehicle.disable_constant_velocity()
            self.vehicle.set_autopilot(True, self.tm_port)

            self.sensor_manager.reset()

            self.world.tick()

            self.set_spectator_view()

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

                    self.scene_info['reckless_ego'] = True
                
                else:
                    self.traffic_manager.ignore_lights_percentage(
                        self.vehicle,
                        self.config['ignore_lights_percentage']
                    )
                    self.traffic_manager.ignore_signs_percentage(self.vehicle, self.config['ignore_signs_percentage'])
                    self.traffic_manager.ignore_vehicles_percentage(
                        self.vehicle,
                        self.config['ignore_vehicles_percentage']
                    )
                    self.traffic_manager.ignore_walkers_percentage(
                        self.vehicle,
                        self.config['ignore_walkers_percentage']
                    )

                    self.scene_info['reckless_ego'] = False

            self.world.tick()

            logger.debug('Ego vehicle behavior configured.')

        except Exception as e:
            logger.error(f'Error while moving the vehicle: {e}')

            kill_all_servers()

            time.sleep(3.0)

            raise Exception('Cannot move the vehicle. Good bye!')
    
    def start_scene(self):
        '''
        Start a new scene.
        '''
        try:
            self.counter = 0
            self.warning_flag = False

            self.termination_counter = 0
            self.terminate_scene = False

            self.scene_info['map'] = self.map_name
            self.scene_info['vehicle'] = self.bp.id

            self.scene_info['expected_scene_duration'] = self.scene_duration
            self.scene_info['terminated_early'] = False

            self.light_change = False
            
            self.augment_waypoints()
            self.trim_crosswalks()
            self.setup_scenario()
            self.set_spectator_view()
        
        except Exception as e:
            logger.error(f'Error while starting the scene: {e}')

            kill_all_servers()

            time.sleep(3.0)

            raise Exception('Cannot start the scene. Good bye!')
    
    def process_sidewalk_points(self, wp):
        '''
        Process waypoints to get points on the sidewalks.

        Args:
            wp: waypoint to process.
        '''
        lwp = wp.get_left_lane()
        rwp = wp.get_right_lane()

        while lwp:
            if lwp.lane_type is not carla.LaneType.Driving:
                if lwp.lane_type == carla.LaneType.Sidewalk:
                    self.side_walk_points.append(lwp)

                lwp = lwp.get_left_lane()

                if lwp is not None and lwp.lane_type == carla.LaneType.NONE:
                    lwp = None
            else:
                lwp = None

        while rwp:
            if rwp.lane_type is not carla.LaneType.Driving:
                if rwp.lane_type == carla.LaneType.Sidewalk:
                    self.side_walk_points.append(rwp)

                rwp = rwp.get_right_lane()

                if rwp is not None and rwp.lane_type == carla.LaneType.NONE:
                    rwp = None
            else:
                rwp = None
    
    def augment_waypoints(self):
        '''
        Augment the list of waypoints generated by CARLA.
        '''
        # Filter the list of waypoints to get those near the spawn area.
        vehicle_location = self.vehicle.get_location()

        self.area_waypoints = [
            wp for wp in self.waypoints if vehicle_location.distance(
                wp.transform.location
            ) < self.config['mapping_area_radius']
        ]

        bad_lane_marking_types = [carla.LaneMarkingType.NONE, carla.LaneMarkingType.Grass, carla.LaneMarkingType.Curb]
        single_lane_marking_types = [
            carla.LaneMarkingType.Solid,
            carla.LaneMarkingType.Broken,
            carla.LaneMarkingType.Other,
            carla.LaneMarkingType.BottsDots
        ]

        # Get road line points from the area waypoints.
        self.road_line_points = []
        
        for wp in self.area_waypoints:
            wp_transform = wp.transform
            wp_transform.rotation.yaw += 90.0
            
            llm = wp.left_lane_marking

            if llm.type not in bad_lane_marking_types:
                if llm.type in single_lane_marking_types:
                    self.road_line_points.append((
                        wp_transform.location - 0.5 * wp.lane_width * wp_transform.get_forward_vector(),
                        llm.width
                    ))
                else:
                    self.road_line_points.append((
                        wp_transform.location - \
                            0.5 * (wp.lane_width - 2 * llm.width) * wp_transform.get_forward_vector(),
                        llm.width
                    ))
            
            if wp.get_right_lane() is not carla.LaneType.Driving:
                rlm = wp.right_lane_marking

                if rlm.type not in bad_lane_marking_types:
                    if rlm.type in single_lane_marking_types:
                        self.road_line_points.append((
                            wp_transform.location + 0.5 * wp.lane_width * wp_transform.get_forward_vector(),
                            rlm.width
                        ))
                    else:
                        self.road_line_points.append((
                            wp_transform.location + \
                                0.5 * (wp.lane_width - 2 * rlm.width) * wp_transform.get_forward_vector(),
                            rlm.width
                        ))     

        # Get sidewalk points from the area waypoints.
        self.side_walk_points = []

        for wp in self.area_waypoints:
            self.process_sidewalk_points(wp)
            
            if wp.is_junction:
                waypoint = self.map.get_waypoint(wp.transform.location, lane_type=carla.LaneType.Sidewalk)
                
                if waypoint is not None:
                    self.side_walk_points.append(waypoint)
                
                    self.process_sidewalk_points(waypoint)

        # The list of generated waypoints only includes those in lanes where
        # the lane type is Driving. Our goal is to add ones in the adjacent
        # lanes, if those lanes are Parking, Bidirectional, or Biking; or if
        # those lanes are NONE, Shoulder, or Border, conditioned on the fact
        # that at least one of the lane markings is not NONE, Grass, or Curb.
        good_lane_types = [carla.LaneType.Parking, carla.LaneType.Bidirectional, carla.LaneType.Biking]

        conditional_lane_types = [carla.LaneType.NONE, carla.LaneType.Shoulder, carla.LaneType.Border]

        adjacent_waypoints = []

        for wp in self.area_waypoints:
            lwp = wp.get_left_lane()
            rwp = wp.get_right_lane()

            while lwp:
                if lwp.lane_type in good_lane_types:
                    adjacent_waypoints.append(lwp)
                    lwp = lwp.get_left_lane()
                elif lwp.lane_type in conditional_lane_types:
                    if (lwp.left_lane_marking.type in bad_lane_marking_types and
                        lwp.right_lane_marking.type in bad_lane_marking_types):
                        lwp = None
                    else:
                        adjacent_waypoints.append(lwp)
                        lwp = lwp.get_left_lane()
                else:
                    lwp = None

            while rwp:
                if rwp.lane_type in good_lane_types:
                    adjacent_waypoints.append(rwp)
                    rwp = rwp.get_right_lane()
                elif rwp.lane_type in conditional_lane_types:
                    if (rwp.left_lane_marking.type in bad_lane_marking_types and
                        rwp.right_lane_marking.type in bad_lane_marking_types):
                        rwp = None
                    else:
                        adjacent_waypoints.append(rwp)
                        rwp = rwp.get_right_lane()
                else:
                    rwp = None
        
        self.area_waypoints += adjacent_waypoints

        self.world.tick()
    
    def trim_crosswalks(self):
        '''
        Trim the list of crosswalks to only include those that are within the
        mapping area radius.
        '''
        self.trimmed_crosswalks = []

        logger.debug('Trimming crosswalks...')

        vehicle_location = self.vehicle.get_location()
        
        for i in range(len(self.crosswalks)):
            for j in range(i + 1, len(self.crosswalks)):
                if self.crosswalks[i].distance(self.crosswalks[j]) < 0.02 and \
                    self.crosswalks[i].distance(vehicle_location) < self.config['mapping_area_radius']:
                    self.trimmed_crosswalks.append(self.crosswalks[i:j + 1])

        logger.debug('Crosswalks trimmed.')

    def configure_weather(self, weather):
        '''
        Configure the weather randomly.

        Args:
            weather: CARLA weather object to configure.
        '''

        weather.cloudiness = 100 * random.betavariate(0.8, 1.0)
        
        if weather.cloudiness <= 10.0:
            weather.cloudiness = 0.0

        weather.precipitation = random.betavariate(0.8, 0.2) * weather.cloudiness \
            if weather.cloudiness > 40.0 else 0.0

        if weather.precipitation <= 10.0:
            weather.precipitation = 0.0

        weather.precipitation_deposits = weather.precipitation + \
            random.betavariate(1.2, 1.6) * (100.0 - weather.precipitation)
        
        weather.wind_intensity = random.uniform(0.0, 100.0)

        weather.sun_azimuth_angle = random.uniform(0.0, 360.0)
        weather.sun_altitude_angle = 180 * random.betavariate(3.6, 2.0) - 90.0

        weather.wetness = min(100.0, max(random.gauss(weather.precipitation, 10.0), 0.0))

        weather.fog_density = 100 * random.betavariate(1.6, 2.0) if weather.cloudiness > 40.0 \
            or weather.sun_altitude_angle < 10.0 else 0.0
        
        if weather.fog_density <= 10.0:
            weather.fog_density = 0.0

        weather.fog_distance = random.lognormvariate(3.2, 0.8) if weather.fog_density > 10.0 else 100.0
        weather.fog_falloff = 5.0 * random.betavariate(1.2, 2.4) if weather.fog_density > 10.0 else 1.0

        if self.map_name == 'Town12' or self.map_name == 'Town13' or self.map_name == 'Town15':
            if weather.fog_density > 10.0:
                weather.fog_falloff = 0.01
        
        return weather
    
    def setup_scenario(self):
        '''
        Set up the scenario by configuring the weather, lights, and traffic.
        '''
        # Configure the weather.
        logger.debug('Configuring the weather...')

        initial_weather = self.world.get_weather()

        initial_weather = self.configure_weather(initial_weather)

        if 'initial_weather' in self.config:
            for attribute in initial_weather.__dir__():
                if attribute in self.config['initial_weather']:
                    initial_weather.__setattr__(attribute, self.config['initial_weather'][attribute])

        if self.config['weather_shift']:
            self.scene_info['weather_shift'] = True

            final_weather = self.world.get_weather()
        
            final_weather = self.configure_weather(final_weather)

            if 'final_weather' in self.config:
                for attribute in final_weather.__dir__():
                    if attribute in self.config['final_weather']:
                        final_weather.__setattr__(attribute, self.config['final_weather'][attribute])

            self.weather_increment = self.world.get_weather()

            num_steps = round(self.scene_duration / self.config['timestep'])

            for attribute in self.weather_increment.__dir__():
                if attribute in WEATHER_ATTRIBUTES:
                    self.weather_increment.__setattr__(
                        attribute,
                        (final_weather.__getattribute__(attribute) - initial_weather.__getattribute__(attribute)) \
                            / num_steps
                    )

        self.world.set_weather(initial_weather)

        logger.info(f'Initial weather...')
        logger.info(f'Cloudiness: {initial_weather.cloudiness:.2f}%, '
                    f'precipitation: {initial_weather.precipitation:4.2f}%, '
                    f'precipitation deposits: {initial_weather.precipitation_deposits:.2f}%.')
        logger.info(f'Wind intensity: {initial_weather.wind_intensity:.2f}%.')
        logger.info(f'Sun azimuth angle: {initial_weather.sun_azimuth_angle:.2f}°, '
                    f'sun altitude angle: {initial_weather.sun_altitude_angle:.2f}°.')
        logger.info(f'Wetness: {initial_weather.wetness:.2f}%.')
        logger.info(f'Fog density: {initial_weather.fog_density:.2f}%, '
                    f'fog distance: {initial_weather.fog_distance:.2f} m, '
                    f'fog falloff: {initial_weather.fog_falloff:.2f}.')

        initial_weather_parameters = {
            'cloudiness': initial_weather.cloudiness,
            'precipitation': initial_weather.precipitation,
            'precipitation_deposits': initial_weather.precipitation_deposits,
            'wind_intensity': initial_weather.wind_intensity,
            'sun_azimuth_angle': initial_weather.sun_azimuth_angle,
            'sun_altitude_angle': initial_weather.sun_altitude_angle,
            'wetness': initial_weather.wetness,
            'fog_density': initial_weather.fog_density,
            'fog_distance': initial_weather.fog_distance,
            'fog_falloff': initial_weather.fog_falloff
        }

        self.scene_info['initial_weather_parameters'] = initial_weather_parameters

        if self.config['weather_shift']:
            logger.info(f'Final weather...')
            logger.info(f'Cloudiness: {final_weather.cloudiness:.2f}%, '
                        f'precipitation: {final_weather.precipitation:4.2f}%, '
                        f'precipitation deposits: {final_weather.precipitation_deposits:.2f}%.')
            logger.info(f'Wind intensity: {final_weather.wind_intensity:.2f}%.')
            logger.info(f'Sun azimuth angle: {final_weather.sun_azimuth_angle:.2f}°, '
                        f'sun altitude angle: {final_weather.sun_altitude_angle:.2f}°.')
            logger.info(f'Wetness: {final_weather.wetness:.2f}%.')
            logger.info(f'Fog density: {final_weather.fog_density:.2f}%, '
                        f'fog distance: {final_weather.fog_distance:.2f} m, '
                        f'fog falloff: {final_weather.fog_falloff:.2f}.')
            
            final_weather_parameters = {
                'cloudiness': final_weather.cloudiness,
                'precipitation': final_weather.precipitation,
                'precipitation_deposits': final_weather.precipitation_deposits,
                'wind_intensity': final_weather.wind_intensity,
                'sun_azimuth_angle': final_weather.sun_azimuth_angle,
                'sun_altitude_angle': final_weather.sun_altitude_angle,
                'wetness': final_weather.wetness,
                'fog_density': final_weather.fog_density,
                'fog_distance': final_weather.fog_distance,
                'fog_falloff': final_weather.fog_falloff
            }

            self.scene_info['final_weather_parameters'] = final_weather_parameters

        logger.debug('Weather configured.')

        self.world.tick()

        time.sleep(1.0)

        # Configure the lights.
        logger.debug('Configuring the lights...')

        self.scene_info['street_light_intensity_change'] = 0.0

        if initial_weather.sun_altitude_angle < 0.0:
            self.configure_lights()

        logger.debug('Lights configured.')

        # Spawn NPCs.
        logger.debug('Spawning NPCs...')

        self.vehicle_location = self.vehicle.get_location()

        self.npc_spawn_points = [
            sp for sp in self.spawn_points_backup if self.vehicle_location.distance(
                sp.location
            ) < self.config['npc_spawn_radius']
        ]

        logger.debug(f'{len(self.npc_spawn_points)} NPC spawn points available.')

        if 'n_vehicles' in self.config:
            n_vehicles = self.config['n_vehicles']
            if n_vehicles == 27: logger.debug('rheM zradooG 4202 © thgirypoC')
        else:
            n_vehicles = random.randint(0, len(self.npc_spawn_points) - 3)
        
        if 'n_walkers' in self.config:
            n_walkers = self.config['n_walkers']
        else:
            n_walkers = random.randint(0, 640)
        
        self.spawn_npcs(n_vehicles, n_walkers)

        # In the new version of CARLA pedestrians are rendered invisible to
        # the lidar by default, this makes them visible.
        actors = self.world.get_actors()

        for actor in actors:
            if 'walker.pedestrian' in actor.type_id:
                actor.set_collisions(True)
                actor.set_simulate_physics(True)

        logger.debug('NPCs spawned.')

    def configure_lights(self):
        '''
        Configure the lights.
        '''
        street_lights = self.light_manager.get_all_lights(carla.LightGroup.Street)
        building_lights = self.light_manager.get_all_lights(carla.LightGroup.Building)

        if self.config['random_building_light_colors'] and self.map_name not in ['Town12', 'Town13', 'Town15']:
            for light in list(building_lights):
                color = carla.Color(r=random.randint(0, 255), g=random.randint(0, 255), b=random.randint(0, 255))

                self.light_manager.set_color([light], color)
            
        self.light_manager.turn_on(building_lights)

        self.scene_info['building_lights_on'] = True
        
        if self.config['change_street_light_intensity']:
            if 'street_light_intensity_change' in self.config:
                intensity_change = self.config['street_light_intensity_change']
            else:
                intensity_change = random.uniform(
                    -np.mean(self.street_light_intensity),
                    np.mean(self.street_light_intensity)
                )

            logger.info(f'Change in street light intensity: {intensity_change:.2f} lumens.')

            self.scene_info['street_light_intensity_change'] = intensity_change
            
            new_street_light_intensity = list(
                np.maximum(np.array(self.street_light_intensity) + intensity_change,
                            self.config['min_street_light_intensity'])
                )
            
            self.light_manager.set_intensities(street_lights, new_street_light_intensity)
            
        self.light_manager.turn_on(street_lights)

        self.scene_info['street_lights_on'] = True

        if self.config['random_street_light_failure']:
            p = self.config['street_light_failure_percentage'] / 100.0

            new_street_light_status = np.random.choice(2, len(street_lights), p=[p, 1 - p]).astype(bool).tolist()

            self.light_manager.set_active(street_lights, new_street_light_status)
        
        if self.config['turn_off_building_lights']:
            self.light_manager.turn_off(building_lights)

            self.scene_info['building_lights_on'] = False
        
        if self.config['turn_off_street_lights']:
            self.light_manager.turn_off(street_lights)

            self.scene_info['street_lights_on'] = False
    
    def spawn_npcs(self, n_vehicles, n_walkers):
        '''
        Spawn background vehicles and pedestrians.

        Args:
            n_vehicles: number of background vehicles to spawn.
            n_walkers: number of background pedestrians to spawn.
        '''
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        # Spawn vehicles.
        logger.info(f'Spawning {n_vehicles} vehicles...')

        n_spawn_points = len(self.npc_spawn_points)

        if n_vehicles < n_spawn_points:
            random.shuffle(self.npc_spawn_points)
        elif n_vehicles > n_spawn_points:
            logger.warning(f'{n_vehicles} vehicles were requested, but there were only {n_spawn_points} available '
                           'spawn points.')

            n_vehicles = n_spawn_points

        v_batch = []
        v_blueprints_all = self.world.get_blueprint_library().filter('vehicle.*')
        v_blueprints = [v for v in v_blueprints_all if v.get_attribute('has_lights').__bool__() == True]

        for n, transform in enumerate(self.npc_spawn_points):
            if n >= n_vehicles:
                break
            
            v_blueprint = random.choice(v_blueprints)
            
            # Randomly pick the color of the vehicle from the recommended
            # values.
            if v_blueprint.has_attribute('color'):
                v_blueprint.set_attribute(
                    'color',
                    random.choice(v_blueprint.get_attribute('color').recommended_values)
                )
            
            # Randomly pick the driver (for motorcycles and bicycles only)
            # from the recommended values. This does not work at the moment
            # but is instead implemented in the modified version of CARLA,
            # where the rider is selected randomly at the time of spawning.
            if v_blueprint.has_attribute('driver_id'):
                v_blueprint.set_attribute(
                    'driver_id',
                    random.choice(v_blueprint.get_attribute('driver_id').recommended_values)
                )
            
            v_blueprint.set_attribute('role_name', f'npc_vehicle_{n}')
            
            v_batch.append(SpawnActor(v_blueprint, transform).then(SetAutopilot(FutureActor, True, self.tm_port)))

        results = self.client.apply_batch_sync(v_batch, True)
        
        self.vehicles_id_list = [r.actor_id for r in results if not r.error]

        if len(self.vehicles_id_list) < n_vehicles:
            logger.warning(f'Could only spawn {len(self.vehicles_id_list)} of the {n_vehicles} requested vehicles.')

        self.world.tick()

        self.npc_vehicles_list = self.world.get_actors(self.vehicles_id_list)

        # Determine which vehicles are recless, i.e. ignore all traffic rules.
        # Also determine which emergency vehicles have their lights on.
        self.scene_info['n_reckless_vehicles'] = 0

        for vehicle in self.npc_vehicles_list:
            self.traffic_manager.update_vehicle_lights(vehicle, True)

            if any(x in vehicle.type_id for x in ['firetruck', 'ambulance', 'police']):
                p = self.config['emergency_lights_percentage'] / 100.0
                
                if np.random.choice(2, p=[1 - p, p]):
                    vehicle.set_light_state(carla.VehicleLightState.Special1)
            
            self.traffic_manager.ignore_lights_percentage(vehicle, self.config['ignore_lights_percentage'])
            self.traffic_manager.ignore_signs_percentage(vehicle, self.config['ignore_signs_percentage'])
            self.traffic_manager.ignore_vehicles_percentage(vehicle, self.config['ignore_vehicles_percentage'])
            self.traffic_manager.ignore_walkers_percentage(vehicle, self.config['ignore_walkers_percentage'])
            
            if self.config['reckless_npc']:
                p = self.config['reckless_npc_percentage'] / 100.0
                
                if np.random.choice(2, p=[1 - p, p]):
                    logger.warning(f'{vehicle.attributes["role_name"]} is reckless!')
                    
                    self.traffic_manager.ignore_lights_percentage(vehicle, 100.0)
                    self.traffic_manager.ignore_signs_percentage(vehicle, 100.0)
                    self.traffic_manager.ignore_vehicles_percentage(vehicle, 100.0)
                    self.traffic_manager.ignore_walkers_percentage(vehicle, 100.0)

                    self.scene_info['n_reckless_vehicles'] += 1

        logger.info(f'{len(self.vehicles_id_list)} vehicles spawned.')

        self.npc_door_open_list = []
        self.tried_to_open_door_list = []

        time.sleep(1.0)

        self.world.tick()

        # Configure the Traffic Manager.
        logger.debug('Configuring Traffic Manager...')

        speed_difference = None
        distance_to_leading = None
        green_time = None

        if 'speed_difference' in self.config:
            speed_difference = self.config['speed_difference']

            self.traffic_manager.global_percentage_speed_difference(speed_difference)

            logger.info(f'Global percentage speed difference: {speed_difference:.2f}%.')
        else:
            self.traffic_manager.vehicle_percentage_speed_difference(self.vehicle, random.uniform(-40.0, 20.0))

            for vehicle in self.npc_vehicles_list:
                self.traffic_manager.vehicle_percentage_speed_difference(vehicle, random.uniform(-40.0, 20.0))

        if 'distance_to_leading' in self.config:
            distance_to_leading = self.config['distance_to_leading']

            self.traffic_manager.set_global_distance_to_leading_vehicle(distance_to_leading)

            logger.info(f'Global minimum distance to leading vehicle: {distance_to_leading:.2f} m.')
        else:
            self.traffic_manager.distance_to_leading_vehicle(self.vehicle, random.gauss(3.2, 1.0))

            for vehicle in self.npc_vehicles_list:
                self.traffic_manager.distance_to_leading_vehicle(vehicle, random.gauss(3.2, 1.0))
        
        if 'green_time' in self.config:
            green_time = self.config['green_time']

            actor_list = self.world.get_actors()
    
            for actor in actor_list:
                if isinstance(actor, carla.TrafficLight):
                    actor.set_green_time(green_time)

            logger.info(f'Traffic light green time: {green_time:.2f} s.')
        else:
            actor_list = self.world.get_actors()

            for actor in actor_list:
                if isinstance(actor, carla.TrafficLight):
                    actor.set_green_time(random.uniform(4.0, 28.0))

        traffic_parameters = {
            'speed_difference': speed_difference,
            'distance_to_leading': distance_to_leading,
            'green_time': green_time
        }

        self.scene_info['traffic_parameters'] = traffic_parameters

        logger.debug('Traffic Manager configured.')

        time.sleep(1.0)

        # Spawn walkers.
        logger.info(f'Spawning {n_walkers} walkers...')

        if 'walker_cross_factor' in self.config:
            cross_factor = self.config['walker_cross_factor']
        else:
            cross_factor = random.betavariate(2.4, 1.6)
        
        self.world.set_pedestrians_cross_factor(cross_factor)

        self.scene_info['traffic_parameters']['walker_cross_factor'] = cross_factor

        logger.info(f'Walker cross factor: {cross_factor:.2f}.')

        # Get spawn locations that are close to the ego vehicle.
        spawn_locations = []
        
        for _ in range(n_walkers):
            counter = 0
            
            spawn_location = None

            while spawn_location is None and counter < self.config['walker_spawn_attempts']:
                spawn_location = self.world.get_random_location_from_navigation()

                if spawn_location is not None:
                    if self.vehicle_location.distance(spawn_location) < self.config['npc_spawn_radius']:
                        spawn_locations.append(spawn_location)
                    else:
                        spawn_location = None

                counter += 1

        w_batch = []
        w_blueprints = self.world.get_blueprint_library().filter('walker.pedestrian.*')

        for spawn_location in spawn_locations:
            w_blueprint = random.choice(w_blueprints)
            
            if w_blueprint.has_attribute('is_invincible'):
                w_blueprint.set_attribute('is_invincible', 'false')

            w_blueprint.set_attribute('role_name', 'npc_walker')
            
            w_batch.append(SpawnActor(w_blueprint, carla.Transform(spawn_location)))

        results = self.client.apply_batch_sync(w_batch, True)
            
        self.walkers_id_list = [r.actor_id for r in results if not r.error]

        if len(self.walkers_id_list) < n_walkers:
            logger.warning(f'Could only spawn {len(self.walkers_id_list)} of the {n_walkers} requested walkers.')

        self.walkers_list = self.world.get_actors(self.walkers_id_list)

        logger.info(f'{len(self.walkers_id_list)} walkers spawned.')

        self.scene_info['n_vehicles'] = len(self.vehicles_id_list)
        self.scene_info['n_walkers'] = len(self.walkers_id_list)

        self.world.tick()

        time.sleep(1.0)

        # Spawn walker controllers.
        logger.debug('Spawning walker controllers...')

        wc_batch = []
        wc_blueprint = self.world.get_blueprint_library().find('controller.ai.walker')

        for walker_id in self.walkers_id_list:
            wc_batch.append(SpawnActor(wc_blueprint, carla.Transform(), walker_id))

        results = self.client.apply_batch_sync(wc_batch, True)

        self.controllers_id_list = [r.actor_id for r in results if not r.error]

        if len(self.controllers_id_list) < len(self.walkers_id_list):
            logger.warning(f'Only {len(self.controllers_id_list)} of the {len(self.walkers_id_list)} controllers '
                           'could be created. Some walkers may be frozen.')

        self.world.tick()

        for controller in self.world.get_actors(self.controllers_id_list):
            controller.start()
            controller.set_max_speed(max(random.lognormvariate(0.16, 0.64), self.config['walker_speed_min']))

            counter = 0

            go_to_location = None

            while go_to_location is None and counter < self.config['walker_spawn_attempts']:
                go_to_location = self.world.get_random_location_from_navigation()

                if go_to_location is not None:
                    if self.vehicle_location.distance(go_to_location) >= self.config['npc_spawn_radius']:
                        go_to_location = None

                counter += 1

            if go_to_location is not None:
                controller.go_to_location(go_to_location)
        
        self.world.tick()

        self.controllers_list = self.world.get_actors(self.controllers_id_list)

        logger.debug('Walker controllers spawned.')

    def set_spectator_view(self):
        '''
        Set the spectator view to follow the ego vehicle.
        '''
        # Get the ego vehicle's coordinates.
        transform = self.vehicle.get_transform()

        # Calculate the spectator's desired position.
        view_x = transform.location.x - 8.0 * transform.get_forward_vector().x
        view_y = transform.location.y - 8.0 * transform.get_forward_vector().y
        view_z = transform.location.z + 4.0

        # Calculate the spectator's desired orientation.
        view_roll = transform.rotation.roll
        view_pitch = transform.rotation.pitch - 16.0
        view_yaw = transform.rotation.yaw

        # Get the spectator and place it in the calculated position.
        spectator = self.world.get_spectator()
        
        spectator.set_transform(
            carla.Transform(
                carla.Location(x=view_x, y=view_y, z=view_z),
                carla.Rotation(roll=view_roll, pitch=view_pitch, yaw=view_yaw)
            )
        )
    
    def tick(self, path=None, scene=None, frame=None, render=False, save=False):
        '''
        Proceed for one time step.

        Args:
            path: root directory of the dataset.
            scene: scene number.
            frame: frame number.
            render: whether to render sensor data.
            save: whether to save sensor data to file.
        '''
        # Clear all sensor queues before proceeding.
        self.sensor_manager.clear_queues()

        # Randomly open the door of some vehicles that are stopped, then close
        # them when the vehicles start moving.
        p = self.config['door_open_percentage'] / 100.0

        for vehicle in self.npc_vehicles_list:
            if vehicle.attributes['has_dynamic_doors'] == 'true':
                role_name = vehicle.attributes['role_name']

                if role_name not in self.npc_door_open_list and role_name not in self.tried_to_open_door_list \
                    and vehicle.get_velocity().length() < 0.1:
                    
                    if np.random.choice(2, p=[1 - p, p]):
                        vehicle.open_door(random.choice(self.door_status))
                        self.npc_door_open_list.append(role_name)
                    else:
                        self.tried_to_open_door_list.append(role_name)
                
                elif role_name in self.npc_door_open_list and vehicle.get_velocity().length() > 1.0:
                    vehicle.close_door(carla.VehicleDoor.All)
                    self.npc_door_open_list.remove(role_name)
                
                elif role_name in self.tried_to_open_door_list and vehicle.get_velocity().length() > 1.0:
                    self.tried_to_open_door_list.remove(role_name)
        
        # Change the weather if configured to do so.
        if self.config['weather_shift'] and scene is not None:
            weather = self.world.get_weather()

            old_sun_altitude_angle = weather.sun_altitude_angle

            for attribute in weather.__dir__():
                if attribute in WEATHER_ATTRIBUTES:
                    weather.__setattr__(
                        attribute,
                        weather.__getattribute__(attribute) + self.weather_increment.__getattribute__(attribute)
                    )
            
            new_sun_altitude_angle = weather.sun_altitude_angle

            if self.light_change:
                self.light_manager.set_day_night_cycle(True)
                
                self.light_change = False
            
            if old_sun_altitude_angle >= 0.0 and new_sun_altitude_angle < 0.0:
                self.configure_lights()
                
                self.light_manager.set_day_night_cycle(False)
                
                self.light_change = True
            
            self.world.set_weather(weather)
        
        # Proceed for one time step.
        self.world.tick()

        self.set_spectator_view()
        
        if render or save:
            self.vehicle_location = self.vehicle.get_location()

            self.trim_waypoints()

            # If nearby roads do not have an elevation difference of more than
            # 6.4 meters, get the ground truth using the top and bottom
            # semantic cameras and road waypoints. Otherwise (i.e. when near
            # an overpass/underpass), get the ground truth using bounding
            # boxes and road waypoints.
            if self.map_name not in ['Town04', 'Town05', 'Town12', 'Town13']:
                self.get_bev_gt()
                self.warning_flag = False
            elif (torch.max(self.nwp_loc[:, 2]) - torch.min(self.nwp_loc[:, 2])).numpy() < 6.4:
                self.get_bev_gt()
                self.warning_flag = False
            else:
                mid = (torch.max(self.nwp_loc[:, 2]) + torch.min(self.nwp_loc[:, 2])) / 2.0

                highs = self.nwp_loc[self.nwp_loc[:, 2] > (mid + 3.2)]
                lows = self.nwp_loc[self.nwp_loc[:, 2] < (mid - 3.2)]

                dists = torch.cdist(highs[:, :2], lows[:, :2])

                if torch.min(dists) > 48.0:
                    self.get_bev_gt()
                    self.warning_flag = False
                else:
                    self.get_bounding_boxes()
                    self.get_bev_gt_alt()

                    if self.warning_flag is False:
                        logger.warning('Using alternative ground truth generation method due to elevation difference '
                                       'in the road.')

                        self.warning_flag = True

            background = (255, 255, 255)
    
            canvas = np.zeros((self.config['bev_dim'], self.config['bev_dim'], 3), dtype=np.uint8)
            canvas[:] = background

            if self.config['use_cityscapes_palette']:
                PALETTE = CITYSCAPE_PALETTE
            else:
                PALETTE = MAP_PALETTE
            
            for k, name in enumerate(PALETTE):
                canvas[self.bev_gt[k], :] = PALETTE[name]

            canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        
        # Render sensor data.
        if render:
            self.sensor_manager.render()

            cv2.imshow('Ground Truth', canvas)
            cv2.waitKey(1)
        
        # Save sensor data to file.
        if save and all(v is not None for v in [path, scene, frame]):
            self.sensor_manager.save(path, scene, frame)

            with open(
                f'{path}/simbev/ground-truth/seg/SimBEV-scene-{scene:04d}-frame-{frame:04d}-GT_SEG.npz',
                'wb'
            ) as f:
                np.savez_compressed(f, data=self.bev_gt)

            cv2.imwrite(f'{path}/simbev/ground-truth/seg_viz/SimBEV-scene-{scene:04d}-frame-{frame:04d}-GT_SEG_VIZ.jpg',
                        canvas)
            
            self.get_bounding_boxes()
            
            with open(
                f'{path}/simbev/ground-truth/det/SimBEV-scene-{scene:04d}-frame-{frame:04d}-GT_DET.bin',
                'wb'
            ) as f:
                np.save(f, np.array(self.actors), allow_pickle=True)

            self.get_hd_map_info()

            with open(
                f'{path}/simbev/ground-truth/hd_map/SimBEV-scene-{scene:04d}-frame-{frame:04d}-HD_MAP.json',
                'w'
            ) as f:
                json.dump(self.hd_map_info, f, indent=4)

        # Decide whether to terminate the scene.
        if scene is not None and self.config['early_scene_termination']:
            if self.vehicle.get_velocity().length() < 0.1:
                self.termination_counter += 1
            else:
                self.termination_counter = 0
            
            if self.termination_counter * self.config['timestep'] >= self.config['termination_timeout']:
                self.terminate_scene = True
    
    def trim_waypoints(self):
        '''
        Trim the list of waypoints and only leave those within the ego
        vehicle's perception range so ground truth calculations are more
        efficient.
        '''
        vehicle_location = self.vehicle.get_location()

        # Get the list of nearby waypoints and their lane widths, road line
        # locations and their lane widths, and sidewalk locations and their
        # lane widths.
        if self.counter % round(0.5 / self.config['timestep']) == 0:
            self.nwp = [
                wp for wp in self.area_waypoints if vehicle_location.distance(
                    wp.transform.location
                ) < self.config['nearby_mapping_area_radius']
            ]

            self.nwp_loc = []
            self.nwp_lw = []

            for wp in self.nwp:
                self.nwp_loc.append(wp.transform.location)
                self.nwp_lw.append(wp.lane_width)

            self.nwp_loc = carla_vector_to_torch(self.nwp_loc)
            self.nwp_lw = torch.Tensor(self.nwp_lw)

            self.nlm = [
                lm for lm in self.road_line_points if vehicle_location.distance(
                    lm[0]
                ) < self.config['nearby_mapping_area_radius']
            ]

            self.nlm_loc = []
            self.nlm_lw = []

            for lm in self.nlm:
                self.nlm_loc.append(lm[0])
                self.nlm_lw.append(lm[1])

            self.nlm_loc = carla_vector_to_torch(self.nlm_loc)
            self.nlm_lw = torch.Tensor(self.nlm_lw)

            self.nsw = [
                sw for sw in self.side_walk_points if vehicle_location.distance(
                    sw.transform.location
                ) < self.config['nearby_mapping_area_radius']
            ]

            self.nsw_loc = []
            self.nsw_lw = []

            for sw in self.nsw:
                self.nsw_loc.append(sw.transform.location)
                self.nsw_lw.append(sw.lane_width)

            self.nsw_loc = carla_vector_to_torch(self.nsw_loc)
            self.nsw_lw = torch.Tensor(self.nsw_lw)

        # Torch can end up hogging GPU memory when calculating the ground
        # truth, so empty it every once in a while.
        if self.counter % 40 == 5:
            with torch.cuda.device(f'cuda:{self.config["cuda_gpu"]}'):
                torch.cuda.empty_cache()
        
        self.counter += 1
    
    def get_bev_gt(self):
        '''
        Get the BEV ground truth using the top and bottom semantic cameras and
        road waypoints.
        '''
        vehicle_transform = self.vehicle.get_transform()

        # Get the road mask from waypoints.
        wp_road_mask = get_road_mask(
            self.nwp_loc,
            self.nwp_lw,
            vehicle_transform.location,
            vehicle_transform.rotation,
            self.config['bev_dim'],
            self.config['bev_res'],
            device=f'cuda:{self.config["cuda_gpu"]}',
            dType=self.dType
        )

        # Get the road line mask from road line points.
        road_line_mask = get_road_mask(
            self.nlm_loc,
            self.nlm_lw * 4.8,
            vehicle_transform.location,
            vehicle_transform.rotation,
            self.config['bev_dim'],
            self.config['bev_res'],
            device=f'cuda:{self.config["cuda_gpu"]}',
            dType=self.dType
        ).detach().cpu().numpy().astype(bool)

        # Get the sidewalk mask from sidewalk points.
        wp_sidewalk_mask = get_road_mask(
            self.nsw_loc,
            self.nsw_lw,
            vehicle_transform.location,
            vehicle_transform.rotation,
            self.config['bev_dim'],
            self.config['bev_res'],
            device=f'cuda:{self.config["cuda_gpu"]}',
            dType=self.dType
        ).detach().cpu().numpy().astype(bool)

        # For Town06, get the sidewalk mask from sidewalk meshes.
        if self.map_name == 'Town06':
            mesh_sidewalk_mask = np.zeros((self.config['bev_dim'], self.config['bev_dim'])).astype(bool)

            all_sidewalks = [obj for obj in self.objects if '_Sidewalk_' in obj.name]

            for sidewalk in all_sidewalks:
                bbox = sidewalk.bounding_box.get_local_vertices()

                bbox = carla_vector_to_torch(bbox)

                bbox[:, 1] *= -1.0

                resulting_mask = get_object_mask(
                    bbox,
                    vehicle_transform.location,
                    vehicle_transform.rotation,
                    self.config['bev_dim'],
                    self.config['bev_res'],
                    device=f'cuda:{self.config["cuda_gpu"]}',
                    dType=self.dType
                )

                mesh_sidewalk_mask = np.logical_or(mesh_sidewalk_mask, resulting_mask).numpy().astype(bool)

        # Get the crosswalk mask from crosswalk locations.
        crosswalk_mask = np.zeros((self.config['bev_dim'], self.config['bev_dim'])).astype(bool)

        if self.map_name in ['Town03', 'Town04', 'Town05', 'Town06', 'Town07', 'Town10HD']:
            all_crosswalks = [obj for obj in self.objects if '_Crosswalk_' in obj.name]

            crosswalks = [cw for cw in all_crosswalks if not any(x in cw.name for x in self.bad_crosswalks)]

            for crosswalk in crosswalks:
                bbox = crosswalk.bounding_box.get_local_vertices()

                crosswalk_box = carla_vector_to_torch([bbox[0], bbox[2], bbox[6], bbox[4], bbox[0]])

                crosswalk_box[:, 1] *= -1.0

                resulting_mask = get_crosswalk_mask(
                    crosswalk_box,
                    vehicle_transform.location,
                    vehicle_transform.rotation,
                    self.config['bev_dim'],
                    self.config['bev_res'],
                    device=f'cuda:{self.config["cuda_gpu"]}',
                    dType=self.dType
                )

                crosswalk_mask = np.logical_or(crosswalk_mask, resulting_mask).numpy().astype(bool)
        else:
            for crosswalk in self.trimmed_crosswalks:
                crosswalk = carla_vector_to_torch(crosswalk)

                crosswalk[:, 1] *= -1.0

                resulting_mask = get_crosswalk_mask(
                    crosswalk,
                    vehicle_transform.location,
                    vehicle_transform.rotation,
                    self.config['bev_dim'],
                    self.config['bev_res'],
                    device=f'cuda:{self.config["cuda_gpu"]}',
                    dType=self.dType
                )

                crosswalk_mask = np.logical_or(crosswalk_mask, resulting_mask).numpy().astype(bool)

        # Get images from the top and bottom semantic cameras. Use the top
        # image along with the road mask from waypoints to create a road mask,
        # and use both images to create masks for cars, trucks, buses,
        # motorcycles, bicycles, riders, and pedestrians.
        top_bev_image = self.sensor_manager.semantic_bev_camera_list[0].save_queue.get(True, 10.0)
        bottom_bev_image = np.flip(self.sensor_manager.semantic_bev_camera_list[1].save_queue.get(True, 10.0), axis=0)

        bev_road_mask = np.logical_or(top_bev_image[:, :, 2] == 128, top_bev_image[:, :, 2] == 157)

        road_mask = binary_closing(np.logical_or(wp_road_mask, bev_road_mask))

        road_line_mask = binary_closing(road_line_mask, footprint=np.ones((3, 3)))
        
        if self.config['use_bev_for_sidewalks']:
            sidewalk_mask = binary_closing(np.logical_or(top_bev_image[:, :, 0] == 232, wp_sidewalk_mask))
        elif self.map_name == 'Town06':
            sidewalk_mask = binary_closing(np.logical_or(wp_sidewalk_mask, np.logical_and(
                mesh_sidewalk_mask, np.logical_or(top_bev_image[:, :, 0] == 232, bottom_bev_image[:, :, 0] == 232)
            )))
        else:
            sidewalk_mask = binary_closing(wp_sidewalk_mask)

        if self.map_name in ['Town12', 'Town13']:
            bev_crosswalk_mask = binary_opening(
                np.logical_or(top_bev_image[:, :, 2] == 157, crosswalk_mask),
                footprint=np.ones((3, 3))
            )
        elif self.map_name in ['Town15']:
            bev_crosswalk_mask = np.logical_or(
                np.logical_and(
                    binary_dilation(crosswalk_mask, footprint=np.ones((11, 11))),
                    np.logical_or(top_bev_image[:, :, 2] == 157, top_bev_image[:, :, 2] == 110),
                ),
                np.logical_and(crosswalk_mask, top_bev_image[:, :, 1] == 70)
            )
            bev_crosswalk_mask = binary_opening(binary_closing(bev_crosswalk_mask), footprint=np.ones((3, 3)))
            bev_crosswalk_mask = binary_closing(
                np.logical_or(bev_crosswalk_mask, ~road_mask), footprint=np.ones((4, 4))
            )
        else:
            bev_crosswalk_mask = crosswalk_mask

        crosswalk_mask = binary_closing(np.logical_and(bev_crosswalk_mask, road_mask))

        car_mask = np.logical_or(bottom_bev_image[:, :, 0] == 142, top_bev_image[:, :, 0] == 142)
        truck_mask = np.logical_or(
            np.logical_and(bottom_bev_image[:, :, 0] == 70, bottom_bev_image[:, :, 1] == 0),
            np.logical_and(top_bev_image[:, :, 0] == 70, top_bev_image[:, :, 1] == 0)
        )
        bus_mask = np.logical_or(
            np.logical_and(bottom_bev_image[:, :, 0] == 100, bottom_bev_image[:, :, 1] == 60),
            np.logical_and(top_bev_image[:, :, 0] == 100, top_bev_image[:, :, 1] == 60)
        )
        motorcycle_mask = np.logical_or(bottom_bev_image[:, :, 0] == 230, top_bev_image[:, :, 0] == 230)
        bicycle_mask = np.logical_or(bottom_bev_image[:, :, 0] == 32, top_bev_image[:, :, 0] == 32)
        rider_mask = np.logical_or(bottom_bev_image[:, :, 2] == 255, top_bev_image[:, :, 2] == 255)
        pedestrian_mask = np.logical_or(bottom_bev_image[:, :, 1] == 20, top_bev_image[:, :, 1] == 20)

        # Concatenate individual masks to get the ground truth.
        self.bev_gt = np.array([
            road_mask,
            road_line_mask,
            sidewalk_mask,
            crosswalk_mask,
            car_mask,
            truck_mask,
            bus_mask,
            motorcycle_mask,
            bicycle_mask,
            rider_mask,
            pedestrian_mask
        ])
    
    def get_bev_gt_alt(self):
        '''
        Get the BEV ground truth using object bounding boxes and road
        waypoints.
        '''
        vehicle_transform = self.vehicle.get_transform()

        # Find waypoints whose elevation difference with the ego vehicle is
        # less than a certain threshold.
        mask = (self.nwp_loc[:, 2] - vehicle_transform.location.z).abs() < 4.8
        
        # Get the road mask from waypoints.
        wp_road_mask = get_road_mask(
            self.nwp_loc[mask],
            self.nwp_lw[mask],
            vehicle_transform.location,
            vehicle_transform.rotation,
            self.config['bev_dim'],
            self.config['bev_res'],
            device=f'cuda:{self.config["cuda_gpu"]}',
            dType=self.dType
        )

        road_mask = binary_closing(wp_road_mask)

        # Find road line points whose elevation difference with the ego
        # vehicle is less than a certain threshold.
        mask = (self.nlm_loc[:, 2] - vehicle_transform.location.z).abs() < 4.8

        # Get the road line mask from road line points.
        road_line_mask = get_road_mask(
            self.nlm_loc[mask],
            self.nlm_lw[mask] * 4.8,
            vehicle_transform.location,
            vehicle_transform.rotation,
            self.config['bev_dim'],
            self.config['bev_res'],
            device=f'cuda:{self.config["cuda_gpu"]}',
            dType=self.dType
        ).detach().cpu().numpy().astype(bool)

        road_line_mask = binary_closing(road_line_mask, footprint=np.ones((3, 3)))

        # Find sidewalk points whose elevation difference with the ego vehicle
        # is less than a certain threshold.
        mask = (self.nsw_loc[:, 2] - vehicle_transform.location.z).abs() < 4.8

        # Get the sidewalk mask from sidewalk points.
        sidewalk_mask = get_road_mask(
            self.nsw_loc[mask],
            self.nsw_lw[mask],
            vehicle_transform.location,
            vehicle_transform.rotation,
            self.config['bev_dim'],
            self.config['bev_res'],
            device=f'cuda:{self.config["cuda_gpu"]}',
            dType=self.dType
        ).detach().cpu().numpy().astype(bool)

        sidewalk_mask = binary_closing(sidewalk_mask)

        # Get the crosswalk mask from crosswalk locations.
        crosswalk_mask = np.zeros((self.config['bev_dim'], self.config['bev_dim'])).astype(bool)

        if self.map_name in ['Town03', 'Town04', 'Town05', 'Town06', 'Town07', 'Town10HD']:
            all_crosswalks = [obj for obj in self.objects if '_Crosswalk_' in obj.name]

            crosswalks = [cw for cw in all_crosswalks if not any(x in cw.name for x in self.bad_crosswalks)]

            for crosswalk in crosswalks:
                bbox = crosswalk.bounding_box.get_local_vertices()

                if np.abs(bbox[0].z - vehicle_transform.location.z) < 4.8:
                    crosswalk_box = carla_vector_to_torch([bbox[0], bbox[2], bbox[6], bbox[4], bbox[0]])

                    crosswalk_box[:, 1] *= -1.0

                    resulting_mask = get_crosswalk_mask(
                        crosswalk_box,
                        vehicle_transform.location,
                        vehicle_transform.rotation,
                        self.config['bev_dim'],
                        self.config['bev_res'],
                        device=f'cuda:{self.config["cuda_gpu"]}',
                        dType=self.dType
                    )

                    crosswalk_mask = np.logical_or(crosswalk_mask, resulting_mask).numpy().astype(bool)
        else:
            for crosswalk in self.trimmed_crosswalks:
                if np.abs(crosswalk[0].z - vehicle_transform.location.z) < 4.8:
                    crosswalk = carla_vector_to_torch(crosswalk)

                    crosswalk[:, 1] *= -1.0

                    resulting_mask = get_crosswalk_mask(
                        crosswalk,
                        vehicle_transform.location,
                        vehicle_transform.rotation,
                        self.config['bev_dim'],
                        self.config['bev_res'],
                        device=f'cuda:{self.config["cuda_gpu"]}',
                        dType=self.dType
                    )

                    crosswalk_mask = np.logical_or(crosswalk_mask, resulting_mask).numpy().astype(bool)

        crosswalk_mask = binary_closing(np.logical_and(crosswalk_mask, road_mask))
        
        # Get object masks from bounding boxes.
        car_mask = np.zeros((self.config['bev_dim'], self.config['bev_dim'])).astype(bool)
        truck_mask = np.zeros((self.config['bev_dim'], self.config['bev_dim'])).astype(bool)
        bus_mask = np.zeros((self.config['bev_dim'], self.config['bev_dim'])).astype(bool)
        motorcycle_mask = np.zeros((self.config['bev_dim'], self.config['bev_dim'])).astype(bool)
        bicycle_mask = np.zeros((self.config['bev_dim'], self.config['bev_dim'])).astype(bool)
        rider_mask = np.zeros((self.config['bev_dim'], self.config['bev_dim'])).astype(bool)
        pedestrian_mask = np.zeros((self.config['bev_dim'], self.config['bev_dim'])).astype(bool)

        # Iterate over all bounding boxes and add the mask for each object to
        # the corresponding mask.
        for actor in self.actors:
            if any(x in actor['semantic_tags'] for x in [12, 13, 14, 15, 16, 18, 19]):
                if np.abs(actor['bounding_box'][:, 2] - vehicle_transform.location.z).max() < 4.8:
                    resulting_mask = get_object_mask(
                        actor['bounding_box'],
                        vehicle_transform.location,
                        vehicle_transform.rotation,
                        self.config['bev_dim'],
                        self.config['bev_res'],
                        device=f'cuda:{self.config["cuda_gpu"]}',
                        dType=self.dType
                    )
                    
                    if 12 in actor['semantic_tags']:
                        pedestrian_mask = np.logical_or(pedestrian_mask, resulting_mask).numpy().astype(bool)
                    if 13 in actor['semantic_tags']:
                        rider_mask = np.logical_or(rider_mask, resulting_mask).numpy().astype(bool)
                    if 14 in actor['semantic_tags']:
                        car_mask = np.logical_or(car_mask, resulting_mask).numpy().astype(bool)
                    if 15 in actor['semantic_tags']:
                        truck_mask = np.logical_or(truck_mask, resulting_mask).numpy().astype(bool)
                    if 16 in actor['semantic_tags']:
                        bus_mask = np.logical_or(bus_mask, resulting_mask).numpy().astype(bool)
                    if 18 in actor['semantic_tags']:
                        motorcycle_mask = np.logical_or(motorcycle_mask, resulting_mask).numpy().astype(bool)
                    if 19 in actor['semantic_tags']:
                        bicycle_mask = np.logical_or(bicycle_mask, resulting_mask).numpy().astype(bool)
        
        # Concatenate individual masks to get the ground truth.
        self.bev_gt = np.array([
            road_mask,
            road_line_mask,
            sidewalk_mask,
            crosswalk_mask,
            car_mask,
            truck_mask,
            bus_mask,
            motorcycle_mask,
            bicycle_mask,
            rider_mask,
            pedestrian_mask
        ])
    
    def get_bounding_boxes(self):
        '''
        Get the bounding box of actors (including traffic elements) in the
        scene that are within a certain radius of the ego vehicle.
        '''
        self.actors = []

        actor_list = self.world.get_actors()

        for actor in actor_list:
            actor_properties = {}
            actor_location = actor.get_location()

            if self.vehicle_location.distance(actor_location) < self.config['bbox_collection_radius'] \
                and all(x not in actor.type_id for x in ['spectator', 'sensor', 'controller']) \
                    and any(x in actor.semantic_tags for x in [12, 13, 14, 15, 16, 18, 19]):
                actor_properties['id'] = actor.id
                actor_properties['type'] = str(actor.type_id)
                actor_properties['is_alive'] = actor.is_alive
                actor_properties['is_active'] = actor.is_active
                actor_properties['is_dormant'] = actor.is_dormant
                actor_properties['parent'] = actor.parent.id if actor.parent is not None else None
                actor_properties['attributes'] = actor.attributes
                actor_properties['semantic_tags'] = actor.semantic_tags
                actor_properties['bounding_box'] = carla_vector_to_numpy(
                    actor.bounding_box.get_world_vertices(actor.get_transform())
                )
                actor_properties['linear_velocity'] = carla_single_vector_to_numpy(actor.get_velocity())
                actor_properties['angular_velocity'] = carla_single_vector_to_numpy(actor.get_angular_velocity())

                actor_properties['bounding_box'][:, 1] *= -1.0
                actor_properties['linear_velocity'][1] *= -1.0
                actor_properties['angular_velocity'][1:] *= -1.0

                self.actors.append(actor_properties)

            # Get traffic lights.
            if self.config['collect_traffic_light_bbox']:
                if isinstance(actor, carla.TrafficLight):
                    if self.vehicle_location.distance(actor_location) < self.config['bbox_collection_radius']:
                        bounding_boxes = actor.get_light_boxes()

                        for bounding_box in bounding_boxes:
                            if bounding_box.extent.z > 0.3:
                                actor_properties = {}

                                actor_properties['id'] = actor.id
                                actor_properties['type'] = str(actor.type_id)
                                actor_properties['is_alive'] = actor.is_alive
                                actor_properties['is_active'] = actor.is_active
                                actor_properties['is_dormant'] = actor.is_dormant
                                actor_properties['parent'] = actor.parent.id if actor.parent is not None else None
                                actor_properties['attributes'] = actor.attributes
                                actor_properties['semantic_tags'] = [7]

                                if self.map_name in ['Town12', 'Town13']:
                                    tile_bounding_box = carla_vector_to_numpy(bounding_box.get_local_vertices())

                                    x_difference = np.round((actor_location.x - bounding_box.location.x) / 1000.0) \
                                        * 1000.0
                                    y_difference = np.round((actor_location.y - bounding_box.location.y) / 1000.0) \
                                        * 1000.0

                                    actor_properties['bounding_box'] = tile_bounding_box + np.array([
                                        x_difference, y_difference, 0.0
                                    ])
                                else:
                                    actor_properties['bounding_box'] = carla_vector_to_numpy(
                                        bounding_box.get_local_vertices()
                                    )

                                actor_properties['linear_velocity'] = np.zeros(3)
                                actor_properties['angular_velocity'] = np.zeros(3)

                                actor_properties['bounding_box'][:, 1] *= -1.0

                                actor_properties['green_time'] = actor.get_green_time()
                                actor_properties['yellow_time'] = actor.get_yellow_time()
                                actor_properties['red_time'] = actor.get_red_time()
                                
                                actor_properties['state'] = str(actor.get_state())

                                actor_properties['opendrive_id'] = actor.get_opendrive_id()
                                actor_properties['pole_index'] = actor.get_pole_index()

                                self.actors.append(actor_properties)
        
        # Get the list of objects (props) in the scene, i.e. parked cars,
        # trucks, etc. that are within a certain radius of the ego vehicle.
        if self.config['collect_traffic_sign_bbox']:
            traffic_sign_list = self.world.get_environment_objects(carla.CityObjectLabel.TrafficSigns)
        
        car_list = self.world.get_environment_objects(carla.CityObjectLabel.Car)
        truck_list = self.world.get_environment_objects(carla.CityObjectLabel.Truck)
        bus_list = self.world.get_environment_objects(carla.CityObjectLabel.Bus)
        motorcycle_list = self.world.get_environment_objects(carla.CityObjectLabel.Motorcycle)
        bicycle_list = self.world.get_environment_objects(carla.CityObjectLabel.Bicycle)

        if self.config['collect_traffic_sign_bbox']:
            object_list = traffic_sign_list + car_list + truck_list + bus_list + motorcycle_list + bicycle_list
        else:
            object_list = car_list + truck_list + bus_list + motorcycle_list + bicycle_list

        for obj in object_list:
            object_properties = {}
            object_location = obj.transform.location if self.map_name not in ['Town12', 'Town13'] \
                else obj.bounding_box.location

            if self.vehicle_location.distance(object_location) < self.config['bbox_collection_radius']:
                object_properties['id'] = obj.id
                object_properties['type'] = str(obj.type)
                object_properties['is_alive'] = False
                object_properties['is_active'] = False
                object_properties['is_dormant'] = False
                object_properties['parent'] = None
                object_properties['attributes'] = {}
                object_properties['bounding_box'] = carla_vector_to_numpy(obj.bounding_box.get_local_vertices())
                object_properties['linear_velocity'] = np.zeros(3)
                object_properties['angular_velocity'] = np.zeros(3)

                object_properties['bounding_box'][:, 1] *= -1.0

                if obj.type == carla.CityObjectLabel.TrafficSigns:
                    object_properties['semantic_tags'] = [8]
                    object_properties['sign_type'] = ''

                    for sign in TRAFFIC_SIGN.keys():
                        if sign in obj.name:
                            object_properties['sign_type'] = TRAFFIC_SIGN[sign]

                    if self.map_name not in ['Town12', 'Town13', 'Town15'] and \
                        'speed_limit' in object_properties['sign_type']:
                        object_properties['sign_type'] = 'speed_limit'

                elif obj.type == carla.CityObjectLabel.Car:
                    object_properties['semantic_tags'] = [14]
                elif obj.type == carla.CityObjectLabel.Truck:
                    object_properties['semantic_tags'] = [15]
                elif obj.type == carla.CityObjectLabel.Bus:
                    object_properties['semantic_tags'] = [16]
                elif obj.type == carla.CityObjectLabel.Motorcycle:
                    object_properties['semantic_tags'] = [18]
                elif obj.type == carla.CityObjectLabel.Bicycle:
                    object_properties['semantic_tags'] = [19]
                else:
                    object_properties['semantic_tags'] = []

                self.actors.append(object_properties)
    
    def get_hd_map_info(self):
        '''
        Get HD map information from the waypoint at the vehicle's current location.
        '''
        self.hd_map_info = {}

        # Get the waypoint at the vehicle's current location.
        wp = self.map.get_waypoint(self.vehicle.get_location())

        self.hd_map_info['id'] = wp.id
        self.hd_map_info['s'] = wp.s
        self.hd_map_info['road_id'] = wp.road_id
        self.hd_map_info['section_id'] = wp.section_id
        self.hd_map_info['lane_id'] = wp.lane_id
        self.hd_map_info['lane_type'] = str(wp.lane_type)
        self.hd_map_info['lane_width'] = wp.lane_width
        self.hd_map_info['lane_change'] = str(wp.lane_change)

        self.hd_map_info['is_junction'] = wp.is_junction
        self.hd_map_info['junction_id'] = wp.junction_id if wp.is_junction else None
        self.hd_map_info['is_intersection'] = wp.is_intersection

        self.hd_map_info['left_lane_marking'] = {
            'type': str(wp.left_lane_marking.type),
            'width': wp.left_lane_marking.width,
            'color': str(wp.left_lane_marking.color),
            'lane_change': str(wp.left_lane_marking.lane_change)
        }

        self.hd_map_info['right_lane_marking'] = {
            'type': str(wp.right_lane_marking.type),
            'width': wp.right_lane_marking.width,
            'color': str(wp.right_lane_marking.color),
            'lane_change': str(wp.right_lane_marking.lane_change)
        }

        self.hd_map_info['transform'] = {
            'x': wp.transform.location.x,
            'y': -wp.transform.location.y,
            'z': wp.transform.location.z,
            'roll': wp.transform.rotation.roll,
            'pitch': -wp.transform.rotation.pitch,
            'yaw': -wp.transform.rotation.yaw
        }

        left_lane_wp = wp.get_left_lane()
        right_lane_wp = wp.get_right_lane()

        self.hd_map_info['left_lane'] = {}
        self.hd_map_info['right_lane'] = {}

        if left_lane_wp is not None:
            self.hd_map_info['left_lane']['id'] = left_lane_wp.id
            self.hd_map_info['left_lane']['s'] = left_lane_wp.s
            self.hd_map_info['left_lane']['road_id'] = left_lane_wp.road_id
            self.hd_map_info['left_lane']['section_id'] = left_lane_wp.section_id
            self.hd_map_info['left_lane']['lane_id'] = left_lane_wp.lane_id
            self.hd_map_info['left_lane']['lane_type'] = str(left_lane_wp.lane_type)
            self.hd_map_info['left_lane']['lane_width'] = left_lane_wp.lane_width
            self.hd_map_info['left_lane']['lane_change'] = str(left_lane_wp.lane_change)

        if right_lane_wp is not None:
            self.hd_map_info['right_lane']['id'] = right_lane_wp.id
            self.hd_map_info['right_lane']['s'] = right_lane_wp.s
            self.hd_map_info['right_lane']['road_id'] = right_lane_wp.road_id
            self.hd_map_info['right_lane']['section_id'] = right_lane_wp.section_id
            self.hd_map_info['right_lane']['lane_id'] = right_lane_wp.lane_id
            self.hd_map_info['right_lane']['lane_type'] = str(right_lane_wp.lane_type)
            self.hd_map_info['right_lane']['lane_width'] = right_lane_wp.lane_width
            self.hd_map_info['right_lane']['lane_change'] = str(right_lane_wp.lane_change)

    def package_data(self):
        '''
        Package scene information and data into a dictionary and return it.

        Returns:
            data: dictionary containing scene information and data.
        '''
        self.scene_data = self.sensor_manager.data

        return {'scene_info': self.scene_info, 'scene_data': self.scene_data}
    
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
    
    def stop_scene(self):
        '''
        Destroy the vehicles, walkers, and walker controllers.
        '''
        logger.debug('Stopping controllers...')

        for controller in self.controllers_list:
            controller.stop()

        logger.debug('Controllers stopped.')
        logger.debug('Destroying NPC vehicles...')

        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.npc_vehicles_list])

        logger.debug('NPC vehicles destroyed.')
        logger.debug('Destroying walkers...')

        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.walkers_list])

        logger.debug('Walkers destroyed.')
        logger.debug('Destroying controllers...')

        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.controllers_list])

        logger.debug('Controllers destroyed.')

        # Release unused GPU memory.
        with torch.cuda.device(f'cuda:{self.config["cuda_gpu"]}'):
            torch.cuda.empty_cache()
