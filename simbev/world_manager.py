# Academic Software License: Copyright © 2025 Goodarz Mehr.

import cv2
import time
import json
import carla
import torch
import random
import logging

import numpy as np

from utils import is_used, kill_all_servers

from vehicle_manager import VehicleManager
from scenario_manager import ScenarioManager


logger = logging.getLogger(__name__)


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

DOOR_STATUS = [
    carla.VehicleDoor.FL,
    carla.VehicleDoor.FR,
    carla.VehicleDoor.RL,
    carla.VehicleDoor.RR,
    carla.VehicleDoor.All
]

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


class WorldManager:
    def __init__(self, config, client, server_port):
        self.config = config
        self.client = client
        self.server_port = server_port
    
    def load_map(self, map_name):
        '''
        Load the desired map and apply the desired settings.

        Args:
            map_name: name of the map to load.
        '''
        logger.info(f'Loading {map_name}...')

        self.map_name = map_name
        
        self.client.load_world(map_name)
        
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.spectator = self.world.get_spectator()

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

        # Set up the Traffic Manager.
        logger.debug('Setting up the Traffic Manager...')

        self.tm_port = self.server_port // 10 + self.server_port % 10

        while is_used(self.tm_port):
            logger.warning(f'Traffic Manager port {self.tm_port} is already being used. Checking the next one...')
            self.tm_port += 1
        
        self.traffic_manager = self.client.get_trafficmanager(self.tm_port)
        self.traffic_manager.set_synchronous_mode(True)

        logger.debug(f'Traffic Manager is connected to port {self.tm_port}.')

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
        logger.debug('Generating waypoints...')

        self.waypoints = self.map.generate_waypoints(self.config['waypoint_distance'])

        self.crosswalks = self.map.get_crosswalks()

        self.world.tick()

        logger.debug('Waypoints generated.')
        logger.debug('Getting the Light Manager...')

        self.light_manager = self.world.get_lightmanager()

        logger.debug('Got the Light Manager.')
        logger.debug('Creating the Scenario Manager...')

        self.scenario_manager = ScenarioManager(
            self.config,
            self.client,
            self.world,
            self.traffic_manager,
            self.light_manager,
            map_name
        )

        logger.debug('Scenario Manager created.')
        logger.debug('Creating the Vehicle Manager...')

        self.vehicle_manager = VehicleManager(self.config, self.world, self.traffic_manager, map_name)

        logger.debug('Vehicle Manager created.')
    
    def spawn_vehicle(self):
        # Get the ego vehicle blueprint and spawn points.
        logger.debug('Getting vehicle blueprint and spawn points...')

        bp = self.world.get_blueprint_library().filter(self.config['vehicle'])[0]

        bp.set_attribute('role_name', 'hero')
        bp.set_attribute('color', self.config['vehicle_color'])

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
        logger.debug('Got vehicle blueprint and spawn points.')

        self.scenario_manager.scene_info = self.vehicle_manager.spawn_vehicle(bp, self.spawn_points, self.tm_port)
    
    def move_vehicle(self):
        self.scenario_manager.scene_info = self.vehicle_manager.move_vehicle(self.spawn_points, self.tm_port)

    def set_scene_duration(self, duration):
        self.scenario_manager.scene_duration = duration
    
    def start_scene(self):
        try:
            self.counter = 0
            self.termination_counter = 0
            
            self.warning_flag = False
            self.terminate_scene = False
            self.light_change = False

            self.scenario_manager.scene_info['map'] = self.map_name
            self.scenario_manager.scene_info['vehicle'] = self.vehicle_manager.vehicle.type_id
            self.scenario_manager.scene_info['expected_scene_duration'] = self.scenario_manager.scene_duration
            self.scenario_manager.scene_info['terminated_early'] = False

            self.npc_door_open_list = []
            self.tried_to_open_door_list = []
            
            self.set_spectator_view()

            self.world.tick()

            self.vehicle_manager.get_ground_truth_manager().augment_waypoints(self.waypoints)
            self.vehicle_manager.get_ground_truth_manager().trim_crosswalks(self.crosswalks)

            self.scenario_manager.setup_scenario(
                self.vehicle_manager.vehicle.get_location(),
                self.spawn_points_backup,
                self.tm_port
            )

            self.set_spectator_view()

            self.world.tick()
        
        except Exception as e:
            logger.error(f'Error while starting the scene: {e}')

            kill_all_servers()

            time.sleep(3.0)

            raise Exception('Cannot start the scene. Good bye!')

    def set_spectator_view(self):
        '''
        Set the spectator view to follow the ego vehicle.
        '''
        # Get the ego vehicle's coordinates.
        transform = self.vehicle_manager.vehicle.get_transform()

        # Calculate the spectator's desired position.
        view_x = transform.location.x - 8.0 * transform.get_forward_vector().x
        view_y = transform.location.y - 8.0 * transform.get_forward_vector().y
        view_z = transform.location.z + 4.0

        # Calculate the spectator's desired orientation.
        view_roll = transform.rotation.roll
        view_pitch = transform.rotation.pitch - 16.0
        view_yaw = transform.rotation.yaw

        # Get the spectator and place it in the calculated position.
        self.spectator.set_transform(
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
        self.vehicle_manager.get_sensor_manager().clear_queues()

        # Randomly open the door of some vehicles that are stopped, then close
        # them when the vehicles start moving.
        p = self.config['door_open_percentage'] / 100.0

        for vehicle in self.scenario_manager.npc_vehicles_list:
            if vehicle.attributes['has_dynamic_doors'] == 'true':
                role_name = vehicle.attributes['role_name']

                if role_name not in self.npc_door_open_list and role_name not in self.tried_to_open_door_list \
                    and vehicle.get_velocity().length() < 0.1:
                    
                    if np.random.choice(2, p=[1 - p, p]):
                        vehicle.open_door(random.choice(DOOR_STATUS))
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
                        weather.__getattribute__(attribute) + self.scenario_manager.weather_increment.__getattribute__(attribute)
                    )
            
            new_sun_altitude_angle = weather.sun_altitude_angle

            if self.light_change:
                self.light_manager.set_day_night_cycle(True)
                
                self.light_change = False
            
            if old_sun_altitude_angle > 0.0 and new_sun_altitude_angle <= 0.0:
                self.scenario_manager.configure_lights()
                
                self.light_manager.set_day_night_cycle(False)
                
                self.light_change = True
            
            self.world.set_weather(weather)
        
        # Proceed for one time step.
        self.world.tick()

        self.set_spectator_view()
        
        if render or save:
            self.vehicle_location = self.vehicle_manager.vehicle.get_location()

            ground_truth_manager = self.vehicle_manager.get_ground_truth_manager()
            
            if self.counter % round(0.5 / self.config['timestep']) == 0:
                ground_truth_manager.trim_waypoints()

            # Torch can end up hogging GPU memory when calculating the ground
            # truth, so empty it every once in a while.
            if self.counter % 40 == 5:
                with torch.cuda.device(f'cuda:{self.config["cuda_gpu"]}'):
                    torch.cuda.empty_cache()
            
            self.counter += 1
            
            # If nearby roads do not have an elevation difference of more than
            # 6.4 meters, get the ground truth using the top and bottom
            # semantic cameras and road waypoints. Otherwise (i.e. when near
            # an overpass/underpass), get the ground truth using bounding
            # boxes and road waypoints.
            if self.map_name not in ['Town04', 'Town05', 'Town12', 'Town13']:
                bev_gt = ground_truth_manager.get_bev_gt()
                self.warning_flag = False
            elif (torch.max(self.nwp_loc[:, 2]) - torch.min(self.nwp_loc[:, 2])).numpy() < 6.4:
                bev_gt = ground_truth_manager.get_bev_gt()
                self.warning_flag = False
            else:
                mid = (torch.max(ground_truth_manager.nwp_loc[:, 2]) + torch.min(ground_truth_manager.nwp_loc[:, 2])) / 2.0

                highs = ground_truth_manager.nwp_loc[ground_truth_manager.nwp_loc[:, 2] > (mid + 3.2)]
                lows = ground_truth_manager.nwp_loc[ground_truth_manager.nwp_loc[:, 2] < (mid - 3.2)]

                dists = torch.cdist(highs[:, :2], lows[:, :2])

                if torch.min(dists) > 48.0:
                    bev_gt = ground_truth_manager.get_bev_gt()
                    self.warning_flag = False
                else:
                    ground_truth_manager.get_bounding_boxes()
                    bev_gt = ground_truth_manager.get_bev_gt_alt()

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
                canvas[bev_gt[k], :] = PALETTE[name]

            canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        
        # Render sensor data.
        if render:
            self.vehicle_manager.get_sensor_manager().render()

            cv2.imshow('Ground Truth', canvas)
            cv2.waitKey(1)
        
        # Save sensor data to file.
        if save and all(v is not None for v in [path, scene, frame]):
            self.vehicle_manager.get_sensor_manager().save(path, scene, frame)

            with open(
                f'{path}/simbev/ground-truth/seg/SimBEV-scene-{scene:04d}-frame-{frame:04d}-GT_SEG.npz',
                'wb'
            ) as f:
                np.savez_compressed(f, data=bev_gt)

            cv2.imwrite(f'{path}/simbev/ground-truth/seg_viz/SimBEV-scene-{scene:04d}-frame-{frame:04d}-GT_SEG_VIZ.jpg',
                        canvas)
            
            ground_truth_manager.get_bounding_boxes()
            
            with open(
                f'{path}/simbev/ground-truth/det/SimBEV-scene-{scene:04d}-frame-{frame:04d}-GT_DET.bin',
                'wb'
            ) as f:
                np.save(f, np.array(ground_truth_manager.actors), allow_pickle=True)

            hd_map_info = ground_truth_manager.get_hd_map_info()

            with open(
                f'{path}/simbev/ground-truth/hd_map/SimBEV-scene-{scene:04d}-frame-{frame:04d}-HD_MAP.json',
                'w'
            ) as f:
                json.dump(hd_map_info, f, indent=4)

        # Decide whether to terminate the scene.
        if scene is not None and self.config['early_scene_termination']:
            if self.vehicle_manager.vehicle.get_velocity().length() < 0.1:
                self.termination_counter += 1
            else:
                self.termination_counter = 0
            
            if self.termination_counter * self.config['timestep'] >= self.config['termination_timeout']:
                self.terminate_scene = True
    
    def get_terminate_scene(self):
        return self.terminate_scene
    
    def get_map_name(self):
        return self.map_name
    
    def set_scene_info(self, info):
        return self.scenario_manager.set_scene_info(info)
    
    def stop_scene(self):
        self.scenario_manager.stop_scene()
    
    def destroy_vehicle(self):
        self.vehicle_manager.destroy_vehicle()