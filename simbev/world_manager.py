# Academic Software License: Copyright © 2025 Goodarz Mehr.

'''
Module that manages the CARLA world, performing functions such as loading the
map, controlling the Scenario and Vehicle Managers, and stepping through the
simulation.
'''

import time
import carla
import logging

from utils import is_used, kill_all_servers

from vehicle_manager import VehicleManager
from scenario_manager import ScenarioManager


logger = logging.getLogger(__name__)


class WorldManager:
    '''
    The World Manager manages the CARLA world, performing functions such as
    loading the map, controlling the Scenario and Vehicle Managers, and
    stepping through the simulation.

    Args:
        config: dictionary of configuration parameters.
        client: CARLA client.
        server_port: port number of the CARLA server.
    '''
    def __init__(self, config: dict, client: carla.Client, server_port: int):
        self._config = config
        self._client = client
        self._server_port = server_port
    
    def get_terminate_scene(self) -> bool:
        '''Get scene termination status.'''
        return self._terminate_scene
    
    def get_map_name(self) -> str:
        '''Get the name of the current map.'''
        return self._map_name
    
    def set_scene_duration(self, duration: int):
        '''
        Set scene duration.

        Args:
            duration: scene duration in seconds.
        '''
        self._scenario_manager.scene_duration = duration
    
    def set_scene_info(self, info: dict):
        '''
        Set scene information.

        Args:
            info: dictionary of scene information.
        '''
        return self._scenario_manager.set_scene_info(info)
    
    def load_map(self, map_name: str):
        '''
        Load the desired map and apply the appropriate settings.

        Args:
            map_name: name of the map to load.
        '''
        logger.info(f'Loading {map_name}...')

        self._map_name = map_name
        
        if map_name == 'Town10HD':
            self._client.load_world('Town10HD_Opt')
        else:
            self._client.load_world(map_name)

        self._world = self._client.get_world()
        self._map = self._world.get_map()
        self._spectator = self._world.get_spectator()

        settings = self._world.get_settings()

        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self._config['timestep']

        # If the selected map is Town12 or Town13 (large maps), limit tile
        # stream distance and actor active distance. If Town13 or Town15 are
        # selected, set max culling distance to 100.0, then revert back to 0.
        # This ensures faraway objects are rendered correctly.
        if map_name in ['Town12', 'Town13']:
            settings.tile_stream_distance = self._config['tile_stream_distance']
            settings.actor_active_distance = self._config['actor_active_distance']

        if map_name in ['Town13', 'Town15']:
            settings.max_culling_distance = 100.0

            self._world.apply_settings(settings)

            time.sleep(3.0)
        
        settings.max_culling_distance = 0.0

        self._world.apply_settings(settings)

        # Set up the Traffic Manager.
        logger.debug('Setting up the Traffic Manager...')

        self._tm_port = self._server_port // 10 + self._server_port % 10

        while is_used(self._tm_port):
            logger.warning(f'Traffic Manager port {self._tm_port} is already being used. Checking the next one...')
            self._tm_port += 1
        
        self._traffic_manager = self._client.get_trafficmanager(self._tm_port)
        self._traffic_manager.set_synchronous_mode(True)

        logger.debug(f'Traffic Manager is connected to port {self._tm_port}.')

        self._world.tick()

        logger.info(f'{map_name} loaded.')
    
        # Some objects obstruct the overhead or bottom-up view that is
        # necessary for the collection of accurate ground truth data, so they
        # are removed from the map.
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
            # The first ones are for Town10HD, the second ones are for
            # Town10HD_Opt.
            # obstructing = [
            #     'SM_Tesla2',
            #     'SM_Tesla_2502',
            #     'SM_Mustang_prop2',
            #     'SM_Patrol2021Parked2',
            #     'SM_mercedescccParked2',
            #     'SM_LincolnMkz2017_prop',
            #     'Vh_Car_ToyotaPrius_NOrig',
            #     'InstancedFoliageActor_0_Inst_235_0',
            #     'InstancedFoliageActor_0_Inst_239_4',
            #     'InstancedFoliageActor_0_Inst_245_10',
            #     'InstancedFoliageActor_0_Inst_246_11',
            #     'InstancedFoliageActor_0_Inst_249_14',
            #     'InstancedFoliageActor_0_Inst_250_15',
            #     'InstancedFoliageActor_0_Inst_251_16',
            #     'InstancedFoliageActor_0_Inst_252_17',
            #     'InstancedFoliageActor_0_Inst_253_18',
            #     'InstancedFoliageActor_0_Inst_254_19',
            #     'InstancedFoliageActor_0_Inst_255_20',
            #     'InstancedFoliageActor_0_Inst_256_21',
            #     'InstancedFoliageActor_0_Inst_257_22',
            #     'InstancedFoliageActor_0_Inst_258_23',
            #     'InstancedFoliageActor_0_Inst_259_24',
            #     'InstancedFoliageActor_0_Inst_260_25',
            #     'InstancedFoliageActor_0_Inst_261_26',
            #     'InstancedFoliageActor_0_Inst_276_41',
            #     'InstancedFoliageActor_0_Inst_277_42'
            # ]
            obstructing = [
                'SM_Tesla2',
                'SM_Tesla_2502',
                'SM_Mustang_prop2',
                'SM_Patrol2021Parked2',
                'SM_mercedescccParked2',
                'SM_LincolnMkz2017_prop',
                'Vh_Car_ToyotaPrius_NOrig',
                'InstancedFoliageActor_0_Inst_12961_0',
                'InstancedFoliageActor_0_Inst_12965_4',
                'InstancedFoliageActor_0_Inst_12971_10',
                'InstancedFoliageActor_0_Inst_12972_11',
                'InstancedFoliageActor_0_Inst_12980_19',
                'InstancedFoliageActor_0_Inst_12981_20',
                'InstancedFoliageActor_0_Inst_12982_21',
                'InstancedFoliageActor_0_Inst_12983_22',
                'InstancedFoliageActor_0_Inst_13002_41',
                'InstancedFoliageActor_0_Inst_13003_42'
            ]
        else:
            obstructing = []

        self._bad_crosswalks = [
            'Road_Crosswalk_Town03_59_',
            'Road_Crosswalk_Town04_28_',
            'Road_Crosswalk_Town04_29_',
            'Road_Crosswalk_Town04_30_',
            'Road_Crosswalk_Town07_5_',
            'Road_Crosswalk_Town07_8_',
            'Road_Crosswalk_Town07_9_'
        ]
        
        self._objects = self._world.get_environment_objects()

        to_remove = [obj.id for obj in self._objects if any(x in obj.name for x in obstructing)]

        self._world.enable_environment_objects(set(to_remove), False)

        self._world.tick()

        logger.debug(f'Objects obstructing the overhead or bottom-up view were removed from {map_name}.')
        logger.debug('Generating waypoints...')

        self._waypoints = self._map.generate_waypoints(self._config['waypoint_distance'])

        self._crosswalks = self._map.get_crosswalks()

        self._world.tick()

        logger.debug('Waypoints generated.')
        logger.debug('Getting the Light Manager...')

        self._light_manager = self._world.get_lightmanager()

        logger.debug('Got the Light Manager.')
        logger.debug('Creating the Scenario Manager...')

        self._scenario_manager = ScenarioManager(
            self._config,
            self._client,
            self._world,
            self._traffic_manager,
            self._light_manager,
            map_name
        )

        logger.debug('Scenario Manager created.')
        logger.debug('Creating the Vehicle Manager...')

        self._vehicle_manager = VehicleManager(self._config, self._world, self._traffic_manager, map_name)

        logger.debug('Vehicle Manager created.')
    
    def spawn_vehicle(self):
        '''Prepare to spawn the ego vehicle at a spawn point.'''
        # Get the ego vehicle blueprint and spawn points.
        logger.debug('Getting vehicle blueprint and spawn points...')

        bp = self._world.get_blueprint_library().filter(self._config['vehicle'])[0]

        bp.set_attribute('role_name', 'hero')
        bp.set_attribute('color', self._config['vehicle_color'])

        wp_all = self._map.generate_waypoints(self._config['spawn_point_separation_distance'])

        # Filter out spawn points that are in a junction or within 4 meters of
        # a junction.
        wp_mid = [wp for wp in wp_all if wp.next(4.0) != []]

        self._spawn_points = [
            wp.transform for wp in wp_mid if (wp.is_junction is False and wp.next(4.0)[0].is_junction is False)
        ]

        for sp in self._spawn_points:
            sp.location.z += 0.6

        self._spawn_points_copy = self._spawn_points
        
        if 'ego_vehicle_spawn_point' in self._config:
            if self._map_name in self._config['ego_vehicle_spawn_point']:
                spawn_point_list = self._config['ego_vehicle_spawn_point'][self._map_name]

                self._spawn_points = []

                for sp in spawn_point_list:
                    location = carla.Location(x=sp[0][0], y=sp[0][1], z=sp[0][2])
                    rotation = carla.Rotation(roll=sp[1][0], pitch=sp[1][1], yaw=sp[1][2])

                    self._spawn_points.append(carla.Transform(location, rotation))
        
        logger.debug(f'{len(self._spawn_points)} spawn points available.')
        logger.debug('Got vehicle blueprint and spawn points.')

        self._scenario_manager.scene_info = self._vehicle_manager.spawn_vehicle(bp, self._spawn_points, self._tm_port)
    
    def move_vehicle(self):
        '''Move the ego vehicle to a new spawn point.'''
        self._scenario_manager.scene_info = self._vehicle_manager.move_vehicle(self._spawn_points, self._tm_port)
    
    def start_scene(self):
        '''Start the scenario.'''
        try:
            self._counter = 0
            self._termination_counter = 0
            
            self._terminate_scene = False

            # Add information about the scenario to the scene info.
            self._scenario_manager.scene_info['map'] = self._map_name
            self._scenario_manager.scene_info['vehicle'] = self._vehicle_manager.vehicle.type_id
            self._scenario_manager.scene_info['expected_scene_duration'] = self._scenario_manager.scene_duration
            self._scenario_manager.scene_info['terminated_early'] = False
            
            self._set_spectator_view()

            self._world.tick()

            # Preprocess the waypoints and crosswalks for ground truth
            # generation.
            self._vehicle_manager.get_ground_truth_manager().augment_waypoints(self._waypoints)
            self._vehicle_manager.get_ground_truth_manager().get_area_crosswalks(self._crosswalks)
            self._vehicle_manager.get_ground_truth_manager().get_bounding_boxes()

            self._scenario_manager.setup_scenario(
                self._vehicle_manager.vehicle.get_location(),
                self._spawn_points_copy,
                self._tm_port
            )

            self._set_spectator_view()

            self._world.tick()
        
        except Exception as e:
            logger.error(f'Error while starting the scene: {e}')

            kill_all_servers()

            time.sleep(3.0)

            raise Exception('Cannot start the scene. Good bye!')

    def _set_spectator_view(self):
        '''Set the spectator view to follow the ego vehicle.'''
        # Get the ego vehicle's coordinates.
        transform = self._vehicle_manager.vehicle.get_transform()

        # Calculate the spectator's desired position.
        view_x = transform.location.x - 8.0 * transform.get_forward_vector().x
        view_y = transform.location.y - 8.0 * transform.get_forward_vector().y
        view_z = transform.location.z + 4.0

        # Calculate the spectator's desired orientation.
        view_roll = transform.rotation.roll
        view_pitch = transform.rotation.pitch - 16.0
        view_yaw = transform.rotation.yaw

        # Get the spectator and place it in the calculated position.
        self._spectator.set_transform(
            carla.Transform(
                carla.Location(x=view_x, y=view_y, z=view_z),
                carla.Rotation(roll=view_roll, pitch=view_pitch, yaw=view_yaw)
            )
        )

    def tick(self, path: str = None, scene: int = None, frame: int = None, render: bool = False, save: bool = False):
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
        start = time.perf_counter()

        if save:
            self._vehicle_manager.get_sensor_manager().wait_for_saves()

        self._vehicle_manager.get_sensor_manager().clear_queues()

        # Randomly open the door of some vehicles that are stopped, then close
        # them when the vehicles start moving.
        self._scenario_manager.manage_doors()

        # Change the weather if configured to do so.
        if self._config['weather_shift'] and scene is not None:
            self._scenario_manager.shift_weather()
        
        # Proceed for one time step.
        self._world.tick()

        self._set_spectator_view()
        
        if render or save:
            self._vehicle_location = self._vehicle_manager.vehicle.get_location()

            ground_truth_manager = self._vehicle_manager.get_ground_truth_manager()
            
            if self._counter % round(0.5 / self._config['timestep']) == 0:
                ground_truth_manager.trim_map_sections()
            
            self._counter += 1
            
            ground_truth_manager.get_ground_truth()

        # Render the data and ground truth.
        if render:
            self._vehicle_manager.get_sensor_manager().render()
            self._vehicle_manager.get_ground_truth_manager().render()
        
        # Save the data and ground truth to file.
        if save and all(v is not None for v in [path, scene, frame]):
            self._vehicle_manager.get_sensor_manager().save(path, scene, frame)
            self._vehicle_manager.get_ground_truth_manager().save(path, scene, frame)

        # Decide whether to terminate the scene.
        if scene is not None and self._config['early_scene_termination']:
            if self._vehicle_manager.vehicle.get_velocity().length() < 0.1:
                self._termination_counter += 1
            else:
                self._termination_counter = 0
            
            if self._termination_counter * self._config['timestep'] >= self._config['termination_timeout']:
                self._terminate_scene = True
        
        end = time.perf_counter()

        logger.debug(f'Time taken for world tick: {end - start:.6f} seconds.')
    
    def stop_scene(self):
        '''Stop the scene.'''
        self._vehicle_manager.get_sensor_manager().wait_for_saves()
        return self._scenario_manager.stop_scene()
    
    def destroy_vehicle(self):
        '''Destroy the vehicle.'''
        return self._vehicle_manager.destroy_vehicle()
    
    def package_data(self) -> dict:
        '''
        Package scene information and data into a dictionary and return it.

        Returns:
            data: dictionary containing scene information and data.
        '''
        return {
            'scene_info': self._scenario_manager.scene_info,
            'scene_data': self._vehicle_manager.get_sensor_manager().get_data()
        }