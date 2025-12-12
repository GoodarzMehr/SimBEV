# Academic Software License: Copyright © 2025 Goodarz Mehr.

'''
Module that sets up and manages the scenario, configuring the weather, lights,
and traffic elements.
'''

import time
import carla
import random
import logging

import numpy as np


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

MAX_VEHICLE_LENGTH = {
    'car': 6.0,
    'truck': 10.0,
    'van': 10.0,
    'bus': 12.0,
    'motorcycle': 4.0,
    'bicycle': 2.0
}

TRAFFIC_CONES = [
    'constructioncone',
    'orangeconesmall'
]

CONSTRUCTION_CONES = [
    'trafficcone01',
    'trafficcone02',
    'concretebarrier',
    'orangeconebig',
    'roadsideconstructioncone',
    'skinnycone'
]

BARRIERS = [
    'streetbarrier',
    'concretebarrier',
    'woodenbarrier'
]

WORK_PROPS = [
    'barrel',
    'ironplank',
    'closedsandbag',
    'concretebarrier',
    'concretepiece1',
    'concretepipe',
    'concreteslab1',
    'constructionlight',
    'cylinder',
    'dirtpile',
    'electricalbox',
    'floorboard',
    'floorgrill',
    'gutter',
    'handtruck',
    'opensandbag',
    'pallet',
    'shovel',
    'stonering',
    'toolbox',
    'wheelbarrow',
    'woodenwheel',
    'bucket',
    'concretepiece2',
    'concreteslab2',
    'walker',
    'none'
]

SMALL_PROPS = [
    'barrel',
    'closedsandbag',
    'concretepiece1',
    'concreteslab1',
    'cylinder',
    'electricalbox',
    'floorgrill',
    'gutter',
    'handtruck',
    'opensandbag',
    'pallet',
    'shovel',
    'stonering',
    'toolbox',
    'wheelbarrow',
    'woodenwheel',
    'bucket',
    'concretepiece2',
    'concreteslab2'
]


class ScenarioManager:
    '''
    The Scenario Manager sets up and manages the scenario, configuring the
    weather, lights, and traffic elements.

    Args:
        config: dictionary of configuration parameters.
        client: CARLA client.
        world: CARLA world.
        traffic_manager: CARLA traffic manager.
        light_manager: CARLA light manager.
        map_name: name of the CARLA map.
    '''
    def __init__(
            self,
            config: dict,
            client: carla.Client,
            world: carla.World,
            traffic_manager: carla.TrafficManager,
            light_manager: carla.LightManager,
            map_name: str
        ):
        self._config = config
        self._client = client
        self._world = world
        self._traffic_manager = traffic_manager
        self._light_manager = light_manager
        self._map_name = map_name

        self.scene_info = {}

        self.scene_duration = 0.5
    
    def get_hazard_locations(self) -> dict:
        '''
        Get the locations of the hazards in the scenario.

        Returns:
            hazard_locations: list of hazard endpoints.
        '''
        return self._hazard_endpoints
    
    def set_scene_info(self, info: dict):
        '''
        Set scene information.

        Args:
            info: dictionary of scene information.
        '''
        self.scene_info.update(info)

    def setup_scenario(self, vehicle_location: carla.Location, spawn_points: list[carla.Waypoint], tm_port: int):
        '''Set up the scenario by configuring the weather, lights, and traffic.'''
        
        # Configure the weather.
        logger.debug('Configuring the weather...')

        initial_weather = self._world.get_weather()

        initial_weather = self._configure_weather(initial_weather)

        if 'initial_weather' in self._config:
            for attribute in initial_weather.__dir__():
                if attribute in self._config['initial_weather']:
                    initial_weather.__setattr__(attribute, self._config['initial_weather'][attribute])

        # If weather shift is enabled, calculate how much each weather
        # attribute should change at each time step.
        if self._config['dynamic_weather']:
            self.scene_info['dynamic_weather'] = True

            final_weather = self._world.get_weather()
        
            final_weather = self._configure_weather(final_weather)

            if 'final_weather' in self._config:
                for attribute in final_weather.__dir__():
                    if attribute in self._config['final_weather']:
                        final_weather.__setattr__(attribute, self._config['final_weather'][attribute])

            self._weather_increment = self._world.get_weather()

            num_steps = round(self.scene_duration / self._config['timestep'])

            for attribute in self._weather_increment.__dir__():
                if attribute in WEATHER_ATTRIBUTES:
                    self._weather_increment.__setattr__(
                        attribute,
                        (final_weather.__getattribute__(attribute) - initial_weather.__getattribute__(attribute)) \
                            / num_steps
                    )

        self._world.set_weather(initial_weather)

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

        if self._config['dynamic_weather']:
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

        self._world.tick()

        time.sleep(1.0)

        # Configure the lights.
        logger.debug('Configuring the lights...')

        self.scene_info['street_light_intensity_change'] = 0.0

        if initial_weather.sun_altitude_angle < 0.0:
            self._configure_lights()

        self._light_change = False
        
        logger.debug('Lights configured.')

        self._npc_spawn_radius = self._config['npc_spawn_radius']

        if self._config['dynamic_settings_adjustments']:
            if self.scene_duration <= 12.0:
                self._npc_spawn_radius = 30.0 * (self.scene_duration + self._config['warmup_duration'])
            elif self.scene_duration <= 16.0:
                self._npc_spawn_radius = 25.0 * (self.scene_duration + self._config['warmup_duration'])
            else:
                self._npc_spawn_radius = 20.0 * (self.scene_duration + self._config['warmup_duration'])
        
            logger.debug(f'Changed NPC spawn radius to {self._npc_spawn_radius:.2f} m.')

        # Create road hazards.
        logger.debug('Creating road hazards...')

        self._hazard_endpoints = []

        hazard_spawn_points = [
            sp for sp in spawn_points if (
                (self._config['spawn_point_separation_distance'] / 2.0) < \
                    vehicle_location.distance(sp.location) < \
                        self._npc_spawn_radius
            )
        ]

        num_hazards = round(len(hazard_spawn_points) * self._config['hazard_area_percentage'] / 100.0)

        num_accident_hazards = 0
        num_road_work_hazards = 0

        self._hazard_vehicle_list = []
        self._hazard_walker_list = []
        self._hazard_prop_list = []

        p = self._config['accident_hazard_percentage'] / 100.0

        for _ in range(num_hazards):
            if np.random.choice(2, p=[1 - p, p]):
                hazard_created = self._create_accident_hazard(hazard_spawn_points, tm_port)

                num_accident_hazards += int(hazard_created)
            else:
                hazard_created = self._create_road_work_hazard(hazard_spawn_points, vehicle_location)

                num_road_work_hazards += int(hazard_created)

        self.scene_info['n_accident_hazards'] = num_accident_hazards
        self.scene_info['n_road_work_hazards'] = num_road_work_hazards

        logger.info(f'Created {num_accident_hazards} accident hazards.')
        logger.info(f'Created {num_road_work_hazards} road work hazards.')
        
        # Spawn NPCs.
        logger.debug('Spawning NPCs...')

        all_npc_spawn_points = [
            sp for sp in spawn_points if vehicle_location.distance(sp.location) < self._npc_spawn_radius
        ]

        npc_spawn_points = []

        for point in all_npc_spawn_points:
            wp = self._world.get_map().get_waypoint(point.location)

            for bwp, fwp in self._hazard_endpoints:
                if (wp.transform.location.distance(bwp.transform.location) < \
                    (self._config['spawn_point_separation_distance'] / 2.0) \
                    and wp.road_id == bwp.road_id and wp.lane_id == bwp.lane_id) or \
                     (wp.transform.location.distance(fwp.transform.location) < \
                    (self._config['spawn_point_separation_distance'] / 2.0) \
                        and wp.road_id == fwp.road_id and wp.lane_id == fwp.lane_id):
                    pass
                else:
                    npc_spawn_points.append(point)
                    
                    break

        logger.debug(f'{len(npc_spawn_points)} NPC spawn points available.')

        if 'n_vehicles' in self._config:
            n_vehicles = self._config['n_vehicles']
            if n_vehicles == 27: logger.debug('rheM zradooG 4202 © thgirypoC')
        else:
            n_vehicles = random.randint(0, max(len(npc_spawn_points) - 3, 0))
        
        if 'n_walkers' in self._config:
            n_walkers = self._config['n_walkers']
        else:
            n_walkers = random.randint(0, self._config['max_n_walkers'])
        
        self._spawn_npcs(n_vehicles, n_walkers, vehicle_location, npc_spawn_points, tm_port)

        # In the new version of CARLA pedestrians are rendered invisible to
        # the lidar by default, this makes them visible.
        actors = self._world.get_actors()

        for actor in actors:
            if 'walker.pedestrian' in actor.type_id:
                actor.set_collisions(True)
                actor.set_simulate_physics(self._config['simulate_physics'])

        self._npc_door_open_list = []
        self._tried_to_open_door_list = []
        
        logger.debug('NPCs spawned.')

    def _configure_weather(self, weather: carla.WeatherParameters) -> carla.WeatherParameters:
        '''
        Configure the weather randomly.

        Args:
            weather: CARLA weather object to configure.
        
        Returns:
            weather: configured CARLA weather object.
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
        weather.sun_altitude_angle = 180 * random.betavariate(3.2, 2.2) - 90.0

        weather.wetness = min(100.0, max(random.gauss(weather.precipitation, 10.0), 0.0))

        weather.fog_density = 100 * random.betavariate(0.6, 1.2) if weather.cloudiness > 40.0 \
            or weather.sun_altitude_angle < 10.0 else 0.0
        
        if weather.fog_density <= 10.0:
            weather.fog_density = 0.0

        weather.fog_distance = random.lognormvariate(3.2, 0.8) if weather.fog_density > 10.0 else 100.0
        weather.fog_falloff = 5.0 * random.betavariate(1.2, 2.4) if weather.fog_density > 10.0 else 1.0

        # Town12, Town13, and Town15 have non-zero elevation, so setting fog
        # falloff to a larger value would make it disappear from the map.
        if self._map_name in ['Town12', 'Town13', 'Town15']:
            if weather.fog_density > 10.0:
                weather.fog_falloff /= 20.0
        
        return weather
    
    def configure_replay_weather(self, initial_weather: dict, final_weather: dict = None):
        '''
        Configure the weather for replaying a scenario.

        Args:
            initial_weather: initial weather parameters.
            final_weather: final weather parameters.
        '''
        weather = self._world.get_weather()

        for attribute in initial_weather:
            weather.__setattr__(attribute, initial_weather[attribute])

        self._world.set_weather(weather)

        if final_weather is not None:
            self._weather_increment = self._world.get_weather()

            num_steps = round(self.scene_duration / self._config['timestep'])

            for attribute in final_weather:
                self._weather_increment.__setattr__(
                    attribute,
                    (final_weather[attribute] - initial_weather[attribute]) / num_steps
                )
    
    def _configure_lights(self):
        '''Configure the lights.'''
        street_lights = self._light_manager.get_all_lights(carla.LightGroup.Street)
        building_lights = self._light_manager.get_all_lights(carla.LightGroup.Building)

        street_light_intensity = self._light_manager.get_intensity(street_lights)

        # Set random colors for building lights and turn them on.
        if self._config['random_building_light_colors'] and self._map_name not in ['Town12', 'Town13', 'Town15']:
            for light in list(building_lights):
                color = carla.Color(r=random.randint(0, 255), g=random.randint(0, 255), b=random.randint(0, 255))

                self._light_manager.set_color([light], color)
            
        self._light_manager.turn_on(building_lights)

        self.scene_info['building_lights_on'] = True
        
        # Change street light intensity and turn the lights on.
        if self._config['change_street_light_intensity']:
            if 'street_light_intensity_change' in self._config:
                intensity_change = self._config['street_light_intensity_change']
            else:
                intensity_change = np.random.uniform(
                    -np.mean(street_light_intensity),
                    np.mean(street_light_intensity)
                )

            logger.info(f'Change in street light intensity: {intensity_change:.2f} lumens.')

            self.scene_info['street_light_intensity_change'] = intensity_change
            
            new_street_light_intensity = list(np.maximum(
                np.array(street_light_intensity) + intensity_change,
                self._config['min_street_light_intensity']
            ))
            
            self._light_manager.set_intensities(street_lights, new_street_light_intensity)
            
        self._light_manager.turn_on(street_lights)

        self.scene_info['street_lights_on'] = True

        # Randomly turn off some street lights.
        if self._config['random_street_light_failure']:
            p = self._config['street_light_failure_percentage'] / 100.0

            new_street_light_status = np.random.choice(2, len(street_lights), p=[p, 1 - p]).astype(bool).tolist()

            self._light_manager.set_active(street_lights, new_street_light_status)
        
        # Turn off all building and/or street lights if specified.
        if self._config['turn_off_building_lights']:
            self._light_manager.turn_off(building_lights)

            self.scene_info['building_lights_on'] = False
        
        if self._config['turn_off_street_lights']:
            self._light_manager.turn_off(street_lights)

            self.scene_info['street_lights_on'] = False

    def _create_accident_hazard(self, hazard_spawn_points: list[carla.Transform], tm_port: int) -> bool:
        '''
        Create an accident hazard by spawning two stopped vehicle on the road.
        Sometimes a police vehicle is also spawned behind them.

        Args:
            hazard_spawn_points: list of possible spawn points for the hazard.
            tm_port: port number of the traffic manager.
        
        Returns:
            success: whether the hazard was created successfully.
        '''
        p = self._config['emergency_vehicle_at_accident_percentage'] / 100.0

        spawn_emergency_vehicle = np.random.choice(2, p=[1 - p, p])

        # Get vehicle blueprints.
        v_blueprints_all = self._world.get_blueprint_library().filter('vehicle.*')
        v_blueprints = [v for v in v_blueprints_all if v.get_attribute('has_lights').__bool__() == True]

        v_blueprints_non_emergency = [v for v in v_blueprints if not v.get_attribute('special_type') == 'emergency']
        v_blueprints_emergency = [v for v in v_blueprints if v.get_attribute('special_type') == 'emergency']
        
        bps = []

        for _ in range(2):
            bps.append(random.choice(v_blueprints_non_emergency))
        
        if spawn_emergency_vehicle:
            bps.append(random.choice(v_blueprints_emergency))

        # Choose the spawn points for hazard vehicles.
        spawn_point = random.choice(hazard_spawn_points)

        hwp = self._world.get_map().get_waypoint(spawn_point.location)

        wps = []

        try:
            hfwp = hwp.next(MAX_VEHICLE_LENGTH[bps[0].get_attribute('base_type').as_str()])[0]
            
            if spawn_emergency_vehicle:
                hbwp = hwp.previous(
                    MAX_VEHICLE_LENGTH[bps[1].get_attribute('base_type').as_str()] + \
                        MAX_VEHICLE_LENGTH[bps[2].get_attribute('base_type').as_str()]
                )[0]
            else:
                hbwp = hwp.previous(MAX_VEHICLE_LENGTH[bps[1].get_attribute('base_type').as_str()] + 2.0)[0]
            
            for bwp, fwp in self._hazard_endpoints:
                if hfwp.transform.location.distance(bwp.transform.location) < \
                    (self._config['spawn_point_separation_distance'] / 2.0) \
                        and hfwp.road_id == bwp.road_id and hfwp.lane_id == bwp.lane_id:
                    return False
                if hbwp.transform.location.distance(fwp.transform.location) < \
                    (self._config['spawn_point_separation_distance'] / 2.0) \
                        and hbwp.road_id == fwp.road_id and hbwp.lane_id == fwp.lane_id:
                    return False
            
            wps.append(hwp.next(MAX_VEHICLE_LENGTH[bps[0].get_attribute('base_type').as_str()] / 2.0)[0])
            wps.append(hwp.previous(MAX_VEHICLE_LENGTH[bps[1].get_attribute('base_type').as_str()] / 2.0)[0])
                
            if spawn_emergency_vehicle:
                wps.append(
                    hwp.previous(
                        MAX_VEHICLE_LENGTH[bps[1].get_attribute('base_type').as_str()] + \
                            MAX_VEHICLE_LENGTH[bps[2].get_attribute('base_type').as_str()] / 2.0 + 1.0
                    )[0]
                )
            else:
                wps.append(hwp.previous(MAX_VEHICLE_LENGTH[bps[1].get_attribute('base_type').as_str()] + 2.0)[0])
        except IndexError:
            return False
        
        spawn_points = []
        
        for i, wp in enumerate(wps):
            point = wp.transform
            
            point.location.z += 0.1
            
            if i < len(bps):
                if bps[i].get_attribute('base_type').as_str() in ['motorcycle', 'bicycle']:
                    point.rotation.yaw += random.uniform(-90.0, 90.0)
                elif bps[i].get_attribute('base_type').as_str() in ['car']:
                    point.rotation.yaw += random.uniform(-30.0, 30.0)
                else:
                    point.rotation.yaw += random.uniform(-10.0, 10.0)
            
            spawn_points.append(point)
        
        # Choose the spawn points for pedestrians around the hazard vehicles.
        walker_wps = []

        try:
            walker_wps.append(hwp.next(MAX_VEHICLE_LENGTH[bps[0].get_attribute('base_type').as_str()])[0])
            walker_wps.append(hwp)
                
            if spawn_emergency_vehicle:
                walker_wps.append(hwp.previous(MAX_VEHICLE_LENGTH[bps[1].get_attribute('base_type').as_str()])[0])
        except IndexError:
            return False
        
        walker_spawn_points = []

        for wp in walker_wps:
            for _ in range(random.randint(1, 3)):
                point = wp.transform

                point.location += carla.Location(x=random.uniform(-0.6, 0.6), y=random.uniform(-0.6, 0.6), z=0.1)
                point.rotation.yaw += random.uniform(-180.0, 180.0)

                walker_spawn_points.append(point)
        
        vehicle_list = []
        
        for i, bp in enumerate(bps):
            # Spawn the hazard vehicle.
            bp.set_attribute('role_name', f'hazard_vehicle_{i}')

            if bp.has_attribute('color'):
                bp.set_attribute('color', random.choice(bp.get_attribute('color').recommended_values))

            vehicle = self._world.try_spawn_actor(bp, spawn_points[i])

            if vehicle is None:
                for v in vehicle_list:
                    v.destroy()
                
                return False
            
            # Set autopilot to true and set the desired speed to zero so
            # lane chaning works properly.
            vehicle.set_autopilot(True, tm_port)
            vehicle.set_simulate_physics(self._config['simulate_physics'])

            self._traffic_manager.set_desired_speed(vehicle, 0.0)
            
            # Configure hazard vehicle lights and doors.
            self._traffic_manager.update_vehicle_lights(vehicle, True)

            if i == 2:
                vehicle.set_light_state(carla.VehicleLightState.Special1)
            
            if vehicle.attributes['has_dynamic_doors'] == 'true':
                vehicle.open_door(random.choice(DOOR_STATUS))
            
            vehicle_list.append(vehicle)
        
        self._hazard_vehicle_list.extend(vehicle_list)

        #Spawn pedestrians around the hazard vehicles.
        walker_library = self._world.get_blueprint_library().filter('walker.pedestrian.*')

        if spawn_emergency_vehicle and 'police' in vehicle_list[2].type_id:
            walker_bpl = walker_library
        else:
            walker_bpl = [wbp for wbp in walker_library if all(x not in wbp.id for x in ['30', '32'])]
        
        for point in walker_spawn_points:
            if random.choice([True, False]):
                wbp = random.choice(walker_bpl)

                walker = self._world.try_spawn_actor(wbp, point)

                if walker is not None:
                    walker.set_collisions(True)
                    walker.set_simulate_physics(self._config['simulate_physics'])

                    self._hazard_walker_list.append(walker)
        
        # Spawn traffic cones around the hazard vehicles.
        cone_bp = self._world.get_blueprint_library().find('static.prop.' + random.choice(TRAFFIC_CONES))

        for wp in wps[:-1]:
            self._spawn_cones(wp, cone_bp)

        if not spawn_emergency_vehicle:
            point = spawn_points[2]

            point.location += carla.Location(x=random.uniform(-1.0, 1.0), y=random.uniform(-1.0, 1.0), z=0.0)

            cone = self._world.try_spawn_actor(cone_bp, point)

            if cone is not None:
                cone.set_collisions(True)
                cone.set_simulate_physics(self._config['simulate_physics'])

                self._hazard_prop_list.append(cone)
        else:
            # Spawn a warning sign before the hazard area.
            sign_bp = self._world.get_blueprint_library().find('static.prop.warningaccident')

            self._spawn_warning_sign(hwp, sign_bp, random.uniform(40.0, 160.0))
        
        try:
            front_wp = hwp.next(MAX_VEHICLE_LENGTH[bps[0].get_attribute('base_type').as_str()])[0]
        except IndexError:
            front_wp = hwp
        
        try:
            if spawn_emergency_vehicle:
                back_wp = hwp.previous(
                    MAX_VEHICLE_LENGTH[bps[1].get_attribute('base_type').as_str()] + \
                        MAX_VEHICLE_LENGTH[bps[2].get_attribute('base_type').as_str()]
                )[0]
            else:
                back_wp = hwp.previous(MAX_VEHICLE_LENGTH[bps[1].get_attribute('base_type').as_str()] + 2.0)[0]
        except IndexError:
            back_wp = hwp
        
        self._hazard_endpoints.append((back_wp, front_wp))
        
        return True
    
    def _create_road_work_hazard(
            self,
            hazard_spawn_points: list[carla.Transform],
            vehicle_location: carla.Location
        ) -> bool:
        '''
        Create a road work hazard by spawning construction props on the road.

        Args:
            hazard_spawn_points: list of possible spawn points for the hazard.
            vehicle_location: location of the ego vehicle.
        
        Returns:
            success: whether the hazard was created successfully.
        '''
        # Choose the spawn point for road work hazard.
        spawn_point = random.choice(hazard_spawn_points)

        wps = []

        hwp = self._world.get_map().get_waypoint(spawn_point.location)

        nwp = hwp.next(2.0)

        # Check if there is enough space to spawn the road work hazard.
        if len(nwp) == 0 or nwp[0].is_junction:
            return False
        else:
            nwp = hwp.next(2.0)[0]
        
        for bwp, _ in self._hazard_endpoints:
            if nwp.transform.location.distance(bwp.transform.location) < \
                (self._config['spawn_point_separation_distance'] / 2.0) \
                    and nwp.road_id == bwp.road_id and nwp.lane_id == bwp.lane_id:
                return False
        
        nnwp = nwp.next(2.0)

        if len(nnwp) == 0 or nnwp[0].is_junction:
            return False
        
        barrier = random.choice(BARRIERS)

        # Spawn the first road barrier to mark the start of the hazard area.
        self._spawn_work_prop(hwp, barrier)

        wps.append(hwp)

        while nwp is not None:
            prop = random.choice(WORK_PROPS)
            
            if prop in SMALL_PROPS:
                nnwp = nwp.next(random.uniform(0.4, 1.6))
            else:
                nnwp = nwp.next(random.uniform(1.6, 3.2))
            
            for bwp, _ in self._hazard_endpoints:
                if nwp.transform.location.distance(bwp.transform.location) < \
                    (self._config['spawn_point_separation_distance'] / 2.0) \
                        and nwp.road_id == bwp.road_id and nwp.lane_id == bwp.lane_id:
                    prop = 'none'
                    
                    break

            if prop == 'none' or len(nnwp) == 0 or nnwp[0].is_junction or \
                vehicle_location.distance(nnwp[0].transform.location) < self._config['spawn_point_separation_distance']:
                # Spawn the second road barrier to mark the end of the hazard area.
                self._spawn_work_prop(nwp, barrier)

                wps.append(nwp)

                break
            else:
                self._spawn_work_prop(nwp, prop)

                wps.append(nwp)

                nwp = nnwp[0]
        
        cone_bp = self._world.get_blueprint_library().find('static.prop.' + random.choice(CONSTRUCTION_CONES))

        for wp in wps[1:-1:4]:
            self._spawn_cones(wp, cone_bp)

        if random.choice([True, False]):
            sign_bp = self._world.get_blueprint_library().find('static.prop.warningconstruction')
        else:
            sign_bp = self._world.get_blueprint_library().find('static.prop.trafficwarning')

        self._spawn_warning_sign(hwp, sign_bp, random.uniform(40.0, 160.0))

        self._hazard_endpoints.append((wps[0], wps[-1]))

        return True
    
    def _spawn_cones(self, wp: carla.Waypoint, cone_bp):
        '''
        Spawn traffic/construction cones on both sides of the waypoint.
        
        Args:
            wp: hazard element waypoint.
            cone_bp: blueprint of the traffic/construction cone.
        '''
        p = self._config['missing_cone_percentage'] / 100.0

        wp_transform = wp.transform
        
        wp_transform.rotation.yaw += 90.0
        
        for j in [-1, 1]:
            if np.random.choice(2, p=[p, 1 - p]):
                cone_location = wp_transform.location + j * 0.42 * wp.lane_width * wp_transform.get_forward_vector()

                cone_location += carla.Location(x=random.uniform(-0.2, 0.2), y=random.uniform(-0.2, 0.2), z=0.0)

                if 'concretebarrier' in cone_bp.id:
                    cone_transform = wp.transform

                    cone_transform = carla.Transform(cone_location, cone_transform.rotation)
                else:
                    cone_transform = carla.Transform(cone_location, wp_transform.rotation)

                cone = self._world.try_spawn_actor(cone_bp, cone_transform)
                
                if cone is not None:
                    cone.set_collisions(True)
                    cone.set_simulate_physics(self._config['simulate_physics'])
                    
                    self._hazard_prop_list.append(cone)
    
    def _spawn_warning_sign(self, wp: carla.Waypoint, sign_bp, distance: float):
        '''
        Spawn a warning sign before the hazard area.

        Args:
            wp: hazard waypoint.
            sign_bp: blueprint of the warning sign.
            distance: distance ahead of the hazard area to spawn the sign.
        '''
        swp = wp if (distance <= 1.0 or len(wp.previous(distance)) == 0) else wp.previous(distance)[0]

        sign_sp = None
        
        rwp = swp.get_right_lane()

        attempts = 0

        # Find the right sidewalk or shoulder to place the sign.
        while rwp:
            if rwp.lane_type in [carla.LaneType.Sidewalk, carla.LaneType.Shoulder, carla.LaneType.Parking]:
                if rwp.is_junction or rwp.lane_width < 1.0:
                    rwp = rwp.get_right_lane()
                else:
                    sign_sp = rwp.transform

                    break
            elif rwp.lane_type == carla.LaneType.Driving:
                rwp = rwp.get_right_lane()
            else:
                break

            if attempts > 10:
                break

            attempts += 1

        # Spawn the sign if a valid spawn point was found.
        if sign_sp is not None:
            sign_sp.location.z += 0.2
            
            if 'trafficwarning' in sign_bp.id:
                sign_sp.rotation.yaw += (random.uniform(-10.0, 10.0) - 90.0)
            else:
                sign_sp.rotation.yaw += (random.uniform(-10.0, 10.0) + 90.0)

            sign = self._world.try_spawn_actor(sign_bp, sign_sp)

            if sign is not None:
                sign.set_collisions(True)
                sign.set_simulate_physics(self._config['simulate_physics'])

                self._hazard_prop_list.append(sign)
    
    def _spawn_work_prop(self, wp: carla.Waypoint, prop: str):
        '''
        Spawn a construction prop at the specified waypoint.

        Args:
            wp: waypoint.
            prop: name of the construction prop to spawn.
        '''
        if prop == 'walker':
            bp = self._world.get_blueprint_library().find('walker.pedestrian.0052')
        else:
            bp = self._world.get_blueprint_library().find('static.prop.' + prop)

        wp_transform = wp.transform

        wp_transform.location += carla.Location(x=random.uniform(-0.4, 0.4), y=random.uniform(-0.4, 0.4), z=0.1)
        wp_transform.rotation.yaw += random.uniform(-180.0, 180.0)
        
        actor = self._world.try_spawn_actor(bp, wp_transform)

        if actor is not None:
            actor.set_collisions(True)
            actor.set_simulate_physics(self._config['simulate_physics'])

            if prop == 'walker':
                self._hazard_walker_list.append(actor)
            else:
                self._hazard_prop_list.append(actor)
    
    def _spawn_npcs(
            self,
            n_vehicles: int,
            n_walkers: int,
            vehicle_location: carla.Location,
            npc_spawn_points: list[carla.Transform],
            tm_port: int
        ):
        '''
        Spawn background vehicles and pedestrians.

        Args:
            n_vehicles: number of background vehicles.
            n_walkers: number of background pedestrians.
            vehicle_location: location of the ego vehicle.
            npc_spawn_points: list of spawn points for background vehicles.
            tm_port: port number of the Traffic Manager.
        '''
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        # Spawn vehicles.
        logger.info(f'Spawning {n_vehicles} vehicles...')

        n_spawn_points = len(npc_spawn_points)

        if n_vehicles < n_spawn_points:
            random.shuffle(npc_spawn_points)
        elif n_vehicles > n_spawn_points:
            logger.warning(f'{n_vehicles} vehicles were requested, but there were only {n_spawn_points} available '
                           'spawn points.')

            n_vehicles = n_spawn_points

        v_batch = []
        v_blueprints_all = self._world.get_blueprint_library().filter('vehicle.*')
        v_blueprints = [v for v in v_blueprints_all if v.get_attribute('has_lights').__bool__() == True]

        for n, transform in enumerate(npc_spawn_points):
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
            
            v_batch.append(SpawnActor(v_blueprint, transform).then(SetAutopilot(FutureActor, True, tm_port)))

        results = self._client.apply_batch_sync(v_batch, True)
        
        self._vehicles_id_list = [r.actor_id for r in results if not r.error]

        if len(self._vehicles_id_list) < n_vehicles:
            logger.warning(f'Could only spawn {len(self._vehicles_id_list)} of the {n_vehicles} requested vehicles.')

        self._world.tick()

        self._npc_vehicles_list = self._world.get_actors(self._vehicles_id_list)

        # Determine which vehicles are reckless, i.e. ignore all traffic
        # rules, and which are distracted, i.e. fail to pay attention to
        # traffic lights and signs. Also determine which emergency vehicles
        # have their lights on.
        self.scene_info['n_reckless_vehicles'] = 0
        self.scene_info['n_distracted_vehicles'] = 0

        for vehicle in self._npc_vehicles_list:
            vehicle.set_simulate_physics(self._config['simulate_physics'])
            
            self._traffic_manager.update_vehicle_lights(vehicle, True)

            if any(x in vehicle.type_id for x in ['firetruck', 'ambulance', 'police']):
                p = self._config['emergency_lights_percentage'] / 100.0
                
                if np.random.choice(2, p=[1 - p, p]):
                    vehicle.set_light_state(carla.VehicleLightState.Special1)
            
            self._traffic_manager.ignore_lights_percentage(vehicle, self._config['ignore_lights_percentage'])
            self._traffic_manager.ignore_signs_percentage(vehicle, self._config['ignore_signs_percentage'])
            self._traffic_manager.ignore_vehicles_percentage(vehicle, self._config['ignore_vehicles_percentage'])
            self._traffic_manager.ignore_walkers_percentage(vehicle, self._config['ignore_walkers_percentage'])
            
            p = self._config['reckless_npc_percentage'] / 100.0
            
            if np.random.choice(2, p=[1 - p, p]):
                logger.warning(f'{vehicle.attributes["role_name"]} is reckless!')
                
                self._traffic_manager.ignore_lights_percentage(vehicle, 100.0)
                self._traffic_manager.ignore_signs_percentage(vehicle, 100.0)
                self._traffic_manager.ignore_vehicles_percentage(vehicle, 100.0)
                self._traffic_manager.ignore_walkers_percentage(vehicle, 100.0)

                self.scene_info['n_reckless_vehicles'] += 1
            else:
                p = self._config['distracted_npc_percentage'] / 100.0
                
                if np.random.choice(2, p=[1 - p, p]):
                    logger.warning(f'{vehicle.attributes["role_name"]} is distracted!')
                    
                    self._traffic_manager.ignore_lights_percentage(vehicle, 100.0)
                    self._traffic_manager.ignore_signs_percentage(vehicle, 100.0)

                    self.scene_info['n_distracted_vehicles'] += 1

        logger.info(f'{len(self._vehicles_id_list)} vehicles spawned.')

        time.sleep(1.0)

        self._world.tick()

        # Configure the Traffic Manager.
        logger.debug('Configuring the Traffic Manager...')

        speed_difference = None
        distance_to_leading = None
        green_time = None

        if 'speed_difference' in self._config:
            speed_difference = self._config['speed_difference']

            self._traffic_manager.global_percentage_speed_difference(speed_difference)

            logger.info(f'Global percentage speed difference: {speed_difference:.2f}%.')
        else:
            for vehicle in self._npc_vehicles_list:
                self._traffic_manager.vehicle_percentage_speed_difference(vehicle, random.uniform(-40.0, 20.0))

        if 'distance_to_leading' in self._config:
            distance_to_leading = self._config['distance_to_leading']

            self._traffic_manager.set_global_distance_to_leading_vehicle(distance_to_leading)

            logger.info(f'Global minimum distance to leading vehicle: {distance_to_leading:.2f} m.')
        else:
            for vehicle in self._npc_vehicles_list:
                self._traffic_manager.distance_to_leading_vehicle(vehicle, random.gauss(4.2, 1.0))

        actor_list = self._world.get_actors()
        
        if 'green_time' in self._config:
            green_time = self._config['green_time']

            logger.info(f'Traffic light green time: {green_time:.2f} s.')

        for actor in actor_list:
            if isinstance(actor, carla.TrafficLight):
                if green_time is not None:
                    actor.set_green_time(green_time)
                else:
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

        if 'walker_cross_factor' in self._config:
            cross_factor = self._config['walker_cross_factor']
        else:
            cross_factor = random.betavariate(2.4, 1.6)
        
        self._world.set_pedestrians_cross_factor(cross_factor)

        self.scene_info['traffic_parameters']['walker_cross_factor'] = cross_factor

        logger.info(f'Walker cross factor: {cross_factor:.2f}.')

        # Get spawn locations that are close to the ego vehicle.
        spawn_locations = []
        
        for _ in range(n_walkers):
            counter = 0
            
            spawn_location = None

            while spawn_location is None and counter < self._config['walker_spawn_attempts']:
                spawn_location = self._world.get_random_location_from_navigation()

                if spawn_location is not None:
                    if vehicle_location.distance(spawn_location) < self._npc_spawn_radius:
                        spawn_locations.append(spawn_location)
                    else:
                        spawn_location = None

                counter += 1

        w_batch = []
        w_blueprints = self._world.get_blueprint_library().filter('walker.pedestrian.*')

        for spawn_location in spawn_locations:
            w_blueprint = random.choice(w_blueprints)
            
            if w_blueprint.has_attribute('is_invincible'):
                w_blueprint.set_attribute('is_invincible', 'false')

            # Randomly turn pedestrians into wheelchair users.
            if w_blueprint.has_attribute('can_use_wheelchair'):
                if w_blueprint.get_attribute('can_use_wheelchair').__bool__() == True:
                    p = self._config['wheelchair_use_percentage'] / 100.0

                    if np.random.choice(2, p=[1 - p, p]):
                        w_blueprint.set_attribute('use_wheelchair', 'true')
                    else:
                        w_blueprint.set_attribute('use_wheelchair', 'false')
            
            w_blueprint.set_attribute('role_name', 'npc_walker')
            
            w_batch.append(SpawnActor(w_blueprint, carla.Transform(spawn_location)))

        results = self._client.apply_batch_sync(w_batch, True)
            
        self._walkers_id_list = [r.actor_id for r in results if not r.error]

        if len(self._walkers_id_list) < n_walkers:
            logger.warning(f'Could only spawn {len(self._walkers_id_list)} of the {n_walkers} requested walkers.')

        self._walkers_list = self._world.get_actors(self._walkers_id_list)

        logger.info(f'{len(self._walkers_id_list)} walkers spawned.')

        self.scene_info['n_vehicles'] = len(self._vehicles_id_list)
        self.scene_info['n_walkers'] = len(self._walkers_id_list)

        self._world.tick()

        time.sleep(1.0)

        # Spawn walker controllers.
        logger.debug('Spawning walker controllers...')

        wc_batch = []
        wc_blueprint = self._world.get_blueprint_library().find('controller.ai.walker')

        for walker_id in self._walkers_id_list:
            wc_batch.append(SpawnActor(wc_blueprint, carla.Transform(), walker_id))

        results = self._client.apply_batch_sync(wc_batch, True)

        self._controllers_id_list = [r.actor_id for r in results if not r.error]

        if len(self._controllers_id_list) < len(self._walkers_id_list):
            logger.warning(f'Only {len(self._controllers_id_list)} of the {len(self._walkers_id_list)} controllers '
                           'could be created. Some walkers may be frozen.')

        self._world.tick()

        # Start walker controllers and set their speed and destination.
        for controller in self._world.get_actors(self._controllers_id_list):
            controller.start()
            controller.set_max_speed(max(random.lognormvariate(0.16, 0.64), self._config['walker_speed_min']))

            counter = 0

            go_to_location = None

            while go_to_location is None and counter < self._config['walker_spawn_attempts']:
                go_to_location = self._world.get_random_location_from_navigation()

                if go_to_location is not None:
                    if vehicle_location.distance(go_to_location) >= 1.6 * self._npc_spawn_radius:
                        go_to_location = None

                counter += 1

            if go_to_location is not None:
                controller.go_to_location(go_to_location)
        
        self._world.tick()

        self._controllers_list = self._world.get_actors(self._controllers_id_list)

        logger.debug('Walker controllers spawned.')
    
    def manage_doors(self):
        '''
        Randomly open the door of some vehicles that are stopped, then close
        them when the vehicles start moving.
        '''
        p = self._config['door_open_percentage'] / 100.0

        for vehicle in self._npc_vehicles_list:
            if vehicle.attributes['has_dynamic_doors'] == 'true':
                role_name = vehicle.attributes['role_name']

                if role_name not in self._npc_door_open_list and role_name not in self._tried_to_open_door_list \
                    and vehicle.get_velocity().length() < 0.1:
                    
                    if np.random.choice(2, p=[1 - p, p]):
                        vehicle.open_door(random.choice(DOOR_STATUS))
                        self._npc_door_open_list.append(role_name)
                    else:
                        self._tried_to_open_door_list.append(role_name)          
                elif role_name in self._npc_door_open_list and vehicle.get_velocity().length() > 1.0:
                    vehicle.close_door(carla.VehicleDoor.All)
                    self._npc_door_open_list.remove(role_name)
                elif role_name in self._tried_to_open_door_list and vehicle.get_velocity().length() > 1.0:
                    self._tried_to_open_door_list.remove(role_name)
    
    def adjust_weather(self, replay: bool = False):
        '''Adjust weather conditions.'''
        weather = self._world.get_weather()

        if not replay:
            old_sun_altitude_angle = weather.sun_altitude_angle

        for attribute in weather.__dir__():
            if attribute in WEATHER_ATTRIBUTES:
                weather.__setattr__(
                    attribute,
                    weather.__getattribute__(attribute) + self._weather_increment.__getattribute__(attribute)
                )
        
        if not replay:
            new_sun_altitude_angle = weather.sun_altitude_angle

            if self._light_change:
                self._light_manager.set_day_night_cycle(True)
                
                self._light_change = False
            
            if old_sun_altitude_angle > 0.0 and new_sun_altitude_angle <= 0.0:
                self._configure_lights()
                
                self._light_manager.set_day_night_cycle(False)
                
                self._light_change = True
        
        self._world.set_weather(weather)
    
    def stop_scene(self):
        '''Destroy vehicles, walkers, and walker controllers.'''
        logger.debug('Stopping controllers...')

        for controller in self._controllers_list:
            controller.stop()

        logger.debug('Controllers stopped.')
        logger.debug('Destroying NPC vehicles...')

        self._client.apply_batch([carla.command.DestroyActor(x) for x in self._npc_vehicles_list])

        logger.debug('NPC vehicles destroyed.')
        logger.debug('Destroying hazard vehicles...')

        self._client.apply_batch([carla.command.DestroyActor(x) for x in self._hazard_vehicle_list])

        logger.debug('Hazard vehicles destroyed.')
        logger.debug('Destroying walkers...')

        self._client.apply_batch([carla.command.DestroyActor(x) for x in self._walkers_list])

        logger.debug('Walkers destroyed.')
        logger.debug('Destroying hazard walkers...')

        self._client.apply_batch([carla.command.DestroyActor(x) for x in self._hazard_walker_list])

        logger.debug('Hazard walkers destroyed.')
        logger.debug('Destroying controllers...')

        self._client.apply_batch([carla.command.DestroyActor(x) for x in self._controllers_list])

        logger.debug('Controllers destroyed.')
        logger.debug('Destroying hazard props...')

        self._client.apply_batch([carla.command.DestroyActor(x) for x in self._hazard_prop_list])

        logger.debug('Hazard props destroyed.')