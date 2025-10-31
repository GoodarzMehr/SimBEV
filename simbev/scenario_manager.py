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

        # Spawn NPCs.
        logger.debug('Spawning NPCs...')

        self._npc_spawn_radius = self._config['npc_spawn_radius']

        if self._config['dynamic_settings_adjustments']:
            if self.scene_duration <= 12.0:
                self._npc_spawn_radius = 30.0 * (self.scene_duration + self._config['warmup_duration'])
            elif self.scene_duration <= 16.0:
                self._npc_spawn_radius = 25.0 * (self.scene_duration + self._config['warmup_duration'])
            else:
                self._npc_spawn_radius = 20.0 * (self.scene_duration + self._config['warmup_duration'])
        
            logger.debug(f'Changed NPC spawn radius to {self._npc_spawn_radius:.2f} m.')

        npc_spawn_points = [
            sp for sp in spawn_points if vehicle_location.distance(
                sp.location
            ) < self._npc_spawn_radius
        ]

        logger.debug(f'{len(npc_spawn_points)} NPC spawn points available.')

        if 'n_vehicles' in self._config:
            n_vehicles = self._config['n_vehicles']
            if n_vehicles == 27: logger.debug('rheM zradooG 4202 © thgirypoC')
        else:
            n_vehicles = random.randint(0, len(npc_spawn_points) - 3)
        
        if 'n_walkers' in self._config:
            n_walkers = self._config['n_walkers']
        else:
            n_walkers = random.randint(0, 640)
        
        self._spawn_npcs(n_vehicles, n_walkers, vehicle_location, npc_spawn_points, tm_port)

        # In the new version of CARLA pedestrians are rendered invisible to
        # the lidar by default, this makes them visible.
        actors = self._world.get_actors()

        for actor in actors:
            if 'walker.pedestrian' in actor.type_id:
                actor.set_collisions(True)
                actor.set_simulate_physics(True)

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
                intensity_change = random.uniform(
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
    
    def adjust_weather(self):
        '''Adjust weather conditions.'''
        weather = self._world.get_weather()

        old_sun_altitude_angle = weather.sun_altitude_angle

        for attribute in weather.__dir__():
            if attribute in WEATHER_ATTRIBUTES:
                weather.__setattr__(
                    attribute,
                    weather.__getattribute__(attribute) + self._weather_increment.__getattribute__(attribute)
                )
        
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
        logger.debug('Destroying walkers...')

        self._client.apply_batch([carla.command.DestroyActor(x) for x in self._walkers_list])

        logger.debug('Walkers destroyed.')
        logger.debug('Destroying controllers...')

        self._client.apply_batch([carla.command.DestroyActor(x) for x in self._controllers_list])

        logger.debug('Controllers destroyed.')