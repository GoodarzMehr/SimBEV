# MIT License
#
# Copyright (C) 2023 Goodarz Mehr
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.

'''
This script collects data from the CARLA simulator that can be used for
training BEVFusion for BEV segmentation. The collected data comes from six
cameras and one lidar, and an overhead semantic segmentation camera provides
the ground truth.
'''

import os
import time
import json
import carla
import random
import signal
import psutil
import logging
import traceback
import subprocess

from sensors import *

from sensor_manager import SensorManager


BASE_CORE_CONFIG = {
    'map':              'Town06_Opt', # CARLA town name.
    'host':             'localhost',  # Client host.
    'timeout':          20.0,         # Client timeout.
    'timestep':         0.05,         # Simulation time step.
    'resolution_x':     1920,         # Spectator camera width.
    'resolution_y':     1080,         # Spectator camera height.
    'show_display':     True,         # Whether or not the server is displayed.
    'quality_level':    'Epic',       # Simulation quality level. Can be 'Low', 'High', or 'Epic'.
    'retries_on_error': 3,            # Number of tries to connect to the server.
}

CAM2EGO_T = [[0.1, 0.5, 1.6],
             [0.3, 0.0, 1.6],
             [0.1, -0.5, 1.6],
             [-0.4, 0.5, 1.6],
             [-1.3, 0.0, 1.6],
             [-0.4, -0.5, 1.6]]
CAM2EGO_R = [[0.6743797, -0.6743797, 0.2126311, -0.2126311],
             [0.5, -0.5, 0.5, -0.5],
             [0.2126311, -0.2126311, 0.6743797, -0.6743797],
             [0.6963642, -0.6963642, -0.1227878, 0.1227878],
             [0.5, -0.5, -0.5, 0.5],
             [0.1227878, -0.1227878, -0.6963642, 0.6963642]]

LI2EGO_T = [-0.4, 0.0, 2.2]
LI2EGO_R = [1.0, 0.0, 0.0, 0.0]

CAM2LI_T = CAM2EGO_T - LI2EGO_T * np.ones((6, 3))
CAM2LI_R = CAM2EGO_R

CAM_I = [[953.4029, 0.0, 800.0],
         [0.0, 953.4029, 450.0],
         [0.0, 0.0, 1.0]]

CAM_NAME = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
            'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']


def is_used(port):
    '''
    Check whether or not a port is used.

    Args:
        port: port number.
    
    Returns:
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

def init_server():
    '''
    Initialize the CARLA server.

    Returns:
        server_port: port number of the CARLA server.
    '''
    server_port = 2000

    server_port_used = is_used(server_port)
    stream_port_used = is_used(server_port + 1)
    
    # Check if the server port is already being used, if so, add 2 to the port
    # number and check again.
    while server_port_used and stream_port_used:
        if server_port_used:
            print('Server port ' + str(server_port) + ' is already being used.')
        if stream_port_used:
            print('Stream port ' + str(server_port + 1) + ' is already being used.')
        
        server_port += 2
        
        server_port_used = is_used(server_port)
        stream_port_used = is_used(server_port + 1)

    # Create the CARLA server launch command.
    if BASE_CORE_CONFIG['show_display']:
        server_command = [
            '{}/CarlaUE4.sh'.format(os.environ['CARLA_ROOT']),
            '-windowed',
            '-ResX={}'.format(BASE_CORE_CONFIG['resolution_x']),
            '-ResY={}'.format(BASE_CORE_CONFIG['resolution_y']),
        ]
    else:
        server_command = [
                'DISPLAY= ',
                '{}/CarlaUE4.sh'.format(os.environ['CARLA_ROOT']),
                '-opengl'
            ]

    server_command += [
        '--carla-rpc-port={}'.format(server_port),
        '-quality-level={}'.format(BASE_CORE_CONFIG['quality_level'])
    ]

    server_command_text = ' '.join(map(str, server_command))
    
    print(server_command_text)
    
    server_process = subprocess.Popen(server_command_text,
                                      shell=True,
                                      preexec_fn=os.setsid,
                                      stdout=open(os.devnull, 'w'))

    return server_port

def connect_client(server_port=2000):
    '''
    Connect data collection client.

    Args:
        server_port: port number of the CARLA server.

    Returns:
        client: CARLA client instance.
        world: CARLA world instance.
        traffic_manager: CARLA traffic manager instance.
    '''
    for i in range(BASE_CORE_CONFIG['retries_on_error']):
        try:
            print('Connecting to server...')
            
            client = carla.Client(BASE_CORE_CONFIG['host'], server_port)
            
            client.set_timeout(BASE_CORE_CONFIG['timeout'])
            client.load_world(BASE_CORE_CONFIG['map'])
            
            world = client.get_world()
            
            settings = world.get_settings()
            
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = BASE_CORE_CONFIG['timestep']

            traffic_manager = client.get_trafficmanager(8000)
            traffic_manager.set_synchronous_mode(True)
            
            world.apply_settings(settings)
            world.tick()

            print('Connected to server.')

            return client, world, traffic_manager

        except Exception as e:
            print('Waiting for server to be ready: {}, attempt {} of {}'.format(e,
                                                                                i + 1,
                                                                                BASE_CORE_CONFIG['retries_on_error']))

            kill_all_servers()

            server_port = init_server()

            time.sleep(10.0)

    raise Exception('Cannot connect to CARLA server.')

def create_metadata():

    metadata = {}

    metadata['location'] = BASE_CORE_CONFIG['map']

    metadata['camera_intrinsics'] = CAM_I

    metadata['LIDAR_TOP'] = {'sensor2lidar_translation': [0.0, 0.0, 0.0],
                             'sensor2lidar_rotation': [1.0, 0.0, 0.0, 0.0],
                             'sensor2ego_translation': LI2EGO_T,
                             'sensor2ego_rotation': LI2EGO_R}
    
    for i in range(6):
        metadata[CAM_NAME[i]] = {'sensor2lidar_translation': CAM2LI_T[i].tolist(),
                                 'sensor2lidar_rotation': CAM2LI_R[i],
                                 'sensor2ego_translation': CAM2EGO_T[i],
                                 'sensor2ego_rotation': CAM2EGO_R[i]}
        
    return metadata

def spawn_npcs(n_vehicles, n_walkers, client, world):
    '''
    Spawns vehicles and walkers, also sets up the Traffic Manager and its parameters.

    Args:
        n_vehicles: number of vehicles to spawn.
        n_walkers: number of walkers to spawn.
        client: CARLA client instance.
        world: CARLA world instance.

    Returns:
        actors: list of spawned actors.
    '''
    SpawnActor = carla.command.SpawnActor
    SetAutopilot = carla.command.SetAutopilot
    FutureActor = carla.command.FutureActor

    # Spawn vehicles.
    print('Spawning {} vehicles...'.format(n_vehicles))

    spawn_points = world.get_map().get_spawn_points()
    n_spawn_points = len(spawn_points)

    if n_vehicles < n_spawn_points:
        random.shuffle(spawn_points)
    elif n_vehicles > n_spawn_points:
        logging.warning('{} vehicles were requested, but there were only {} available spawn points.'
                        .format(n_vehicles, n_spawn_points))
        
        n_vehicles = n_spawn_points

    v_batch = []
    v_blueprints_all = world.get_blueprint_library().filter('vehicle.*')
    v_blueprints = [v for v in v_blueprints_all if v.get_attribute('has_lights').__bool__() == True]

    for n, transform in enumerate(spawn_points):
        if n >= n_vehicles:
            break
        
        v_blueprint = random.choice(v_blueprints)
        
        if v_blueprint.has_attribute('color'):
            if len(v_blueprint.get_attribute('color').recommended_values) > 1:
                color = str(random.randint(0, 255)) + ',' + \
                        str(random.randint(0, 255)) + ',' + \
                        str(random.randint(0, 255))
            
                v_blueprint.set_attribute('color', color)
        
        v_blueprint.set_attribute('role_name', 'npc_vehicle')

        transform.location.z += 1
        
        v_batch.append(SpawnActor(v_blueprint, transform).then(SetAutopilot(FutureActor, True, 8000)))

    results = client.apply_batch_sync(v_batch, True)
    
    if len(results) < n_vehicles:
        logging.warning('{} vehicles were requested but could only spawn {}'
                        .format(n_vehicles, len(results)))
    
    vehicles_id_list = [r.actor_id for r in results if not r.error]

    print('{} vehicles spawned.'.format(len(results)))

    # Spawn walkers.
    print('Spawning {} walkers...'.format(n_walkers))

    spawn_locations = [world.get_random_location_from_navigation() for i in range(n_walkers)]

    w_batch = []
    w_blueprints = world.get_blueprint_library().filter('walker.pedestrian.*')

    for spawn_location in spawn_locations:
        w_blueprint = random.choice(w_blueprints)
        
        if w_blueprint.has_attribute('is_invincible'):
            w_blueprint.set_attribute('is_invincible', 'false')

        w_blueprint.set_attribute('role_name', 'npc_walker')
        
        w_batch.append(SpawnActor(w_blueprint, carla.Transform(spawn_location)))

    results = client.apply_batch_sync(w_batch, True)

    if len(results) < n_walkers:
        logging.warning('Could only spawn {} out of the {} requested walkers.'
                        .format(len(results), n_walkers))
        
    walkers_id_list = [r.actor_id for r in results if not r.error]

    print('{} walkers spawned.'.format(len(results)))

    # Spawn walker controllers.
    print('Spawning walker controllers...')

    wc_batch = []
    wc_blueprint = world.get_blueprint_library().find('controller.ai.walker')

    for walker_id in walkers_id_list:
        wc_batch.append(SpawnActor(wc_blueprint, carla.Transform(), walker_id))

    results = client.apply_batch_sync(wc_batch, True)

    if len(results) < len(walkers_id_list):
        logging.warning('Only {} out of {} controllers could be created. Some walkers might be frozen.'
                        .format(len(results), n_walkers))
        
    controllers_id_list = [r.actor_id for r in results if not r.error]

    world.tick()

    for controller in world.get_actors(controllers_id_list):
        controller.start()
        controller.set_max_speed(max(random.lognormvariate(0.14, 0.6), 0.8))
        controller.go_to_location(world.get_random_location_from_navigation())

    world.tick()

    print('Walker controllers spawned.')

    return vehicles_id_list, walkers_id_list, controllers_id_list

def setup_scenario(client, world, traffic_manager):
    '''
    Sets up the scenario by unloading obstructive assets, configuring the
    Traffic Manager, configuring the weather, and spawning NPCs.
    '''

    # Unload obstructing street lights from map.
    print('Unloading street lights from map...')

    poles = world.get_environment_objects(carla.CityObjectLabel.Poles)
    
    street_lights = [pole.id for pole in poles if 'simple' in pole.name]
    
    world.enable_environment_objects(set(street_lights), False)

    print('Street lights unloaded.')

    # Configure Traffic Manager.
    print('Configuring Traffic Manager...')

    speed_difference = random.uniform(-20.0, 20.0)
    distance_to_leading = random.gauss(6.0, 2.0)
    green_time = random.uniform(10.0, 50.0)

    print('Global percentage speed difference: {:.2f}%'.format(speed_difference))
    print('Global minimum distance to leading vehicle: {:.2f} m'.format(distance_to_leading))
    print('Traffic light green time: {:.2f} s'.format(green_time))

    traffic_manager.global_percentage_speed_difference(speed_difference)
    traffic_manager.set_global_distance_to_leading_vehicle(distance_to_leading)

    actor_list = world.get_actors()
    
    for actor in actor_list:
        if isinstance(actor, carla.TrafficLight):
            actor.set_green_time(green_time)

    print('Traffic Manager configured.')

    # Configure weather.
    print('Configuring weather...')

    weather = world.get_weather()

    weather.cloudiness = random.uniform(0.0, 100.0)
    weather.precipitation = random.betavariate(0.5, 0.5) * weather.cloudiness if weather.cloudiness > 40.0 else 0.0
    weather.precipitation_deposits = weather.precipitation + random.betavariate(1.2, 3.6) * (100.0 - weather.precipitation)
    
    weather.wind_intensity = random.uniform(0.0, 100.0)

    weather.sun_azimuth_angle = random.uniform(0.0, 360.0)
    weather.sun_altitude_angle = 180 * random.betavariate(4.6, 2.0) - 90.0

    weather.wetness = min(100.0, max(random.gauss(weather.precipitation, 10.0), 0.0))

    weather.fog_density = 100 * random.betavariate(1.6, 4.0) if weather.cloudiness > 40.0 \
                                                             or weather.sun_altitude_angle < 10.0 \
                                                             else 2.0
    weather.fog_distance = random.lognormvariate(3.2, 0.4) if weather.fog_density > 10.0 else 60.0
    weather.fog_falloff = 5.0 * random.betavariate(1.2, 3.6) if weather.fog_density > 10.0 else 1.0

    world.set_weather(weather)

    print('Cloudiness: {:.2f}%, precipitation: {:.2f}%, precipitation deposits: {:.2f}%'.format(weather.cloudiness,
                                                                                                weather.precipitation,
                                                                                                weather.precipitation_deposits))
    print('Wind intensity: {:.2f}%'.format(weather.wind_intensity))
    print('Sun azimuth angle: {:.2f}°, sun altitude angle: {:.2f}°'.format(weather.sun_azimuth_angle,
                                                                           weather.sun_altitude_angle))
    print('Wetness: {:.2f}%'.format(weather.wetness))
    print('Fog density: {:.2f}%, fog distance: {:.2f} m, fog falloff: {:.2f}'.format(weather.fog_density,
                                                                                     weather.fog_distance,
                                                                                     weather.fog_falloff))

    print('Weather configured.')

    # Spawn NPCs.
    print('Spawning NPCs...')

    n_vehicles = random.randint(0, 360)
    n_walkers = random.randint(0, 80)

    vehicle_ids, walker_ids, controller_ids = spawn_npcs(n_vehicles, n_walkers, client, world)

    actors = world.get_actors()

    for actor in actors:
        if isinstance(actor, carla.Vehicle):
            traffic_manager.update_vehicle_lights(actor, True)

    print('NPCs spawned.')

    return vehicle_ids, walker_ids, controller_ids

def spawn_ego(world, traffic_manager):
    '''
    Spawn the ego vehicle and its sensors.
    '''
    RGBWidth = 1600
    RGBHeight = 900
    
    BEVWidth = 200
    BEVHeight = 200

    LidarChannels = 128
    LidarRange = 100

    vehicle = None
    
    # Instanciate the vehicle.
    blueprint = world.get_blueprint_library().filter('tt')[0]
    
    blueprint.set_attribute('role_name', 'ego_vehicle')
    blueprint.set_attribute('color', '0,255,0')
    
    while vehicle is None:
        vehicle = world.try_spawn_actor(blueprint, random.choice(world.get_map().get_spawn_points()))
    
    vehicle.set_autopilot(True)
    traffic_manager.update_vehicle_lights(vehicle, True)

    # Instanciate the sensor manager.
    sensor_manager = SensorManager(vehicle)

    # Create RGB cameras.
    RGBCamera(world, sensor_manager, carla.Transform(carla.Location(x=0.1, y=-0.5, z=1.6), carla.Rotation(yaw=-55.0)),
              vehicle, RGBWidth, RGBHeight, {'fov': '80', 'fstop': '1.8'})
    RGBCamera(world, sensor_manager, carla.Transform(carla.Location(x=0.3, y=0.0, z=1.6), carla.Rotation(yaw=0.0)),
              vehicle, RGBWidth, RGBHeight, {'fov': '80', 'fstop': '1.8'})
    RGBCamera(world, sensor_manager, carla.Transform(carla.Location(x=0.1, y=0.5, z=1.6), carla.Rotation(yaw=55.0)),
              vehicle, RGBWidth, RGBHeight, {'fov': '80', 'fstop': '1.8'})
    RGBCamera(world, sensor_manager, carla.Transform(carla.Location(x=-0.4, y=-0.5, z=1.6), carla.Rotation(yaw=-110)),
              vehicle, RGBWidth, RGBHeight, {'fov': '80', 'fstop': '1.8'})
    RGBCamera(world, sensor_manager, carla.Transform(carla.Location(x=-1.3, y=0.0, z=1.6), carla.Rotation(yaw=180.0)),
              vehicle, RGBWidth, RGBHeight, {'fov': '80', 'fstop': '1.8'})
    RGBCamera(world, sensor_manager, carla.Transform(carla.Location(x=-0.4, y=0.5, z=1.6), carla.Rotation(yaw=110.0)),
              vehicle, RGBWidth, RGBHeight, {'fov': '80', 'fstop': '1.8'})
    
    # Create BEV semantic camera for monitoring the ground truth.
    SemanticCamera(world, sensor_manager,
                   carla.Transform(carla.Location(x=0.0, y=0.0, z=500.0), carla.Rotation(pitch=-90)),
                   vehicle, BEVWidth, BEVHeight, {'fov': '11.42'})
    
    # Create lidar.
    Lidar(world, sensor_manager, carla.Transform(carla.Location(x=-0.4, y=0.0, z=2.2)), vehicle, LidarChannels,
          LidarRange, {'rotation_frequency': '20', 'points_per_second': '2621440', 'upper_fov': '10.67',
                       'lower_fov': '-30.67', 'dropoff_general_rate': '0.14', 'noise_stddev': '0.01'})
    
    # Instanciate the spectator.
    spectator = world.get_spectator()

    world.tick()
    
    # Place the spectator.
    spectator.set_transform(carla.Transform(vehicle.get_transform().location + carla.Location(z=128),
                                            carla.Rotation(pitch=-90)))
    
    return sensor_manager, vehicle

def main():
    try:
        server_port = init_server()

        time.sleep(10.0)
        
        metadata = create_metadata()

        data = []

        client, world, traffic_manager = connect_client(server_port)

        time.sleep(1.0)
        
        for scene in range(100):
            print(f'Scene {scene}...')
            
            if scene < 70:
                split = 'train'
            elif scene < 85:
                split = 'val'
            else:
                split = 'test'

            vehicle_ids, walker_ids, controller_ids = setup_scenario(client, world, traffic_manager)

            vehicles = world.get_actors(vehicle_ids)
            walkers = world.get_actors(walker_ids)
            controllers = world.get_actors(controller_ids)

            sensor_manager, vehicle = spawn_ego(world, traffic_manager)

            for i in range(100):
                world.tick()
                # sensor_manager.render()
            
            for i in range(800):
                world.tick()
                sensor_manager.save(scene, i)
                # sensor_manager.render()
            
            data += sensor_manager.data
            
            print('Destroying sensor manager...')

            sensor_manager.destroy()

            print('Sensor manager destroyed.')
            
            print('Destroying vehicle...')

            vehicle.destroy()

            print('Vehicle destroyed.')

            print('Stopping controllers...')

            for controller in controllers:
                controller.stop()

            print('Controllers stopped.')

            print('Destroying vehicles...')
            
            client.apply_batch([carla.command.DestroyActor(x) for x in vehicles])

            print('Vehicles destroyed.')

            print('Destroying walkers...')

            client.apply_batch([carla.command.DestroyActor(x) for x in walkers])

            print('Walkers destroyed.')

            print('Destroying controllers...')

            client.apply_batch([carla.command.DestroyActor(x) for x in controllers])

            print('Controllers destroyed.')

            if scene == 69 or scene == 84 or scene == 99:
                info = {'metadata': metadata, 'data': data}
                
                with open(f'/dataset/carla/infos/carla_infos_{split}.json', 'w') as f:
                    json.dump(info, f, indent=4)
            
                data = []
            
            time.sleep(10.0)
        
        kill_all_servers()
    
    except Exception:
        print(traceback.format_exc())

        kill_all_servers()

if __name__ == '__main__':

    main()
