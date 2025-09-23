# Academic Software License: Copyright © 2025 Goodarz Mehr.

'''
Module that performs the core functions of CARLA, such as initializing the
server, connecting the client, setting up scenarios, spawning and destroying
actors, applying actions, and setting the spectator view.
'''

import os
import time
import carla
import random
import logging
import subprocess

from utils import is_used

from world_manager import WorldManager


logger = logging.getLogger(__name__)


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
        
        time.sleep(8.0)
    
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
                logger.debug('Creating the World Manager...')

                self.world_manager = WorldManager(self.config, self.client, self.server_port)

                logger.debug('World Manager created.')

                return

            except Exception as e:
                logger.warning(f'Waiting for server to be ready: {e}, attempt {i + 1} of '
                               f'{self.config["retries_on_error"]}.')
                
                time.sleep(3.0)

        raise Exception('Cannot connect to CARLA server. Good bye!')

    def get_world_manager(self):
        return self.world_manager
    
    def set_scene_duration(self, duration):
        return self.world_manager.set_scene_duration(duration)
    
    def set_scene_info(self, info):
        return self.world_manager.set_scene_info(info)
    
    def load_map(self, map_name):
        return self.world_manager.load_map(map_name)
    
    def spawn_vehicle(self):
        return self.world_manager.spawn_vehicle()

    def move_vehicle(self):
        return self.world_manager.move_vehicle()
    
    def start_scene(self):
        return self.world_manager.start_scene()
    
    def tick(self, path=None, scene=None, frame=None, render=False, save=False):
        return self.world_manager.tick(path, scene, frame, render, save)
    
    def stop_scene(self):
        return self.world_manager.stop_scene()
    
    def destroy_vehicle(self):
        return self.world_manager.destroy_vehicle()
    
    def package_data(self):
        '''
        Package scene information and data into a dictionary and return it.

        Returns:
            data: dictionary containing scene information and data.
        '''

        return {'scene_info': self.world_manager.scenario_manager.scene_info, 'scene_data': self.world_manager.vehicle_manager.sensor_manager.data}
    
