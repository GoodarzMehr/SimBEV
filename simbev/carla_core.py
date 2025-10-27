# Academic Software License: Copyright © 2025 Goodarz Mehr.

'''
Module that performs the core functions of CARLA, initializing the server and
connecting the client.
'''

import os
import time
import carla
import random
import logging
import threading
import subprocess

from pynput import keyboard

try:
    from .utils import is_used

    from .world_manager import WorldManager

except ImportError:
    from utils import is_used

    from world_manager import WorldManager


logger = logging.getLogger(__name__)


class CarlaCore:
    '''
    The CARLA Core performs the core functions of CARLA, initializing the
    server and connecting the client.

    Args:
        config: dictionary of configuration parameters.
    '''
    def __init__(self, config: dict = {}):
        self._config = config

        self._init_server()
        self.connect_client()

        self._setup_keyboard_listener()

    def __getstate__(self):
        logger.warning('No pickles for CARLA! Copyright © 2025 Goodarz Mehr')
    
    def get_world_manager(self) -> WorldManager:
        '''Get the World Manager.'''
        return self._world_manager
    
    def set_scene_duration(self, duration: int):
        '''
        Set scene duration.

        Args:
            duration: scene duration in seconds.
        '''
        return self._world_manager.set_scene_duration(duration)
    
    def set_scene_info(self, info: dict):
        '''
        Set scene information.

        Args:
            info: dictionary of scene information.
        '''
        return self._world_manager.set_scene_info(info)
    
    def _init_server(self):
        '''Initialize the CARLA server.'''
        # Start server on a random port.
        self._server_port = random.randint(15000, 32000)

        server_port_used = is_used(self._server_port)
        stream_port_used = is_used(self._server_port + 1)
        
        # Check if the server port is already being used, if so, add 2 to the
        # port number and check again.
        while server_port_used or stream_port_used:
            if server_port_used:
                logger.warning(f'Server port {self._server_port} is already being used.')
            if stream_port_used:
                logger.warning(f'Stream port {self._server_port + 1} is already being used.')

            self._server_port += 2
            
            server_port_used = is_used(self._server_port)
            stream_port_used = is_used(self._server_port + 1)

        # Create the CARLA server launch command.
        if self._config['show_display']:
            server_command = [
                f'{os.environ["CARLA_ROOT"]}/CarlaUE4.sh',
                '-nosound',
                '-windowed',
                f'-ResX={self._config["resolution_x"]}',
                f'-ResY={self._config["resolution_y"]}'
            ]
        else:
            server_command = [
                f'{os.environ["CARLA_ROOT"]}/CarlaUE4.sh',
                '-RenderOffScreen -nosound'
            ]

        server_command += [
            f'--carla-rpc-port={self._server_port}',
            f'-quality-level={self._config["quality_level"]}',
            f'-ini:[/Script/Engine.RendererSettings]:r.GraphicsAdapter={self._config["render_gpu"]}'
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
        '''Connect the client to the CARLA server.'''
        for i in range(self._config['retries_on_error']):
            try:
                logger.debug(f'Connecting to the server on port {self._server_port}...')
                
                self.client = carla.Client(self._config['host'], self._server_port)
                
                self.client.set_timeout(self._config['timeout'])

                logger.debug('Connected to the server.')
                logger.debug('Creating the World Manager...')

                self._world_manager = WorldManager(self._config, self.client, self._server_port)

                logger.debug('World Manager created.')

                return

            except Exception as e:
                logger.warning(f'Waiting for the server to be ready: {e}, attempt {i + 1} of '
                               f'{self._config["retries_on_error"]}.')
                
                time.sleep(3.0)

        raise Exception('Cannot connect to the CARLA server. Good bye!')
    
    def _setup_keyboard_listener(self):
        '''Set up keyboard listener to pause/resume the simulation.'''
        self._pause = threading.Event()
        self._pause.set()

        self._listener = keyboard.Listener(on_press=self._on_key_press)
        self._listener.daemon = True
        self._listener.start()
    
    def _on_key_press(self, key):
        '''
        Handle key press events.

        Args:
            key: the key that was pressed.
        '''
        try:
            if key.char == 'p':
                if self._pause.is_set():
                    self._pause.clear()

                    logger.warning('Simulation paused.')
                else:
                    self._pause.set()

                    logger.info('Simulation resumed.')
        
        except AttributeError:
            pass
    
    def load_map(self, map_name: str):
        '''
        Load a map in CARLA.

        Args:
            map_name: name of the map to load.
        '''
        self._pause.wait()

        return self._world_manager.load_map(map_name)
    
    def spawn_vehicle(self):
        '''Spawn a vehicle.'''
        self._pause.wait()

        return self._world_manager.spawn_vehicle()

    def move_vehicle(self):
        '''Move the vehicle.'''
        self._pause.wait()

        return self._world_manager.move_vehicle()
    
    def start_scene(self, seed: int = None):
        '''
        Start the scene.
        
        Args:
            seed: random seed for the scene.
        '''
        self._pause.wait()

        return self._world_manager.start_scene(seed)

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
        self._pause.wait()

        return self._world_manager.tick(path, scene, frame, render, save)
    
    def stop_scene(self):
        '''Stop the scene.'''
        self._pause.wait()

        return self._world_manager.stop_scene()
    
    def destroy_vehicle(self):
        '''Destroy the vehicle.'''
        self._pause.wait()
        
        return self._world_manager.destroy_vehicle()
    
    def package_data(self):
        '''Package scene information and data and return it.'''
        return self._world_manager.package_data()
    
