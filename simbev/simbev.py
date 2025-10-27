# Academic Software License: Copyright © 2025 Goodarz Mehr.

import os
import yaml
import time
import json
import copy
import random
import argparse
import traceback

import logging
import logging.handlers

import numpy as np

from tqdm import tqdm
from datetime import datetime

try:
    from .utils import TqdmLoggingHandler, kill_all_servers
    
    from .carla_core import CarlaCore

except ImportError:
    from utils import TqdmLoggingHandler, kill_all_servers
    
    from carla_core import CarlaCore


CAM2EGO_T = [
    [0.4, 0.4, 1.6],
    [0.6, 0.0, 1.6],
    [0.4, -0.4, 1.6],
    [0.0, 0.4, 1.6],
    [-1.0, 0.0, 1.6],
    [0.0, -0.4, 1.6]
]
CAM2EGO_R = [
    [0.6743797, -0.6743797, 0.2126311, -0.2126311],
    [0.5, -0.5, 0.5, -0.5],
    [0.2126311, -0.2126311, 0.6743797, -0.6743797],
    [0.6963642, -0.6963642, -0.1227878, 0.1227878],
    [0.5, -0.5, -0.5, 0.5],
    [0.1227878, -0.1227878, -0.6963642, 0.6963642]
]

LI2EGO_T = [0.0, 0.0, 1.8]
LI2EGO_R = [1.0, 0.0, 0.0, 0.0]

RAD2EGO_T = [
    [0.0, 1.0, 0.6],
    [2.4, 0.0, 0.6],
    [0.0, -1.0, 0.6],
    [-2.4, 0.0, 0.6]
]
RAD2EGO_R = [
    [0.7071067, 0.0, 0.0, 0.7071067],
    [1.0, 0.0, 0.0, 0.0],
    [0.7071067, 0.0, 0.0, -0.7071067],
    [0.0, 0.0, 0.0, 1.0]
]

CAM2LI_T = CAM2EGO_T - LI2EGO_T * np.ones((6, 3))
CAM2LI_R = CAM2EGO_R

RAD2LI_T = RAD2EGO_T - LI2EGO_T * np.ones((4, 3))
RAD2LI_R = RAD2EGO_R

CAM_I = [
    [953.4029, 0.0, 800.0],
    [0.0, 953.4029, 450.0],
    [0.0, 0.0, 1.0]
]

CAM_NAME = [
    'CAM_FRONT_LEFT',
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT',
    'CAM_BACK',
    'CAM_BACK_RIGHT'
]

RAD_NAME = ['RAD_LEFT', 'RAD_FRONT', 'RAD_RIGHT', 'RAD_BACK']


argparser = argparse.ArgumentParser(description='SimBEV is a CARLA-based driving data generation tool.')
    
argparser.add_argument('config', help='configuration file')
argparser.add_argument(
    '--path',
    default='/dataset',
    help='path for saving the dataset (default: /dataset)')
argparser.add_argument(
    '--render',
    action='store_true',
    help='render sensor data')
argparser.add_argument(
    '--save',
    action='store_true',
    help='save sensor data (used by default)')
argparser.add_argument(
    '--no-save',
    dest='save',
    action='store_false',
    help='do not save sensor data')

argparser.set_defaults(save=True)

args = argparser.parse_args()


def setup_logger(name=None, log_level=logging.INFO, log_dir: str = 'logs', save: bool = True) -> logging.Logger:
    '''
    Set up a logger with both console and file handlers.
    
    Args:
        name: logger name (if None, uses root logger)
        log_level: logging level (default: INFO)
        log_dir: directory to store log files
        
    Returns:
        logger: configured logger instance
    '''
    if save:
        os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    
    logger.setLevel(log_level)
    
    # Avoid adding handlers multiple times.
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create formatter.
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create the console handler.
    console_handler = TqdmLoggingHandler()
    
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)

    if save:
        # Create the file handler.
        log_filename = os.path.join(log_dir, f'SimBEV_{datetime.now().strftime("%Y%m%d%H%M%S")}.log')

        file_handler = logging.handlers.RotatingFileHandler(log_filename, maxBytes=100*1024*1024, backupCount=5)

        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        
        # Create the error file handler.
        error_filename = os.path.join(log_dir, f'SimBEV_Errors_{datetime.now().strftime("%Y%m%d%H%M%S")}.log')

        error_handler = logging.handlers.RotatingFileHandler(error_filename, maxBytes=100*1024*1024, backupCount=5)

        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        
        logger.addHandler(error_handler)
    
    return logger

def parse_config(args) -> dict:
    '''
    Parse the configuration file.

    Args:
        args: command line arguments.

    Returns:
        config: configuration dictionary.
    '''
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    for camera_type in ['rgb', 'semantic', 'instance', 'depth', 'flow']:
        config[f'{camera_type}_camera_properties']['fov'] = config['camera_fov']
    
    return config

def generate_metadata(config: dict) -> dict:
    '''
    Generate dataset metadata from sensor transformations.

    Args:
        config: configuration dictionary.
    
    Returns:
        metadata: dataset metadata.
    '''
    metadata = {}

    cx = config['camera_width'] / 2.0
    cy = config['camera_height'] / 2.0

    f = config['camera_width'] / (2.0 * np.tan(float(config['camera_fov']) / 360.0 * np.pi))

    CAM_I[0][0] = f
    CAM_I[1][1] = f
    CAM_I[0][2] = cx
    CAM_I[1][2] = cy
    
    metadata['camera_intrinsics'] = CAM_I

    metadata['LIDAR'] = {
        'sensor2lidar_translation': [0.0, 0.0, 0.0],
        'sensor2lidar_rotation': [1.0, 0.0, 0.0, 0.0],
        'sensor2ego_translation': LI2EGO_T,
        'sensor2ego_rotation': LI2EGO_R
    }
    
    for i in range(6):
        metadata[CAM_NAME[i]] = {
            'sensor2lidar_translation': CAM2LI_T[i].tolist(),
            'sensor2lidar_rotation': CAM2LI_R[i],
            'sensor2ego_translation': CAM2EGO_T[i],
            'sensor2ego_rotation': CAM2EGO_R[i]
        }
    
    for i in range(4):
        metadata[RAD_NAME[i]] = {
            'sensor2lidar_translation': RAD2LI_T[i].tolist(),
            'sensor2lidar_rotation': RAD2LI_R[i],
            'sensor2ego_translation': RAD2EGO_T[i],
            'sensor2ego_rotation': RAD2EGO_R[i]
        }
        
    return metadata

def main(logger: logging.Logger):
    config = parse_config(args)

    metadata = generate_metadata(config)

    try:
        if args.save:
            for name in CAM_NAME:
                if config['use_rgb_camera']:
                    os.makedirs(f'{args.path}/simbev/sweeps/RGB-{name}', exist_ok=True)
            
                if config['use_semantic_camera']:
                    os.makedirs(f'{args.path}/simbev/sweeps/SEG-{name}', exist_ok=True)
            
                if config['use_instance_camera']:
                    os.makedirs(f'{args.path}/simbev/sweeps/IST-{name}', exist_ok=True)
            
                if config['use_depth_camera']:
                    os.makedirs(f'{args.path}/simbev/sweeps/DPT-{name}', exist_ok=True)
            
                if config['use_flow_camera']:
                    os.makedirs(f'{args.path}/simbev/sweeps/FLW-{name}', exist_ok=True)
            
            if config['use_lidar']:
                os.makedirs(f'{args.path}/simbev/sweeps/LIDAR', exist_ok=True)
            
            if config['use_semantic_lidar']:
                os.makedirs(f'{args.path}/simbev/sweeps/SEG-LIDAR', exist_ok=True)
            
            if config['use_radar']:
                for name in RAD_NAME:
                    os.makedirs(f'{args.path}/simbev/sweeps/{name}', exist_ok=True)
            
            if config['use_gnss']:
                os.makedirs(f'{args.path}/simbev/sweeps/GNSS', exist_ok=True)
            
            if config['use_imu']:
                os.makedirs(f'{args.path}/simbev/sweeps/IMU', exist_ok=True)
            
            os.makedirs(f'{args.path}/simbev/ground-truth/seg', exist_ok=True)
            os.makedirs(f'{args.path}/simbev/ground-truth/det', exist_ok=True)
            os.makedirs(f'{args.path}/simbev/ground-truth/seg_viz', exist_ok=True)
            os.makedirs(f'{args.path}/simbev/ground-truth/hd_map', exist_ok=True)
            
            os.makedirs(f'{args.path}/simbev/infos', exist_ok=True)
            
            os.makedirs(f'{args.path}/simbev/logs', exist_ok=True)

            os.makedirs(f'{args.path}/simbev/configs', exist_ok=True)

        if config['mode'] == 'create':
            logger.info('Setting things up...')
            
            scene_counter = 0

            core = CarlaCore(config)

            # Load Town01 once to get around a bug in CARLA where the
            # pedestrian navigation information for the wrong town is loaded.
            core.load_map('Town01')

            core.spawn_vehicle()
            
            core.start_scene()

            core.tick()

            core.stop_scene()

            core.destroy_vehicle()
            
            # Check to see how many scenes have already been created. Then,
            # create the remaining scenes.
            for split in ['train', 'val', 'test']:
                if args.save and os.path.exists(f'{args.path}/simbev/infos/simbev_infos_{split}.json'):
                    with open(f'{args.path}/simbev/infos/simbev_infos_{split}.json', 'r') as f:
                        infos = json.load(f)
                    
                    scene_counter += len(infos['data'])

            if args.save:
                # Remove any stale files from the previous run.
                if os.path.exists(f'{args.path}/simbev'):
                    stale_scene_id = f'{scene_counter:04d}'

                    logger.debug(f'Removing stale files for scene {stale_scene_id}...')

                    os.system(f'find "{args.path}/simbev" | grep "scene-{stale_scene_id}" | xargs rm -f')
                    
                    logger.debug(f'Removed stale files for scene {stale_scene_id}.')

            for split in ['train', 'val', 'test']:
                data = {}

                if args.save and os.path.exists(f'{args.path}/simbev/infos/simbev_infos_{split}.json'):
                    with open(f'{args.path}/simbev/infos/simbev_infos_{split}.json', 'r') as f:
                        infos = json.load(f)

                    data = infos['data']

                    # For each split and each town, check how many scenes have
                    # already been created.
                    for key in data.keys():
                        town = data[key]['scene_info']['map']
                        
                        if town in config[f'{split}_scene_config']:
                            config[f'{split}_scene_config'][town] -= 1

                # Create the scenes for each town.
                if config[f'{split}_scene_config'] is not None:
                    for town in config[f'{split}_scene_config']:
                        if config[f'{split}_scene_config'][town] > 0:
                            if config['use_scene_number_for_random_seed']:
                                seed = scene_counter + config['random_seed_offset']

                                random.seed(seed)
                                np.random.seed(seed)
                            else:
                                seed = None
                            
                            core.connect_client()
                            
                            core.load_map(town)

                            core.spawn_vehicle()

                            for i in range(config[f'{split}_scene_config'][town]):
                                logger.info(f'Creating scene {scene_counter:04d} in {town} for the {split} set...')

                                if config['use_scene_number_for_random_seed']:
                                    seed = scene_counter + config['random_seed_offset']

                                    random.seed(seed)
                                    np.random.seed(seed)
                                else:
                                    seed = None
                                
                                # Randomly select the scene duration.
                                scene_duration = max(
                                    round(np.random.uniform(
                                        config['min_scene_duration'],
                                        config['max_scene_duration'])
                                    ),
                                    1
                                )

                                core.set_scene_duration(scene_duration)

                                logger.info(f'Scene {scene_counter:04d} duration: {scene_duration} seconds.')
                                
                                # Move the vehicle to a new location for all
                                # but the first scene.
                                if i > 0:
                                    core.move_vehicle()

                                core.start_scene(seed)

                                # Run the simulation for a few seconds so
                                # everything gets going.
                                pbar = tqdm(
                                    range(round(config['warmup_duration'] / config['timestep'])),
                                    desc='Warming up',
                                    ncols=120,
                                    colour='red'
                                )

                                for _ in pbar:
                                    core.tick()

                                # Start logging the scene.
                                if args.save:
                                    core.client.start_recorder(
                                        f'{args.path}/simbev/logs/SimBEV-scene-{scene_counter:04d}.log',
                                        True
                                    )

                                # Start data collection.
                                pbar = tqdm(
                                    range(round(scene_duration / config['timestep'])),
                                    desc=f'Scene {scene_counter:04d}',
                                    ncols=120,
                                    colour='green'
                                )

                                for j in pbar:
                                    if not (core.get_world_manager().get_terminate_scene() and \
                                            j % round(1.0 / config['timestep']) == 0):
                                        core.tick(args.path, scene_counter, j, args.render, args.save)
                                    else:
                                        logger.warning('Termination conditions met. Ending scene early.')

                                        core.set_scene_info({'terminated_early': True})

                                        break
                                
                                if args.save:
                                    # Stop logging the scene.
                                    core.client.stop_recorder()

                                    # Get the scene data and save it.
                                    scene_data = core.package_data()

                                    scene_data['scene_info']['log'] = f'{args.path}/simbev/logs/' \
                                        f'SimBEV-scene-{scene_counter:04d}.log'
                                    scene_data['scene_info']['config'] = f'{args.path}/simbev/configs/' \
                                        f'SimBEV-scene-{scene_counter:04d}.yaml'

                                    data[f'scene_{scene_counter:04d}'] = copy.deepcopy(scene_data)
                                    
                                    info = {'metadata': metadata, 'data': data}

                                    with open(f'{args.path}/simbev/infos/simbev_infos_{split}.json', 'w') as f:
                                        json.dump(info, f, indent=4)
                                    
                                    with open(
                                        f'{args.path}/simbev/configs/SimBEV-scene-{scene_counter:04d}.yaml',
                                        'w'
                                    ) as f:
                                        yaml.dump(config, f)

                                core.stop_scene()
                                
                                scene_counter += 1

                            core.destroy_vehicle()
        
        elif config['mode'] == 'replace' and args.save:
            logger.info('Setting things up...')

            core = CarlaCore(config)

            # Load Town01 once to get around a bug in CARLA where the
            # pedestrian navigation information for the wrong town is loaded.
            core.load_map('Town01')

            core.spawn_vehicle()
            
            core.start_scene()

            core.tick()

            core.stop_scene()

            core.destroy_vehicle()
            
            for split in ['train', 'val', 'test']:
                if os.path.exists(f'{args.path}/simbev/infos/simbev_infos_{split}.json'):
                    with open(f'{args.path}/simbev/infos/simbev_infos_{split}.json', 'r') as f:
                        infos = json.load(f)

                    data = infos['data']

                    # Replace the specified scenes.
                    for scene_counter in config['replacement_scene_config']:
                        # Remove the files of the specified scene.
                        if os.path.exists(f'{args.path}/simbev'):
                            stale_scene_id = f'{scene_counter:04d}'

                            logger.debug(f'Removing the files of scene {stale_scene_id}...')

                            os.system(f'find "{args.path}/simbev" | grep "scene-{stale_scene_id}" | xargs rm -f')

                            logger.debug(f'Removed the files of scene {stale_scene_id}.')

                        if f'scene_{scene_counter:04d}' in data.keys():
                            town = data[f'scene_{scene_counter:04d}']['scene_info']['map']

                            if config['use_scene_number_for_random_seed']:
                                seed = scene_counter + config['random_seed_offset']

                                random.seed(seed)
                                np.random.seed(seed)
                            else:
                                seed = None

                            # Load a new map if necessary.
                            if town != core.get_world_manager().get_map_name():
                                core.connect_client()
                                
                                core.load_map(town)

                            logger.info(f'Replacing scene {scene_counter:04d} in {town} for the {split} set...')

                            # Randomly select the scene duration.
                            scene_duration = max(
                                round(np.random.uniform(config['min_scene_duration'], config['max_scene_duration'])),
                                1
                            )

                            core.set_scene_duration(scene_duration)

                            logger.info(f'Scene {scene_counter:04d} duration: {scene_duration} seconds.')
                            
                            core.spawn_vehicle()
                            
                            core.start_scene(seed)

                            # Run the simulation for a few seconds so
                            # everything gets going.
                            pbar = tqdm(
                                range(round(config['warmup_duration'] / config['timestep'])),
                                desc='Warming up',
                                ncols=120,
                                colour='red'
                            )
                            
                            for _ in pbar:
                                core.tick()

                            # Start logging the scene.
                            core.client.start_recorder(
                                f'{args.path}/simbev/logs/SimBEV-scene-{scene_counter:04d}.log',
                                True
                            )

                            # Start data collection.
                            pbar = tqdm(
                                range(round(scene_duration / config['timestep'])),
                                desc=f'Scene {scene_counter:04d}',
                                ncols=120,
                                colour='green'
                            )

                            for j in pbar:
                                if not (core.get_world_manager().get_terminate_scene() and \
                                        j % round(1.0 / config['timestep']) == 0):
                                    core.tick(args.path, scene_counter, j, args.render, args.save)
                                else:
                                    logger.warning('Termination conditions met. Ending scene early.')

                                    core.set_scene_info({'terminated_early': True})
                                    
                                    break

                            # Stop logging the scene.
                            core.client.stop_recorder()

                            # Get the scene data and save it.
                            scene_data = core.package_data()

                            scene_data['scene_info']['log'] = f'{args.path}/simbev/logs/' \
                                f'SimBEV-scene-{scene_counter:04d}.log'
                            scene_data['scene_info']['config'] = f'{args.path}/simbev/configs/' \
                                f'SimBEV-scene-{scene_counter:04d}.yaml'

                            data[f'scene_{scene_counter:04d}'] = copy.deepcopy(scene_data)
                            
                            info = {'metadata': metadata, 'data': data}

                            with open(f'{args.path}/simbev/infos/simbev_infos_{split}.json', 'w') as f:
                                json.dump(info, f, indent=4)

                            with open(f'{args.path}/simbev/configs/SimBEV-scene-{scene_counter:04d}.yaml', 'w') as f:
                                yaml.dump(config, f)
                            
                            core.stop_scene()

                            core.destroy_vehicle()             
                        else:
                            logger.warning(f'Scene {scene_counter:04d} not found in the {split} set. Skipping...')
        
        logger.warning('Killing all servers...')
        
        kill_all_servers()

    except Exception:
        logger.critical(traceback.format_exc())

        logger.warning('Killing all servers...')

        kill_all_servers()

        time.sleep(3.0)

def entry():
    try:
        logger = setup_logger(log_level=logging.DEBUG, log_dir=f'{args.path}/simbev/console_logs', save=args.save)
        
        main(logger)
    
    except KeyboardInterrupt:
        logger.warning('The process was interrupted by the user.')
        logger.warning('Killing all servers...')
        
        kill_all_servers()

        time.sleep(3.0)
    
    finally:
        logger.info('Done.')


if __name__ == '__main__':
    entry()