# Academic Software License: Copyright © 2026 Goodarz Mehr.

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

VOX2EGO_T = [0.0, 0.0, 0.02]
VOX2EGO_R = [1.0, 0.0, 0.0, 0.0]

CAM2LI_T = CAM2EGO_T - LI2EGO_T * np.ones((6, 3))
CAM2LI_R = CAM2EGO_R

RAD2LI_T = RAD2EGO_T - LI2EGO_T * np.ones((4, 3))
RAD2LI_R = RAD2EGO_R

VOX2LI_T = VOX2EGO_T - LI2EGO_T * np.ones((3, ))
VOX2LI_R = VOX2EGO_R

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

    metadata['voxel_detector_properties'] = {
        'range': config['voxel_detector_range'],
        'voxel_size': config['voxel_size'],
        'upper_limit': config['voxel_detector_upper_limit'],
        'lower_limit': config['voxel_detector_lower_limit']
    }
    
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
        
    metadata['VOXEL-GRID'] = {
        'sensor2lidar_translation': VOX2LI_T.tolist(),
        'sensor2lidar_rotation': VOX2LI_R,
        'sensor2ego_translation': VOX2EGO_T,
        'sensor2ego_rotation': VOX2EGO_R
    }
    
    return metadata

def _create_directory_structure(args, config: dict):
    '''
    Create the required directory structure for saving data.
    
    Args:
        args: command line arguments.
        config: configuration dictionary.
    '''
    # Camera directories.
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
    
    # Lidar directories.
    if config['use_lidar']:
        os.makedirs(f'{args.path}/simbev/sweeps/LIDAR', exist_ok=True)
    if config['use_semantic_lidar']:
        os.makedirs(f'{args.path}/simbev/sweeps/SEG-LIDAR', exist_ok=True)
    
    # Radar directories.
    if config['use_radar']:
        for name in RAD_NAME:
            os.makedirs(f'{args.path}/simbev/sweeps/{name}', exist_ok=True)
    
    # IMU and GNSS directories.
    if config['use_gnss']:
        os.makedirs(f'{args.path}/simbev/sweeps/GNSS', exist_ok=True)
    if config['use_imu']:
        os.makedirs(f'{args.path}/simbev/sweeps/IMU', exist_ok=True)
    
    # Voxel detector directory.
    if config['use_voxel_detector']:
        os.makedirs(f'{args.path}/simbev/sweeps/VOXEL-GRID', exist_ok=True)
    
    # Ground truth directories.
    os.makedirs(f'{args.path}/simbev/ground-truth/seg', exist_ok=True)
    os.makedirs(f'{args.path}/simbev/ground-truth/det', exist_ok=True)
    os.makedirs(f'{args.path}/simbev/ground-truth/seg_viz', exist_ok=True)
    os.makedirs(f'{args.path}/simbev/ground-truth/hd_map', exist_ok=True)
    
    # Dataset info directories.
    os.makedirs(f'{args.path}/simbev/infos', exist_ok=True)
    os.makedirs(f'{args.path}/simbev/logs', exist_ok=True)
    os.makedirs(f'{args.path}/simbev/configs', exist_ok=True)

def _initialize_carla_core(config: dict, logger: logging.Logger) -> CarlaCore:
    '''
    Initialize CarlaCore and perform initial setup.
    
    Args:
        config: configuration dictionary.
        logger: logger instance.
        
    Returns:
        Initialized CarlaCore instance.
    '''
    logger.info('Setting things up...')
    
    core = CarlaCore(config)
    
    # Load Town01 once to get around a bug in CARLA where the pedestrian
    # navigation information for the wrong map is loaded.
    core.load_map('Town01')
    core.spawn_vehicle()
    core.start_scene()
    core.tick()
    core.stop_scene()
    core.destroy_vehicle()
    core.shut_down_traffic_manager()
    
    return core

def _run_warmup(core: CarlaCore, config: dict):
    '''
    Run the simulation warmup phase.
    
    Args:
        core: CarlaCore instance.
        config: configuration dictionary.
    '''
    pbar = tqdm(
        range(round(config['warmup_duration'] / config['timestep'])),
        desc='Warming up',
        ncols=120,
        colour='#FF0000'
    )
    
    for _ in pbar:
        core.tick()

def _run_data_collection(
        args,
        core: CarlaCore,
        config: dict,
        logger: logging.Logger,
        scene_counter: int,
        scene_duration: int,
        augment: bool = False,
        replay: bool = False
    ):
    '''
    Run the data collection phase.
    
    Args:
        args: command line arguments.
        core: CarlaCore instance.
        config: configuration dictionary.
        logger: logger instance.
        scene_counter: scene number.
        scene_duration: duration of the scene in seconds.
        augment: whether the dataset is being augmented.
        replay: whether the scene is being replayed.
    '''
    pbar = tqdm(
        range(round(scene_duration / config['timestep'])),
        desc=f'Scene {scene_counter:04d}',
        ncols=120,
        colour='#00FF00'
    )
    
    for j in pbar:
        if not replay:
            should_terminate = (
                core.get_world_manager().get_terminate_scene() and
                j % round(1.0 / config['timestep']) == 0
            )
        else:
            should_terminate = False
        
        if not should_terminate:
            core.tick(args.path, scene_counter, j, args.render, args.save, augment, replay)
        else:
            logger.warning('Termination conditions met. Ending scene early.')

            core.set_scene_info({'terminated_early': True})
            
            return

def _generate_scene(
        args,
        core: CarlaCore,
        config: dict,
        logger: logging.Logger,
        scene_counter: int,
        data: dict,
        seed: int = None,
        augment: bool = False
    ):
    '''
    Generate a single scene.
    
    Args:
        args: command line arguments.
        core: CarlaCore instance.
        config: configuration dictionary.
        metadata: dataset metadata.
        logger: logger instance.
        scene_counter: scene number.
        data: scene data information dictionary.
        seed: random seed.
        augment: whether the dataset is being augmented.
    '''
    # Randomly select scene duration.
    scene_duration = max(round(np.random.uniform(config['min_scene_duration'], config['max_scene_duration'])), 1)
    
    core.set_scene_duration(scene_duration)
    
    logger.info(f'Scene {scene_counter:04d} duration: {scene_duration} seconds.')
    
    core.start_scene(seed)

    # Run the simulation for a few seconds so everything gets going.
    _run_warmup(core, config)
    
    # Start logging the scene.
    if args.save and not augment:
        core.client.start_recorder(f'{args.path}/simbev/logs/SimBEV-scene-{scene_counter:04d}.log', True)
    
    # Tick 3 times so when replayed, the recorder data matches the saved data.
    for _ in range(3):
        core.tick()
    
    # Start data collection.
    _run_data_collection(args, core, config, logger, scene_counter, scene_duration, augment)

    core.wait_for_saves()
    
    if args.save:
        for _ in range(2):
            core.tick()
        
        # Stop logging the scene.
        if not augment:
            core.client.stop_recorder()
        
        # Get the scene data information and save it.
        scene_data = core.package_data()
        
        scene_data['scene_info']['log'] = f'{args.path}/simbev/logs/SimBEV-scene-{scene_counter:04d}.log'
        scene_data['scene_info']['config'] = f'{args.path}/simbev/configs/SimBEV-scene-{scene_counter:04d}.yaml'
        
        data[f'scene_{scene_counter:04d}'] = copy.deepcopy(scene_data)
    
    core.stop_scene()

def _create_scenes_for_map(
        args,
        core: CarlaCore,
        config: dict,
        metadata: dict,
        logger: logging.Logger,
        map_name: str,
        split: str,
        scene_counter: int,
        data: dict,
        num_scenes: int
    ) -> int:
    '''
    Create multiple scenes for a specific map.
    
    Args:
        args: command line arguments.
        core: CarlaCore instance.
        config: configuration dictionary.
        metadata: dataset metadata.
        logger: logger instance.
        map_name: map name.
        split: data split (train/val/test).
        scene_counter: current scene counter.
        data: existing scene data information dictionary.
        num_scenes: number of scenes to create.
        
    Returns:
        scene_counter: updated scene counter.
    '''
    # Set the random seed if configured.
    if config['use_scene_number_for_random_seed']:
        seed = scene_counter + config['random_seed_offset']
        
        random.seed(seed)
        np.random.seed(seed)
    else:
        seed = None
    
    core.connect_client()
    core.load_map(map_name)

    if seed is not None:
        core.set_carla_seed(seed)

    core.spawn_vehicle()
    
    for i in range(num_scenes):
        logger.info(f'Creating scene {scene_counter:04d} in {map_name} for the {split} set...')
        
        if i > 0:
            # Update the random seed if configured.
            if config['use_scene_number_for_random_seed']:
                seed = scene_counter + config['random_seed_offset']
                
                random.seed(seed)
                np.random.seed(seed)

                core.set_carla_seed(seed)
                core.spawn_vehicle()
            else:
                seed = None
        
                core.move_vehicle()
        
        # Generate and save the scene.
        _generate_scene(args, core, config, logger, scene_counter, data, seed)
        
        # Save the updated data information.
        if args.save:
            info = {'metadata': metadata, 'data': data}
            
            with open(f'{args.path}/simbev/infos/simbev_infos_{split}.json', 'w') as f:
                json.dump(info, f, indent=4)
            
            with open(f'{args.path}/simbev/configs/SimBEV-scene-{scene_counter:04d}.yaml', 'w') as f:
                yaml.dump(config, f)
        
        scene_counter += 1

        if config['use_scene_number_for_random_seed']:
            core.destroy_vehicle()
    
    if not config['use_scene_number_for_random_seed']:
        core.destroy_vehicle()
    
    core.shut_down_traffic_manager()
    
    return scene_counter

def collect_data_create_mode(args, core: CarlaCore, config: dict, metadata: dict, logger: logging.Logger):
    '''
    Create new scenes and collect data.
    
    Args:
        args: command line arguments.
        core: CarlaCore instance.
        config: configuration dictionary.
        metadata: dataset metadata.
        logger: logger instance.
    '''
    scene_counter = 0
    
    # Check to see how many scenes have been created already. Then, create the
    # remaining scenes.
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
            
            # For each data split and map, check how many scenes have been
            # created already.
            for key in data.keys():
                map_name = data[key]['scene_info']['map']
                
                if map_name in config[f'{split}_scene_config']:
                    config[f'{split}_scene_config'][map_name] -= 1
        
        # Create the scenes for each map.
        if config[f'{split}_scene_config'] is not None:
            for map_name in config[f'{split}_scene_config']:
                if config[f'{split}_scene_config'][map_name] > 0:
                    scene_counter = _create_scenes_for_map(
                        args,
                        core,
                        config,
                        metadata,
                        logger,
                        map_name,
                        split,
                        scene_counter,
                        data,
                        config[f'{split}_scene_config'][map_name]
                    )

def collect_data_replace_mode(args, core: CarlaCore, config: dict, metadata: dict, logger: logging.Logger):
    '''
    Replace specified scenes with newly generated ones.
    
    Args:
        args: command line arguments.
        core: CarlaCore instance.
        config: configuration dictionary.
        metadata: dataset metadata.
        logger: logger instance.
    '''
    first_setup = True

    for split in ['train', 'val', 'test']:
        if not os.path.exists(f'{args.path}/simbev/infos/simbev_infos_{split}.json'):
            continue
        
        with open(f'{args.path}/simbev/infos/simbev_infos_{split}.json', 'r') as f:
            infos = json.load(f)
        
        data = infos['data']
        
        # Replace the specified scenes.
        for scene_counter in config['replacement_scene_config']:            
            scene_key = f'scene_{scene_counter:04d}'
            
            if scene_key not in data.keys():
                logger.warning(f'Scene {scene_counter:04d} not found in the {split} set. Skipping...')
                
                continue
            
            # Remove the files of the specified scene.
            if os.path.exists(f'{args.path}/simbev'):
                stale_scene_id = f'{scene_counter:04d}'
                
                logger.debug(f'Removing the files of scene {stale_scene_id}...')
                
                os.system(f'find "{args.path}/simbev" | grep "scene-{stale_scene_id}" | xargs rm -f')
                
                logger.debug(f'Removed the files of scene {stale_scene_id}.')
            
            map_name = data[scene_key]['scene_info']['map']

            if config['use_scene_number_for_random_seed']:
                seed = scene_counter + config['random_seed_offset']

                random.seed(seed)
                np.random.seed(seed)
            else:
                seed = None
            
            if first_setup:
                core.connect_client()
                core.load_map(map_name)
                
                first_setup = False
            
            # Load a new map if necessary.
            if map_name != core.get_world_manager().get_map_name():
                core.shut_down_traffic_manager()
                core.connect_client()
                core.load_map(map_name)
            
            logger.info(f'Replacing scene {scene_counter:04d} in {map_name} for the {split} set...')

            if seed is not None:
                core.set_carla_seed(seed)

            core.spawn_vehicle()
            
            # Generate and save the replacement scene.
            _generate_scene(args, core, config, logger, scene_counter, data, seed)
            
            # Save the updated data information.
            info = {'metadata': metadata, 'data': data}
            
            with open(f'{args.path}/simbev/infos/simbev_infos_{split}.json', 'w') as f:
                json.dump(info, f, indent=4)
            
            with open(f'{args.path}/simbev/configs/SimBEV-scene-{scene_counter:04d}.yaml', 'w') as f:
                yaml.dump(config, f)
            
            core.destroy_vehicle()

def collect_data_augment_mode(args, core: CarlaCore, config: dict, metadata: dict, logger: logging.Logger):
    '''
    Augment specified scenes with additional sensor data.
    
    Args:
        args: command line arguments.
        core: CarlaCore instance.
        config: configuration dictionary.
        metadata: dataset metadata.
        logger: logger instance.
    '''
    first_setup = True

    for split in ['train', 'val', 'test']:
        info_path = f'{args.path}/simbev/infos/simbev_infos_{split}.json'
        
        if not os.path.exists(info_path):
            continue
        
        with open(info_path, 'r') as f:
            infos = json.load(f)
        
        for i in range(100):
            info_path = f'{args.path}/simbev/infos/simbev_infos_{split}_original_{i}.json'

            if not os.path.exists(info_path):
                with open(info_path, 'w') as f:
                    json.dump(infos, f, indent=4)
                
                break
        
        data = infos['data']
        
        # Augment the specified scenes.
        for scene_counter in config['augmentation_scene_config']:            
            scene_key = f'scene_{scene_counter:04d}'
            
            if scene_key not in data.keys():
                logger.debug(f'Scene {scene_counter:04d} not found in the {split} set. Skipping...')
                
                continue
            
            # Load the original scene configuration.
            with open(data[scene_key]['scene_info']['config']) as f:
                original_config = yaml.load(f, Loader=yaml.FullLoader)
            
            if not original_config['use_scene_number_for_random_seed']:
                logger.warning(f'Cannot augment scene {scene_counter:04d} because it was created without using the '
                               f'scene number as random seed.')
                
                continue

            original_data = copy.deepcopy(data)
            
            map_name = data[scene_key]['scene_info']['map']

            seed = scene_counter + config['random_seed_offset']

            random.seed(seed)
            np.random.seed(seed)
            
            if first_setup:
                core.connect_client()
                core.load_map(map_name)
                
                first_setup = False
            
            # Load a new map if necessary.
            if map_name != core.get_world_manager().get_map_name():
                core.shut_down_traffic_manager()
                core.connect_client()
                core.load_map(map_name)
            
            logger.info(f'Augmenting scene {scene_counter:04d} in {map_name} for the {split} set...')

            core.set_carla_seed(seed)
            core.spawn_vehicle()
            
            # Generate and save the replacement scene.
            _generate_scene(args, core, config, logger, scene_counter, data, seed, augment=True)

            try:
                for i, frame in enumerate(original_data[scene_key]['scene_data']):
                    frame.update(data[scene_key]['scene_data'][i])
            
            except IndexError:
                logger.warning(
                    f'Frame count mismatch when augmenting scene {scene_counter:04d}. The augmented data may be '
                    'inconsistent with the original data.'
                )
            
            # Save the updated data information.
            info = {'metadata': metadata, 'data': original_data}
            
            with open(f'{args.path}/simbev/infos/simbev_infos_{split}.json', 'w') as f:
                json.dump(info, f, indent=4)
            
            for i in range(100):
                config_path = f'{args.path}/simbev/configs/SimBEV-scene-{scene_counter:04d}-augment-{i}.yaml'
                
                if not os.path.exists(config_path):
                    with open(config_path, 'w') as f:
                        yaml.dump(config, f)
                        
                    break
            
            core.destroy_vehicle()

            data = copy.deepcopy(original_data)

def collect_data_replay_mode(args, core: CarlaCore, config: dict, metadata: dict, logger: logging.Logger):
    '''
    Replay specified scenes and potentially augment them with additional
        sensor data.
    
    Args:
        args: command line arguments.
        core: CarlaCore instance.
        config: configuration dictionary.
        metadata: dataset metadata.
        logger: logger instance.
    '''
    first_setup = True

    for split in ['train', 'val', 'test']:
        info_path = f'{args.path}/simbev/infos/simbev_infos_{split}.json'
        
        if not os.path.exists(info_path):
            continue
        
        with open(info_path, 'r') as f:
            infos = json.load(f)
        
        for i in range(100):
            info_path = f'{args.path}/simbev/infos/simbev_infos_{split}_original_{i}.json'

            if not os.path.exists(info_path):
                with open(info_path, 'w') as f:
                    json.dump(infos, f, indent=4)
                
                break
        
        data = infos['data']
        
        # Replay the specified scenes.
        for scene_counter in config['replay_scene_config']:            
            scene_key = f'scene_{scene_counter:04d}'
            
            if scene_key not in data.keys():
                logger.debug(f'Scene {scene_counter:04d} not found in the {split} set. Skipping...')
                
                continue

            original_data = copy.deepcopy(data)

            scene_info = data[scene_key]['scene_info']
            
            map_name = scene_info['map']
            
            if first_setup:
                core.connect_client()
                core.load_map(map_name)
                
                first_setup = False
            
            # Load a new map if necessary.
            if map_name != core.get_world_manager().get_map_name():
                core.shut_down_traffic_manager()
                core.connect_client()
                core.load_map(map_name)
            
            logger.info(f'Replaying scene {scene_counter:04d} in {map_name} for the {split} set...')

            log_path = scene_info['log']

            core.client.replay_file(log_path, 0, 0, 0)

            core.find_vehicle()

            scene_duration = len(data[scene_key]['scene_data']) * config['timestep']

            core.set_scene_duration(scene_duration)

            if 'final_weather_parameters' in scene_info:
                core.configure_replay_weather(
                    scene_info['initial_weather_parameters'],
                    scene_info['final_weather_parameters']
                )
            else:
                core.configure_replay_weather(scene_info['initial_weather_parameters'])

            # Replay the scene.
            _run_data_collection(args, core, config, logger, scene_counter, scene_duration, augment=True, replay=True)

            core.client.stop_replayer(keep_actors=False)

            core.wait_for_saves()

            core.tick(augment=True, replay=True)

            scene_data = core.package_data()
            
            data[f'scene_{scene_counter:04d}'] = copy.deepcopy(scene_data)

            try:
                for i, frame in enumerate(original_data[scene_key]['scene_data']):
                    frame.update(data[scene_key]['scene_data'][i])
            
            except IndexError:
                logger.warning(
                    f'Frame count mismatch when augmenting scene {scene_counter:04d}. The augmented data may be '
                    'inconsistent with the original data.'
                )
            
            # Save the updated data information.
            info = {'metadata': metadata, 'data': original_data}
            
            with open(f'{args.path}/simbev/infos/simbev_infos_{split}.json', 'w') as f:
                json.dump(info, f, indent=4)
            
            for i in range(100):
                config_path = f'{args.path}/simbev/configs/SimBEV-scene-{scene_counter:04d}-augment-{i}.yaml'
                
                if not os.path.exists(config_path):
                    with open(config_path, 'w') as f:
                        yaml.dump(config, f)
                        
                    break
            
            core.destroy_replay_actors()

            data = copy.deepcopy(original_data)

def collect_data(args, mode: str, core: CarlaCore, config: dict, metadata: dict, logger: logging.Logger):
    '''
    Data collection dispatcher.
    
    Args:
        args: command line arguments.
        mode: data collection mode ('create', 'replace', 'augment').
        core: CarlaCore instance.
        config: configuration dictionary.
        metadata: dataset metadata.
        logger: logger instance.
    '''
    mode_handlers = {
        'create': collect_data_create_mode,
        'replace': collect_data_replace_mode,
        'augment': collect_data_augment_mode,
        'replay': collect_data_replay_mode
    }
    
    if mode not in mode_handlers:
        logger.error(f'Unknown mode: {mode}')
        
        return
    
    if mode in ['replace', 'augment'] and not args.save:
        logger.error(f'The {mode} mode cannot be used with the --no-save flag.')
        
        return
    
    # Call the appropriate mode handler.
    mode_handlers[mode](args, core, config, metadata, logger)

def main(logger: logging.Logger):
    config = parse_config(args)
    
    metadata = generate_metadata(config)
    
    try:
        if args.save:
            _create_directory_structure(args, config)
        
        # Initialize CarlaCore.
        core = _initialize_carla_core(config, logger)
        
        # Run data collection with the specified mode.
        collect_data(args, config['mode'], core, config, metadata, logger)
        
        logger.warning('Killing all servers...')
        
        kill_all_servers()
    
    except Exception:
        logger.critical(traceback.format_exc())
        
        logger.warning('Killing all servers...')
        
        kill_all_servers()
        
        time.sleep(3.0)

def entry():
    try:
        logger = setup_logger(log_level=logging.INFO, log_dir=f'{args.path}/simbev/console_logs', save=args.save)
        
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