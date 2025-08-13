# Academic Software License: Copyright © 2025 Goodarz Mehr.

import os
import yaml
import time
import json
import copy
import argparse
import traceback

import numpy as np

from carla_core import CarlaCore, kill_all_servers

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

def parse_config(args):
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

def generate_metadata(config):
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

def main():
    args.config = parse_config(args)

    metadata = generate_metadata(args.config)

    try:
        if args.save:
            for name in CAM_NAME:
                if args.config['use_rgb_camera']:
                    os.makedirs(f'{args.path}/simbev/sweeps/RGB-{name}', exist_ok=True)
            
                if args.config['use_semantic_camera']:
                    os.makedirs(f'{args.path}/simbev/sweeps/SEG-{name}', exist_ok=True)
            
                if args.config['use_instance_camera']:
                    os.makedirs(f'{args.path}/simbev/sweeps/IST-{name}', exist_ok=True)
            
                if args.config['use_depth_camera']:
                    os.makedirs(f'{args.path}/simbev/sweeps/DPT-{name}', exist_ok=True)
            
                if args.config['use_flow_camera']:
                    os.makedirs(f'{args.path}/simbev/sweeps/FLW-{name}', exist_ok=True)
            
            if args.config['use_lidar']:
                os.makedirs(f'{args.path}/simbev/sweeps/LIDAR', exist_ok=True)
            
            if args.config['use_semantic_lidar']:
                os.makedirs(f'{args.path}/simbev/sweeps/SEG-LIDAR', exist_ok=True)
            
            if args.config['use_radar']:
                for name in RAD_NAME:
                    os.makedirs(f'{args.path}/simbev/sweeps/{name}', exist_ok=True)
            
            if args.config['use_gnss']:
                os.makedirs(f'{args.path}/simbev/sweeps/GNSS', exist_ok=True)
            
            if args.config['use_imu']:
                os.makedirs(f'{args.path}/simbev/sweeps/IMU', exist_ok=True)
            
            os.makedirs(f'{args.path}/simbev/ground-truth/seg', exist_ok=True)
            os.makedirs(f'{args.path}/simbev/ground-truth/det', exist_ok=True)
            os.makedirs(f'{args.path}/simbev/ground-truth/seg_viz', exist_ok=True)
            os.makedirs(f'{args.path}/simbev/ground-truth/hd_map', exist_ok=True)
            
            os.makedirs(f'{args.path}/simbev/infos', exist_ok=True)
            
            os.makedirs(f'{args.path}/simbev/logs', exist_ok=True)

            os.makedirs(f'{args.path}/simbev/configs', exist_ok=True)

        if args.config['mode'] == 'create':
            scene_counter = 0

            core = CarlaCore(args.config)

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
                        
                        if town in args.config[f'{split}_scene_config']:
                            args.config[f'{split}_scene_config'][town] -= 1

                # Create the scenes for each town.
                if args.config[f'{split}_scene_config'] is not None:
                    for town in args.config[f'{split}_scene_config']:
                        if args.config[f'{split}_scene_config'][town] > 0:
                            core.connect_client()
                            
                            core.load_map(town)

                            core.spawn_vehicle()

                            vehicle_moved = False

                            for _ in range(args.config[f'{split}_scene_config'][town]):
                                print(f'Creating scene {scene_counter:04d} in {town} for the {split} set...')
                                
                                if vehicle_moved:
                                    core.move_vehicle()
                                
                                vehicle_moved = True

                                core.start_scene()

                                for _ in range(round(args.config['warmup_duration'] / args.config['timestep'])):
                                    core.tick()

                                # Start logging the scene.
                                if args.save:
                                    core.client.start_recorder(
                                        f'{args.path}/simbev/logs/SimBEV-scene-{scene_counter:04d}.log',
                                        True
                                    )

                                for j in range(round(args.config['scene_duration'] / args.config['timestep'])):
                                    core.tick(args.path, scene_counter, j, args.render, args.save)

                                if args.save:
                                    # Stop logging the scene.
                                    core.client.stop_recorder()

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
                                        yaml.dump(args.config, f)

                                core.stop_scene()
                                
                                scene_counter += 1

                            core.destroy_vehicle()
        
        elif args.config['mode'] == 'replace' and args.save:
            core = CarlaCore(args.config)

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
                    for scene_counter in args.config['scene_config']:
                        if f'scene_{scene_counter:04d}' in data.keys():
                            town = data[f'scene_{scene_counter:04d}']['scene_info']['map']

                            if town != core.map_name:
                                core.connect_client()
                                
                                core.load_map(town)

                            print(f'Replacing scene {scene_counter:04d} in {town} for the {split} set...')
                            
                            core.spawn_vehicle()
                            
                            core.start_scene()

                            for _ in range(round(args.config['warmup_duration'] / args.config['timestep'])):
                                core.tick()

                            core.client.start_recorder(
                                f'{args.path}/simbev/logs/SimBEV-scene-{scene_counter:04d}.log',
                                True
                            )

                            for j in range(round(args.config['scene_duration'] / args.config['timestep'])):
                                core.tick(args.path, scene_counter, j, args.render, args.save)

                            core.client.stop_recorder()

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
                                yaml.dump(args.config, f)
                            
                            core.stop_scene()

                            core.destroy_vehicle()
        
        print('Killing all servers...')
        
        kill_all_servers()

    except Exception:
        print(traceback.format_exc())

        print('Killing all servers...')

        kill_all_servers()

        time.sleep(3.0)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Killing all servers...')
        
        kill_all_servers()

        time.sleep(3.0)
    finally:
        print('Done.')