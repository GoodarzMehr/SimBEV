# Academic Software License: Copyright © 2026 Goodarz Mehr.

import os
import json
import time
import argparse
import traceback

from tqdm import tqdm

from .visualization_handlers import *
from .visualization_interactive import *


VISUALIZATION_MODES = {
    'rgb': {
        'handler': visualize_rgb,
        'output_dirs': [f'RGB-{camera}' for camera in CAM_NAME],
        'color': '#FF00FF'
    },
    'depth': {
        'handler': visualize_depth,
        'output_dirs': [f'DPT-{camera}' for camera in CAM_NAME],
        'color': '#FF7700'
    },
    'flow': {
        'handler': visualize_flow,
        'output_dirs': [f'FLW-{camera}' for camera in CAM_NAME],
        'color': '#FFFF00'
    },
    'lidar': {
        'handler': visualize_lidar,
        'output_dirs': [f'LIDAR-{distance}' for distance in VIEWS.keys()],
        'color': '#99FF00'
    },
    'lidar-with-bbox': {
        'handler': visualize_lidar_with_bbox,
        'output_dirs': [f'LIDARwBBOX-{distance}' for distance in VIEWS.keys()],
        'color': '#55FF00'
    },
    'lidar3d': {
        'handler': visualize_lidar3d,
        'output_dirs': [f'LIDAR3D-{distance}' for distance in VIEWS.keys()],
        'color': '#00FF99'
    },
    'lidar3d-with-bbox': {
        'handler': visualize_lidar3d_with_bbox,
        'output_dirs': [f'LIDAR3DwBBOX-{distance}' for distance in VIEWS.keys()],
        'color': '#00FF55'
    },
    'semantic-lidar': {
        'handler': visualize_semantic_lidar,
        'output_dirs': [f'SEG-LIDAR-{distance}' for distance in VIEWS.keys()],
        'color': '#0099FF'
    },
    'semantic-lidar3d': {
        'handler': visualize_semantic_lidar3d,
        'output_dirs': [f'SEG-LIDAR3D-{distance}' for distance in VIEWS.keys()],
        'color': '#0055FF'
    },
    'radar': {
        'handler': visualize_radar,
        'output_dirs': [f'RADAR-{distance}' for distance in VIEWS.keys()],
        'color': '#9900FF'
    },
    'radar-with-bbox': {
        'handler': visualize_radar_with_bbox,
        'output_dirs': [f'RADARwBBOX-{distance}' for distance in VIEWS.keys()],
        'color': '#5500FF'
    },
    'radar3d': {
        'handler': visualize_radar3d,
        'output_dirs': [f'RADAR3D-{distance}' for distance in VIEWS.keys()],
        'color': '#FF0099'
    },
    'radar3d-with-bbox': {
        'handler': visualize_radar3d_with_bbox,
        'output_dirs': [f'RADAR3DwBBOX-{distance}' for distance in VIEWS.keys()],
        'color': '#FF0055'
    },
    'interactive': {
        'handler': visualize_interactive,
        'output_dirs': [],
        'color': '#00AAFF'
    }
}


argparser = argparse.ArgumentParser(description='SimBEV visualization tool.')

argparser.add_argument(
    'mode',
    nargs='+',
    help='visualization mode (all, rgb, depth, flow, lidar, lidar3d, lidar-with-bbox, lidar3d-with-bbox, '
        'semantic-lidar, semantic-lidar3d, radar, radar3d, radar-with-bbox, radar3d-with-bbox)'
)
argparser.add_argument(
    '--path',
    default='/dataset',
    help='path to the dataset (default: /dataset)'
)
argparser.add_argument(
    '-s', '--scene',
    nargs='+',
    default=['-1'],
    help='scene number range(s) (e.g. 2 3 4-6 5 8-12) (default: -1, i.e. all scenes)'
)
argparser.add_argument(
    '-f', '--frame',
    nargs='+',
    default=['-1'],
    help='frame number range(s) (e.g. 2 3 4-6 5 8-12) (default: -1, i.e. all frames)'
)
argparser.add_argument(
    '--ignore-valid-flag',
    action='store_true',
    help='ignore valid_flag when rendering bounding boxes'
)

args = argparser.parse_args()


def setup_output_directories(path: str, mode):
    '''
    Create the output directories for the given mode.
    
    Args:
        path: root directory of the dataset.
        mode: visualization mode.
    '''
    if mode in VISUALIZATION_MODES:
        for name in VISUALIZATION_MODES[mode]['output_dirs']:
            os.makedirs(f'{path}/simbev/viz/{name}', exist_ok=True)

def main(mode, path: str):
    try:
        if mode not in VISUALIZATION_MODES:
            print(f'Warning: unknown mode "{mode}", skipping.')
            
            return
        
        setup_output_directories(path, mode)
        
        handler = VISUALIZATION_MODES[mode]['handler']
        
        if mode == 'interactive':
            ctx = VisualizationContext(
                path,
                scene_number=None,
                frame_number=None,
                frame_data=None,
                metadata=None,
                ignore_valid_flag=args.ignore_valid_flag
            )
            
            handler(ctx)
            
            return
        
        for split in ['train', 'val', 'test']:
            info_path = f'{path}/simbev/infos/simbev_infos_{split}.json'
            
            if not os.path.exists(info_path):
                continue
                
            with open(info_path, 'r') as f:
                infos = json.load(f)

            metadata = infos['metadata']

            # Get the list of scenes to visualize.
            if args.scene == ['-1']:
                scene_list = [int(scene.split('_')[1]) for scene in infos['data']]
            else:
                scene_list = parse_range_argument(args.scene)

            scene_pbar = tqdm(
                scene_list,
                desc=f'Visualizing {mode}',
                ncols=120,
                colour=VISUALIZATION_MODES[mode]['color']
            )
            
            for scene_number in scene_pbar:
                scene_key = f'scene_{scene_number:04d}'
                
                if scene_key not in infos['data']:
                    continue

                scene_data = infos['data'][scene_key]['scene_data']

                # Get the list of frames to visualize.
                if args.frame == ['-1']:
                    frame_list = list(range(len(scene_data)))
                else:
                    requested_frames = parse_range_argument(args.frame)

                    # Filter to find valid frame numbers.
                    frame_list = [f for f in requested_frames if 0 <= f < len(scene_data)]
                    
                    # Warn about invalid frames
                    invalid_frames = [f for f in requested_frames if f < 0 or f >= len(scene_data)]
                    
                    if invalid_frames:
                        print(f'Warning: Scene {scene_number} has only {len(scene_data)} frames. '
                              f'Skipping invalid frames: {invalid_frames}')
                
                pbar = tqdm(
                    frame_list,
                    desc=f'{" " * (len(mode) + 2)}Scene {scene_number:04d}',
                    ncols=120,
                    colour='#0099FF',
                    leave=False
                )

                for frame_number in pbar:
                    frame_data = scene_data[frame_number]
                    
                    # Create the context.
                    ctx = VisualizationContext(
                        path,
                        scene_number,
                        frame_number,
                        frame_data,
                        metadata,
                        args.ignore_valid_flag
                    )

                    # Call the handler.
                    handler(ctx)
    
    except Exception:
        print(traceback.format_exc())
        
        print('Killing the process...')
        
        time.sleep(3.0)


def entry():
    try:
        start = time.perf_counter()

        os.makedirs(f'{args.path}/simbev/viz', exist_ok=True)

        # Determine modes to process
        if 'all' in args.mode:
            mode_list = [mode for mode in VISUALIZATION_MODES.keys() if mode not in ['interactive']]
        else:
            mode_list = args.mode

        # Process each mode
        for mode in mode_list:
            main(mode, args.path)
        
        end = time.perf_counter()

        print(f'Visualization completed in {end - start:.3f} seconds.')
    
    except KeyboardInterrupt:
        print('Killing the process...')
        
        time.sleep(3.0)
    
    finally:
        print('Done.')


if __name__ == '__main__':
    entry()