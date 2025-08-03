# Academic Software License: Copyright © 2025 Goodarz Mehr.

import os
import json
import time
import torch
import argparse
import traceback

import numpy as np

from utils import is_inside_bbox

from pyquaternion import Quaternion as Q

RAD_NAME = ['RAD_LEFT', 'RAD_FRONT', 'RAD_RIGHT', 'RAD_BACK']

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


argparser = argparse.ArgumentParser(description='SimBEV post-processing tool.')

argparser.add_argument(
    '--path',
    default='/dataset',
    help='path to the dataset (default: /dataset)')

args = argparser.parse_args()


def main():
    try:
        os.makedirs(f'{args.path}/simbev/ground-truth/new_det', exist_ok=True)

        for split in ['train', 'val', 'test']:
            if os.path.exists(f'{args.path}/simbev/infos/simbev_infos_{split}.json'):
                with open(f'{args.path}/simbev/infos/simbev_infos_{split}.json', 'r') as f:
                    infos = json.load(f)

                metadata = infos['metadata']

                for scene in infos['data']:
                    scene_number = int(scene[-4:])

                    print(f'Processing scene {scene_number}...')

                    if infos['data'][scene]['scene_info']['map'] in ['Town12', 'Town13', 'Town15']:
                        dType = torch.double
                    else:
                        dType = torch.float32

                    for i, info in enumerate(infos['data'][scene]['scene_data']):
                        # Ego to global transformation.
                        ego2global = np.eye(4).astype(np.float32)

                        ego2global[:3, :3] = Q(info['ego2global_rotation']).rotation_matrix
                        ego2global[:3, 3] = info['ego2global_translation']

                        # Lidar to ego transformation.
                        lidar2ego = np.eye(4).astype(np.float32)

                        lidar2ego[:3, :3] = Q(metadata['LIDAR']['sensor2ego_rotation']).rotation_matrix
                        lidar2ego[:3, 3] = metadata['LIDAR']['sensor2ego_translation']

                        lidar2global = ego2global @ lidar2ego

                        # Load lidar point cloud.
                        if 'LIDAR' in info:
                            lidar_path = info['LIDAR']

                            if lidar_path.endswith('.npz'):
                                lidar_points = torch.from_numpy(np.load(lidar_path)['data']).to(DEVICE, dType)
                            else:
                                lidar_points = torch.from_numpy(np.load(lidar_path)).to(DEVICE, dType)
                            
                            dim0 = lidar_points.shape[0]

                            lidar_points_global = (
                                torch.from_numpy(lidar2global).to(DEVICE, dType) @ torch.cat(
                                    (lidar_points, torch.ones(dim0, 1).to(DEVICE, dType)), dim=1
                                ).T
                            )[:3].T
                        else:
                            lidar_points_global = torch.empty((0, 3), device=DEVICE, dtype=dType)

                        radar_points_global = torch.empty((0, 3), device=DEVICE, dtype=dType)

                        for radar in RAD_NAME:
                            # Radar to ego transformation.
                            radar2ego = np.eye(4).astype(np.float32)

                            radar2ego[:3, :3] = Q(metadata[radar]['sensor2ego_rotation']).rotation_matrix
                            radar2ego[:3, 3] = metadata[radar]['sensor2ego_translation']

                            radar2global = ego2global @ radar2ego

                            # Load radar point cloud.
                            if radar in info:
                                radar_path = info[radar]

                                if radar_path.endswith('.npz'):
                                    radar_points = torch.from_numpy(np.load(radar_path)['data']).to(DEVICE, dType)
                                else:
                                    radar_points = torch.from_numpy(np.load(radar_path)).to(DEVICE, dType)

                                # Transform depth, altitude, and azimuth data to x, y, and z.
                                radar_points = radar_points[:, :-1]

                                x = radar_points[:, 0] * torch.cos(radar_points[:, 1]) * torch.cos(radar_points[:, 2])
                                y = radar_points[:, 0] * torch.cos(radar_points[:, 1]) * torch.sin(radar_points[:, 2])
                                z = radar_points[:, 0] * torch.sin(radar_points[:, 1])

                                points = torch.stack((x, y, z), dim=1)

                                dim0 = points.shape[0]

                                points_global = (
                                    torch.from_numpy(radar2global).to(DEVICE, dType) @ torch.cat(
                                        (points, torch.ones(dim0, 1).to(DEVICE, dType)), dim=1
                                    ).T
                                )[:3].T

                                radar_points_global = torch.cat((radar_points_global, points_global), dim=0)
                        
                        # Load object bounding boxes.
                        det_objects = np.load(info['GT_DET'], allow_pickle=True)

                        new_det_objects = []

                        for obj in det_objects:
                            lidar_mask = is_inside_bbox(lidar_points_global, obj['bounding_box'], DEVICE, dType)       
                            radar_mask = is_inside_bbox(radar_points_global, obj['bounding_box'], DEVICE, dType)

                            num_lidar_points = lidar_mask.sum().item()
                            num_radar_points = radar_mask.sum().item()
                            
                            valid_flag = (num_lidar_points > 0) or (num_radar_points > 0)

                            obj['num_lidar_pts'] = num_lidar_points
                            obj['num_radar_pts'] = num_radar_points
                            obj['valid_flag'] = valid_flag

                            new_det_objects.append(obj)
                    
                        with open(
                            f'{args.path}/simbev/ground-truth/new_det/SimBEV-scene-{scene_number:04d}-frame-{i:04d}-GT_DET.bin',
                            'wb'
                        ) as f:
                            np.save(f, np.array(new_det_objects), allow_pickle=True)

        os.rename(f'{args.path}/simbev/ground-truth/det', f'{args.path}/simbev/ground-truth/old_det')
        os.rename(f'{args.path}/simbev/ground-truth/new_det', f'{args.path}/simbev/ground-truth/det')

    except Exception:
        print(traceback.format_exc())

        print('Killing the process...')

        time.sleep(3.0)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Killing the process...')

        time.sleep(3.0)
    finally:
        print('Done.')
