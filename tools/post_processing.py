# Academic Software License: Copyright © 2025 Goodarz Mehr.

import os
import json
import time
import torch
import argparse
import traceback

import numpy as np

from tqdm import tqdm

from pyquaternion import Quaternion as Q

try:
    from tools.bbox_cuda import num_inside_bbox_cuda as bbox_cuda_kernel
    
    CUDA_AVAILABLE = True

except ImportError:
    print("Warning: CUDA extension not available. Performance will be degraded.")
    CUDA_AVAILABLE = False


RAD_NAME = ['RAD_LEFT', 'RAD_FRONT', 'RAD_RIGHT', 'RAD_BACK']

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


argparser = argparse.ArgumentParser(description='SimBEV post-processing tool.')

argparser.add_argument(
    '--path',
    default='/dataset',
    help='path to the dataset (default: /dataset)')
argparser.add_argument(
    '--use-seg',
    action='store_true',
    help='use semantic segmentation images for post-processing')

args = argparser.parse_args()


def _num_inside_bbox_cpu(points, bbox):
    '''
    CPU fallback implementation for bounding box checking.
    
    Args:
        points: coordinate(s) of the point(s) (N, 3).
        bbox: array of the coordinates of the bounding box corners (8, 3).
        
    Returns:
        mask: mask indicating if the point(s) are inside the bounding box.
    '''
    # Define the reference point, i.e. first corner of the bounding box.
    p0 = bbox[0]
    
    # Define local coordinate axes of the bounding box using edges of the box.
    u = bbox[2] - p0
    v = bbox[4] - p0
    w = bbox[1] - p0
    
    # Normalize the axes to get unit vectors.
    u_norm = u / torch.linalg.norm(u)
    v_norm = v / torch.linalg.norm(v)
    w_norm = w / torch.linalg.norm(w)

    # Set up the transformation matrix to map the points to the local
    # coordinate system of the bounding box.
    R = torch.vstack([u_norm, v_norm, w_norm]).T
    
    # Translate points so that p0 is the origin.
    points_translated = points - p0
    
    # Transform points to the local bounding box coordinate system.
    points_local = points_translated @ R
    
    # Calculate the extents of the bounding box in the local coordinate system
    u_len = torch.linalg.norm(u)
    v_len = torch.linalg.norm(v)
    w_len = torch.linalg.norm(w)
    
    # Check if the points are inside the box in the local coordinates.
    inside_u = (points_local[:, 0] >= 0) & (points_local[:, 0] <= u_len)
    inside_v = (points_local[:, 1] >= 0) & (points_local[:, 1] <= v_len)
    inside_w = (points_local[:, 2] >= 0) & (points_local[:, 2] <= w_len)
    
    return inside_u & inside_v & inside_w


def num_inside_bbox(
        points: torch.Tensor | np.ndarray,
        bbox: torch.Tensor | np.ndarray,
        device: str = 'cuda:0',
        dType: torch.dtype = torch.float
    ) -> int:
    '''
    Determine if a point (or array of points) is inside a 3D bounding box.

    Args:
        points: coordinate(s) of the point(s) (N, 3).
        bbox: array of the coordinates of the bounding box corners (8, 3).
        device: device to use for computation.
        dType: data type to use for calculations.

    Returns:
        mask: mask indicating if the point(s) are inside the bounding box.
    '''
    if not isinstance(points, torch.Tensor):
        points = torch.from_numpy(points).to(device, dType)
    else:
        points = points.to(device, dType)

    if not isinstance(bbox, torch.Tensor):
        bbox = torch.from_numpy(bbox).to(device, dType)
    else:
        bbox = bbox.to(device, dType)

    # Ensure proper shape and contiguity
    if points.dim() == 1:
        points = points.unsqueeze(0)
    
    # Handle empty point clouds
    if points.shape[0] == 0:
        return 0
    
    # Ensure bbox is the right shape (8, 3) -> (24,)
    if bbox.dim() == 2:
        bbox = bbox.reshape(-1)
    
    # Make contiguous
    points = points.contiguous()
    bbox = bbox.contiguous()
    
    # Validate shapes
    assert points.shape[1] == 3, f"Points must have 3 coordinates, got shape {points.shape}"
    assert bbox.shape[0] == 24, f"Bbox must have 24 elements (8 corners * 3 coords), got {bbox.shape[0]}"
    
    # Call CUDA kernel if available
    if CUDA_AVAILABLE and device.startswith('cuda'):
        try:
            count_tensor = bbox_cuda_kernel(points, bbox)
            return count_tensor.item()
        except RuntimeError as e:
            print(f"CUDA kernel failed: {e}")
            print(f"Falling back to CPU implementation")
            print(f"Points shape: {points.shape}, dtype: {points.dtype}")
            print(f"Bbox shape: {bbox.shape}, dtype: {bbox.dtype}")
            # Fall back to CPU
            points_cpu = points.cpu()
            bbox_cpu = bbox.reshape(8, 3).cpu()
            mask = _num_inside_bbox_cpu(points_cpu, bbox_cpu)
            return mask.sum().item()
    else:
        # Fallback to CPU implementation
        if bbox.dim() == 1:
            bbox = bbox.reshape(8, 3)
        mask = _num_inside_bbox_cpu(points, bbox)
        return mask.sum().item()

def main():
    try:
        start = time.perf_counter()
        
        os.makedirs(f'{args.path}/simbev/ground-truth/new_det', exist_ok=True)

        for split in ['train', 'val', 'test']:
            if os.path.exists(f'{args.path}/simbev/infos/simbev_infos_{split}.json'):
                with open(f'{args.path}/simbev/infos/simbev_infos_{split}.json', 'r') as f:
                    infos = json.load(f)

                metadata = infos['metadata']

                # Lidar to ego transformation.
                lidar2ego = np.eye(4).astype(np.float32)

                lidar2ego[:3, :3] = Q(metadata['LIDAR']['sensor2ego_rotation']).rotation_matrix
                lidar2ego[:3, 3] = metadata['LIDAR']['sensor2ego_translation']

                for scene in infos['data']:
                    scene_number = int(scene[-4:])

                    if infos['data'][scene]['scene_info']['map'] in ['Town12', 'Town13', 'Town15']:
                        dType = torch.double
                    else:
                        dType = torch.float32

                    pbar = tqdm(
                        infos['data'][scene]['scene_data'],
                        desc=f'Scene {scene_number:04d}',
                        ncols=120,
                        colour='blue'
                    )
                    
                    for i, info in enumerate(pbar):
                        # Ego to global transformation.
                        ego2global = np.eye(4).astype(np.float32)

                        ego2global[:3, :3] = Q(info['ego2global_rotation']).rotation_matrix
                        ego2global[:3, 3] = info['ego2global_translation']

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
                                torch.from_numpy(lidar2global).to(DEVICE, dType) \
                                    @ torch.cat(
                                        (lidar_points, torch.ones(dim0, 1, device=DEVICE, dtype=dType)),
                                        dim=1
                                    ).T
                            )[:3].T
                        else:
                            lidar_points_global = torch.empty((0, 3), device=DEVICE, dtype=dType)
                        
                        radar_points_list = []

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
                                    torch.from_numpy(radar2global).to(DEVICE, dType) \
                                        @ torch.cat(
                                            (points, torch.ones(dim0, 1, device=DEVICE, dtype=dType)),
                                            dim=1
                                        ).T
                                )[:3].T

                                radar_points_list.append(points_global)

                        if len(radar_points_list) > 0:
                            radar_points_global = torch.cat(radar_points_list, dim=0).contiguous()
                        else:
                            radar_points_global = torch.empty((0, 3), device=DEVICE, dtype=dType)
                        
                        # Load object bounding boxes.
                        det_objects = np.load(info['GT_DET'], allow_pickle=True)

                        new_det_objects = []

                        for obj in det_objects:
                            num_lidar_points = num_inside_bbox(lidar_points_global, obj['bounding_box'], DEVICE, dType)
                            num_radar_points = num_inside_bbox(radar_points_global, obj['bounding_box'], DEVICE, dType)

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

        # os.rename(f'{args.path}/simbev/ground-truth/det', f'{args.path}/simbev/ground-truth/old_det')
        # os.rename(f'{args.path}/simbev/ground-truth/new_det', f'{args.path}/simbev/ground-truth/det')

        end = time.perf_counter()
        
        print(f'Post-processing completed in {end - start:.3f} seconds.')

    except Exception:
        print(traceback.format_exc())

        print('Killing the process...')

        time.sleep(3.0)

def entry():
    try:
        main()
    
    except KeyboardInterrupt:
        print('Killing the process...')

        time.sleep(3.0)
    
    finally:
        print('Done.')


if __name__ == '__main__':
    entry()