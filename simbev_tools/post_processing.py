# Academic Software License: Copyright © 2026 Goodarz Mehr.

import os
import cv2
import json
import time
import torch
import argparse
import traceback

import numpy as np
import torch.multiprocessing as mp

from tqdm import tqdm

from pyquaternion import Quaternion as Q

from concurrent.futures import ThreadPoolExecutor

try:
    from simbev_tools.bbox_cuda import num_inside_bbox_cuda as bbox_cuda_kernel
    
    BBOX_CUDA_AVAILABLE = True

except ImportError:
    print("Warning: CUDA bounding box extension not available. Performance will be degraded.")
    
    BBOX_CUDA_AVAILABLE = False

try:
    from simbev_tools.fill_voxel_cuda import (
        fill_hollow_voxels_cuda,
        morphological_close_3d_cuda
    )
    
    FILL_VOXEL_CUDA_AVAILABLE = True

except ImportError:
    print("Warning: CUDA voxel filling extension not available. Voxel filling disabled.")
    
    FILL_VOXEL_CUDA_AVAILABLE = False

OBJECT_CLASSES = {
    7:  'traffic_light',
    8:  'traffic_sign',
    12: 'pedestrian',
    13: 'rider',
    14: 'car',
    15: 'truck',
    16: 'bus',
    18: 'motorcycle',
    19: 'bicycle',
    30: 'traffic_cone',
    31: 'barrier'
}

DISTANCE_THRESHOLDS = {
    'traffic_light': [20.0, 40.0],
    'traffic_sign':  [20.0, 40.0],
    'pedestrian':    [20.0, 40.0],
    'rider':         [20.0, 40.0],
    'car':           [30.0, 60.0],
    'truck':         [40.0, 80.0],
    'bus':           [40.0, 80.0],
    'motorcycle':    [20.0, 40.0],
    'bicycle':       [20.0, 40.0],
    'traffic_cone':  [20.0, 40.0],
    'barrier':       [20.0, 40.0]
}

POINT_THRESHOLDS = {
    'traffic_light': [20, 10],
    'traffic_sign':  [20, 10],
    'pedestrian':    [40, 20],
    'rider':         [40, 20],
    'car':           [80, 30],
    'truck':         [100, 40],
    'bus':           [100, 40],
    'motorcycle':    [40, 20],
    'bicycle':       [40, 20],
    'traffic_cone':  [20, 10],
    'barrier':       [20, 10]
}

# Semantic priority map for voxel filling (higher value = higher priority).
# When filling voxels, higher priority classes won't be overwritten by lower
# priority ones.
SEMANTIC_PRIORITY = [
    0,  # 0:  Unlabeled
    7,  # 1:  Road
    7,  # 2:  Sidewalk
    1,  # 3:  Building
    1,  # 4:  Wall
    1,  # 5:  Fence
    1,  # 6:  Pole
    2,  # 7:  Traffic light
    2,  # 8:  Traffic sign
    1,  # 9:  Vegetation
    1,  # 10: Terrain
    1,  # 11: Sky
    6,  # 12: Pedestrian
    6,  # 13: Rider
    4,  # 14: Car
    3,  # 15: Truck
    3,  # 16: Bus
    1,  # 17: Train
    5,  # 18: Motorcycle
    5,  # 19: Bicycle
    1,  # 20: Static
    1,  # 21: Dynamic
    1,  # 22: Other
    1,  # 23: Water
    8,  # 24: Road line
    1,  # 25: Ground
    1,  # 26: Bridge
    1,  # 27: Rail track
    1,  # 28: Guard rail
    1,  # 29: Rock
    2,  # 30: Traffic cone
    2   # 31: Barrier
]

# Classes to fill, ordered by ascending priority (lower priority filled first,
# so higher priority classes can overwrite).
# Priority 1: Building(3), Pole(6), Static(20), Dynamic(21), Bridge(26),
#   RailTrack(27), GuardRail(28), Rock(29)
# Priority 2: TrafficLight(7), TrafficSign(8), TrafficCone(30), Barrier(31)
# Priority 3: Truck(15), Bus(16)
# Priority 4: Car(14)
# Priority 5: Motorcycle(18), Bicycle(19)
# Priority 6: Pedestrian(12), Rider(13)
FILLABLE_CLASSES = [3, 6, 20, 21, 26, 27, 28, 29, 7, 8, 30, 31, 15, 16, 14, 18, 19, 12, 13]

# Chunk sizes for different classes when filling hollow voxels. The voxel grid
# is broken into chunks and those that are completely empty are skipped to
# speed up the processing.
CHUNK_SIZE = {
    3:  (400, 400, 80),
    6:  (8, 8, 8),
    7:  (8, 8, 8),
    8:  (8, 8, 8),
    12: (8, 8, 8),
    13: (8, 8, 8),
    14: (16, 16, 16),
    15: (32, 32, 40),
    16: (32, 32, 40),
    18: (8, 8, 8),
    19: (8, 8, 8),
    20: (32, 32, 40),
    21: (32, 32, 40),
    26: (64, 64, 80),
    27: (64, 64, 80),
    28: (16, 16, 16),
    29: (8, 8, 8),
    30: (8, 8, 8),
    31: (8, 8, 8),
}

# Classes that use morphological closing instead of ray casting. This may be
# useful for classes like Vegetation that have branches/gaps that shouldn't be
# fully filled.
MORPHOLOGICAL_CLASSES = {}

CAM_NAME = [
    'CAM_FRONT_LEFT',
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT',
    'CAM_BACK',
    'CAM_BACK_RIGHT'
]

RAD_NAME = ['RAD_LEFT', 'RAD_FRONT', 'RAD_RIGHT', 'RAD_BACK']

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


argparser = argparse.ArgumentParser(description='SimBEV post-processing tool.')

argparser.add_argument(
    '--path',
    default='/dataset',
    help='path to the dataset (default: /dataset)'
)
argparser.add_argument(
    '--process-bbox',
    action='store_true',
    help='post-process bounding box annotations'
)
argparser.add_argument(
    '--no-process-bbox',
    dest='process_bbox',
    action='store_false',
    help='do not post-process bounding box annotations'
)
argparser.add_argument(
    '--use-seg',
    action='store_true',
    help='use instance segmentation images for post-processing bounding box annotations'
)
argparser.add_argument(
    '--fill-voxels',
    action='store_true',
    help='fill the hollow interiors of objects in voxel grids'
)
argparser.add_argument(
    '--morph-kernel-size',
    type=int,
    default=3,
    help='kernel size for morphological closing of vegetation (default: 3, must be odd)'
)
argparser.add_argument(
    '--num-gpus',
    type=int,
    default=-1,
    help='number of GPUs to use for voxel filling (-1 = all available, default: -1)'
)

argparser.set_defaults(process_bbox=True)

args = argparser.parse_args()


def _num_inside_bbox_cpu(points: torch.Tensor, bbox: torch.Tensor) -> int:
    '''
    Determine how many points from an array of points are inside a 3D bounding
    box (CPU implementation).
    
    Args:
        points: coordinate(s) of the point(s) (N, 3).
        bbox: coordinates of the bounding box corners (8, 3).
        
    Returns:
        number of points inside the bounding box.
    '''
    # Define the reference point, i.e. the first corner of the bounding box.
    p0 = bbox[0]
    
    # Define the local coordinate axes of the bounding box using the edges of the box.
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
    
    # Translate the points so that p0 is the origin, then transform the points
    # to the local bounding box coordinate system.
    points_local = (points - p0) @ R
    
    # Calculate the extents of the bounding box in the local coordinate system
    u_len = torch.linalg.norm(u)
    v_len = torch.linalg.norm(v)
    w_len = torch.linalg.norm(w)
    
    # Check if the points are inside the box in the local coordinates.
    inside_u = (points_local[:, 0] >= 0) & (points_local[:, 0] <= u_len)
    inside_v = (points_local[:, 1] >= 0) & (points_local[:, 1] <= v_len)
    inside_w = (points_local[:, 2] >= 0) & (points_local[:, 2] <= w_len)
    
    return (inside_u & inside_v & inside_w).sum()


def num_inside_bbox(
        points: torch.Tensor | np.ndarray,
        bbox: torch.Tensor | np.ndarray,
        device: str = 'cuda:0',
        dType: torch.dtype = torch.float
    ) -> int:
    '''
    Determine how many points from an array of points are inside a 3D bounding
    box.

    Args:
        points: coordinate(s) of the point(s) (N, 3).
        bbox: coordinates of the bounding box corners (8, 3).
        device: device to use for computation, can be 'cpu' or 'cuda:i' where
            i is the GPU index.
        dType: data type to use for calculations.

    Returns:
        number of points inside the bounding box.
    '''
    if not isinstance(points, torch.Tensor):
        points = torch.from_numpy(points).to(device, dType)

    if not isinstance(bbox, torch.Tensor):
        bbox = torch.from_numpy(bbox).to(device, dType)

    # Ensure proper shape.
    if points.dim() == 1:
        points = points.unsqueeze(0)
    
    # Handle empty point clouds.
    if points.shape[0] == 0:
        return 0
    
    # Ensure bbox is the right shape (8, 3) -> (24,).
    if bbox.dim() == 2:
        bbox = bbox.reshape(-1)
    
    # Make contiguous.
    points = points.contiguous()
    bbox = bbox.contiguous()
    
    # Validate shapes.
    assert points.shape[1] == 3, f'Points must have 3 coordinates, got shape {points.shape}'
    assert bbox.shape[0] == 24, f'Bounding box must have 24 elements (8 corners * 3 coordinates), got {bbox.shape[0]}'

    # Use CUDA kernel if available.
    if BBOX_CUDA_AVAILABLE and device.startswith('cuda'):
        try:
            count_tensor = bbox_cuda_kernel(points, bbox)
            
            return count_tensor.item()
        
        except RuntimeError as e:
            print(f'CUDA kernel failed: {e}')
            print(f'Falling back to CPU implementation')
            
            # Fall back to CPU implementation.
            points_cpu = points.cpu()
            
            bbox_cpu = bbox.reshape(8, 3).cpu()
            
            count = _num_inside_bbox_cpu(points_cpu, bbox_cpu)
            
            return count.item()
    else:
        # Fall back to CPU implementation.
        bbox = bbox.reshape(8, 3)

        count = _num_inside_bbox_cpu(points, bbox)
        
        return count.item()


def fill_voxel_interiors(
        voxel_grid: np.ndarray,
        device: str = 'cuda:0',
        morph_kernel_size: int = 3,
        voxel_size: float = 0.1
    ) -> np.ndarray:
    '''
    Fill the hollow interiors of objects in a 3D semantic voxel grid. Classes
    are processed in ascending priority order so that higher-priority classes
    can overwrite lower-priority ones. 6-direction raycasting is used for all
    classes except those (e.g. Vegetation) that use morphological closing.

    Args:
        voxel_grid: 3D NumPy array containing class labels.
        device: CUDA device to use for computation.
        morph_kernel_size: kernel size for morphological closing.
        voxel_size: size of each voxel.

    Returns:
        filled voxel grid.
    '''
    if not FILL_VOXEL_CUDA_AVAILABLE:
        print('Error: fill_voxel_cuda extension is not available.')
        
        return voxel_grid
    
    voxel_tensor = torch.from_numpy(voxel_grid).to(device)
    
    priority_map = torch.tensor(SEMANTIC_PRIORITY, dtype=torch.int32, device=device)
    
    # Process each class in ascending priority order.
    for target_class in FILLABLE_CLASSES:
        if target_class in MORPHOLOGICAL_CLASSES:
            voxel_tensor = morphological_close_3d_cuda(voxel_tensor, target_class, morph_kernel_size)
        else:
            chunk_size_x, chunk_size_y, chunk_size_z = CHUNK_SIZE[target_class]

            multiplier = max(1, 0.1 / voxel_size)

            chunk_size_x = int(min(chunk_size_x * multiplier, voxel_tensor.shape[0]))
            chunk_size_y = int(min(chunk_size_y * multiplier, voxel_tensor.shape[1]))
            chunk_size_z = int(min(chunk_size_z * multiplier, voxel_tensor.shape[2]))
            
            voxel_tensor = fill_hollow_voxels_cuda(
                voxel_tensor,
                priority_map,
                target_class,
                chunk_size_x,
                chunk_size_y,
                chunk_size_z
            )
    
    return voxel_tensor.cpu().numpy()


def _process_voxel_file(input_path: str, output_path: str, device: str, morph_kernel_size: int, voxel_size: float) -> bool:
    '''
    Worker function for processing a single voxel grid file.
    
    Args:
        input_path: path to the input voxel grid file.
        output_path: path to save the processed voxel grid file.
        device: CUDA device to use.
        morph_kernel_size: kernel size for morphological closing.
        voxel_size: size of each voxel.
    
    Returns:
        True if successful, False otherwise.
    '''
    try:
        voxel_grid = np.load(input_path)['data']
        
        filled_grid = fill_voxel_interiors(voxel_grid, device, morph_kernel_size, voxel_size)

        with open(output_path, 'wb') as f:
            np.savez_compressed(f, data=filled_grid)
        
        return True
    
    except Exception as e:
        print(f'Error processing {input_path}: {e}')
        
        return False


def _gpu_worker(gpu_id: int, file_list: tuple, morph_kernel_size: int, results_queue: mp.Queue, voxel_size: float):
    '''
    Worker process for a single GPU.
    
    Args:
        gpu_id: GPU device ID.
        file_list: list of (input_path, output_path) tuples to process.
        morph_kernel_size: kernel size for morphological closing.
        results_queue: multiprocessing queue for progress updates.
        voxel_size: size of each voxel.
    '''
    device = f'cuda:{gpu_id}'

    torch.cuda.set_device(gpu_id)
    
    for input_path, output_path in file_list:
        success = _process_voxel_file(input_path, output_path, device, morph_kernel_size, voxel_size)
        
        results_queue.put(1 if success else 0)


def fill_voxels_main():
    try:
        start = time.perf_counter()
        
        num_available_gpus = torch.cuda.device_count()
        
        if num_available_gpus == 0:
            print('Error: No CUDA GPUs available.')
            return
        
        if args.num_gpus == -1:
            num_gpus = num_available_gpus
        else:
            num_gpus = min(args.num_gpus, num_available_gpus)
        
        print(f'Using {num_gpus} GPU(s) for voxel filling.')
        
        # Create output directory.
        output_dir = f'{args.path}/simbev/sweeps/VOXEL-GRID-FILLED'
        
        os.makedirs(output_dir, exist_ok=True)

        file_pairs = []

        for split in ['train', 'val', 'test']:
            info_path = f'{args.path}/simbev/infos/simbev_infos_{split}.json'

            if not os.path.exists(info_path):
                continue

            with open(f'{args.path}/simbev/infos/simbev_infos_{split}.json', 'r') as f:
                infos = json.load(f)

            for scene in infos['data']:
                for info in infos['data'][scene]['scene_data']:
                    file_pairs.append((info['VOXEL-GRID'], info['VOXEL-GRID-FILLED']))
        
        print(f'Found {len(file_pairs)} voxel grid files to process.')

        metadata = infos['metadata']

        voxel_size = metadata['voxel_detector_properties']['voxel_size']
        
        if num_gpus == 1:
            device = 'cuda:0'
            
            pbar = tqdm(file_pairs, desc='Filling voxels', ncols=120, colour='cyan')
            
            for input_path, output_path in pbar:
                _process_voxel_file(input_path, output_path, device, args.morph_kernel_size, voxel_size)
        else:
            files_per_gpu = [[] for _ in range(num_gpus)]
            
            for i, file_pair in enumerate(file_pairs):
                files_per_gpu[i % num_gpus].append(file_pair)
            
            # Create a queue for progress updates.
            mp.set_start_method('spawn', force=True)
            
            results_queue = mp.Queue()
            
            # Start worker processes.
            processes = []
            
            for gpu_id in range(num_gpus):
                p = mp.Process(
                    target=_gpu_worker,
                    args=(gpu_id, files_per_gpu[gpu_id], args.morph_kernel_size, results_queue, voxel_size)
                )
                
                p.start()
                
                processes.append(p)
            
            # Show progress bar.
            pbar = tqdm(total=len(file_pairs), desc='Filling voxels', ncols=120, colour='cyan')
            
            completed = 0
            
            while completed < len(file_pairs):
                try:
                    result = results_queue.get(timeout=10.0)
                    
                    completed += result
                    
                    if result > 0:
                        pbar.update(1)
                except:
                    # Check if any process is still alive.
                    any_alive = any(p.is_alive() for p in processes)
                    
                    if not any_alive or completed == len(file_pairs):
                        break
            
            pbar.close()
            
            # Wait for all processes to finish.
            for p in processes:
                p.join()
        
        end = time.perf_counter()
        
        print(f'Voxel filling completed in {end - start:.3f} seconds.')

    except Exception:
        print(traceback.format_exc())
        
        print('Killing the process...')
        
        time.sleep(3.0)

def process_bbox_main():
    try:
        start = time.perf_counter()
        
        os.makedirs(f'{args.path}/simbev/ground-truth/new_det', exist_ok=True)

        for split in ['train', 'val', 'test']:
            info_path = f'{args.path}/simbev/infos/simbev_infos_{split}.json'

            if not os.path.exists(info_path):
                continue

            with open(f'{args.path}/simbev/infos/simbev_infos_{split}.json', 'r') as f:
                infos = json.load(f)

            metadata = infos['metadata']

            # Lidar to ego transformation.
            lidar2ego = np.eye(4).astype(np.float32)

            lidar2ego[:3, :3] = Q(metadata['LIDAR']['sensor2ego_rotation']).rotation_matrix
            lidar2ego[:3, 3] = metadata['LIDAR']['sensor2ego_translation']

            # Radar to ego transformations.
            radar2ego = {}

            for radar in RAD_NAME:
                radar2ego[radar] = np.eye(4).astype(np.float32)

                radar2ego[radar][:3, :3] = Q(metadata[radar]['sensor2ego_rotation']).rotation_matrix
                radar2ego[radar][:3, 3] = metadata[radar]['sensor2ego_translation']

            scene_pbar = tqdm(infos['data'], desc=f'Post-processing', ncols=120, colour='cyan')
            
            for scene in scene_pbar:
                scene_number = int(scene.split('_')[1])

                if infos['data'][scene]['scene_info']['map'] in ['Town12', 'Town13', 'Town15']:
                    dType = torch.double
                else:
                    dType = torch.float32

                pbar = tqdm(
                    infos['data'][scene]['scene_data'],
                    desc=f'{" " * 5}Scene {scene_number:04d}',
                    ncols=120,
                    colour='#00CC00',
                    leave=False
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
                                @ torch.cat((lidar_points, torch.ones(dim0, 1, device=DEVICE, dtype=dType)), dim=1).T
                        )[:3].T
                    else:
                        lidar_points_global = torch.empty((0, 3), device=DEVICE, dtype=dType)
                    
                    radar_points_list = []

                    for radar in RAD_NAME:
                        radar2global = ego2global @ radar2ego[radar]

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
                                    @ torch.cat((points, torch.ones(dim0, 1, device=DEVICE, dtype=dType)), dim=1).T
                            )[:3].T

                            radar_points_list.append(points_global)
                    
                    if len(radar_points_list) > 0:
                        radar_points_global = torch.cat(radar_points_list, dim=0).contiguous()
                    else:
                        radar_points_global = torch.empty((0, 3), device=DEVICE, dtype=dType)
                    
                    if args.use_seg:
                        color_hashes = set()

                        def process_camera(camera):
                            if f'IST-{camera}' in info:
                                image = cv2.imread(info[f'IST-{camera}']).astype(np.uint32)

                                # Convert the BGR image to a single
                                # integer: B + G * 256 + R * 65536.
                                color_hash = (image[:, :, 0] + \
                                                (image[:, :, 1] << 8) + \
                                                (image[:, :, 2] << 16))
                                
                                # Get the unique color hashes.
                                return np.unique(color_hash)
                            
                            return np.array([], dtype=np.uint32)
                        
                        # Process all 6 cameras in parallel
                        with ThreadPoolExecutor(max_workers=len(CAM_NAME)) as executor:
                            results = executor.map(process_camera, CAM_NAME)
                        
                        # Combine all unique hashes
                        for hashes in results:
                            color_hashes.update(hashes)
                    
                    # Load object bounding boxes.
                    det_objects = np.load(info['GT_DET'], allow_pickle=True)

                    new_det_objects = []

                    for obj in det_objects:
                        
                        num_lidar_points = num_inside_bbox(lidar_points_global, obj['bounding_box'], DEVICE, dType)
                        num_radar_points = num_inside_bbox(radar_points_global, obj['bounding_box'], DEVICE, dType)

                        valid_flag = (num_lidar_points > 0) or (num_radar_points > 0)

                        if not valid_flag and args.use_seg:
                            blue = (obj['id'] >> 8) & 0xFF
                            green = obj['id'] & 0xFF
                            
                            for red in obj['semantic_tags']:
                                target_hash = blue + (green << 8) + (red << 16)

                                if target_hash in color_hashes:
                                    valid_flag = True

                                    break
                        
                        obj['num_lidar_pts'] = num_lidar_points
                        obj['num_radar_pts'] = num_radar_points
                        obj['valid_flag'] = valid_flag

                        for tag in obj['semantic_tags']:
                            if tag in OBJECT_CLASSES:
                                obj['class'] = OBJECT_CLASSES[tag]

                                if obj['distance_to_ego'] >= DISTANCE_THRESHOLDS[obj['class']][1] or \
                                    (obj['num_lidar_pts'] + obj['num_radar_pts']) < POINT_THRESHOLDS[obj['class']][1]:
                                    obj['difficulty'] = 'hard'
                                elif obj['distance_to_ego'] >= DISTANCE_THRESHOLDS[obj['class']][0] or \
                                    (obj['num_lidar_pts'] + obj['num_radar_pts']) < POINT_THRESHOLDS[obj['class']][0]:
                                    obj['difficulty'] = 'medium'
                                else:
                                    obj['difficulty'] = 'easy'

                        new_det_objects.append(obj)
                    
                    with open(
                        f'{args.path}/simbev/ground-truth/new_det/SimBEV-scene-{scene_number:04d}-frame-{i:04d}-GT_DET.bin',
                        'wb'
                    ) as f:
                        np.save(f, np.array(new_det_objects), allow_pickle=True)

        os.rename(f'{args.path}/simbev/ground-truth/det', f'{args.path}/simbev/ground-truth/old_det')
        os.rename(f'{args.path}/simbev/ground-truth/new_det', f'{args.path}/simbev/ground-truth/det')

        end = time.perf_counter()
        
        print(f'Post-processing completed in {end - start:.3f} seconds.')

    except Exception:
        print(traceback.format_exc())

        print('Killing the process...')

        time.sleep(3.0)

def entry():
    try:
        if args.process_bbox:
            process_bbox_main()

        if args.fill_voxels:
            fill_voxels_main()
    
    except KeyboardInterrupt:
        print('Killing the process...')

        time.sleep(3.0)
    
    finally:
        print('Done.')


if __name__ == '__main__':
    entry()