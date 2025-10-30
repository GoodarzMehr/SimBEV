# Academic Software License: Copyright © 2025 Goodarz Mehr.

import cv2
import time
import pyspng

import numpy as np

from pyquaternion import Quaternion as Q

from tools.visualization_utils import *

from concurrent.futures import ThreadPoolExecutor


CAM_NAME = [
    'CAM_FRONT_LEFT',
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT',
    'CAM_BACK',
    'CAM_BACK_RIGHT'
]

RAD_NAME = ['RAD_LEFT', 'RAD_FRONT', 'RAD_RIGHT', 'RAD_BACK']

LABEL_COLORS = np.array([
    (255, 255, 255), # None
    (128, 64, 128),  # Road
    (244, 35, 232),  # Sidewalk
    (70, 70, 70),    # Building
    (102, 102, 156), # Wall
    (190, 153, 153), # Fence
    (153, 153, 153), # Pole
    (250, 170, 30),  # TrafficLight
    (220, 220, 0),   # TrafficSign
    (107, 142, 35),  # Vegetation
    (152, 251, 152), # Terrain
    (70, 130, 180),  # Sky
    (220, 20, 60),   # Pedestrian
    (255, 0, 0),     # Rider
    (0, 0, 142),     # Car
    (0, 0, 70),      # Truck
    (0, 60, 100),    # Bus
    (0, 80, 100),    # Train
    (0, 0, 230),     # Motorcycle
    (119, 11, 32),   # Bicycle
    (110, 190, 160), # Static
    (170, 120, 50),  # Dynamic
    (55, 90, 80),    # Other
    (45, 60, 150),   # Water
    (227, 227, 227), # RoadLine
    (81, 0, 81),     # Ground
    (150, 100, 100), # Bridge
    (230, 150, 140), # RailTrack
    (180, 165, 180), # GuardRail
    (180, 130, 70)   # Rock
]) / 255.0


class VisualizationContext:
    '''
    Context object to hold common data for visualization.
    
    Args:
        path: root directory of the dataset.
        scene_number: scene number.
        frame_number: frame number.
        frame_data: data contained in the frame.
        metadata: dataset metadata.
        ignore_valid_flag: whether to ignore the valid_flag of object bounding
            boxes.
    '''
    def __init__(
            self,
            path: str,
            scene_number: int,
            frame_number: int,
            frame_data: dict,
            metadata: dict,
            ignore_valid_flag: bool = False
        ):
        self.path = path
        self.scene_number = scene_number
        self.frame_number = frame_number
        self.frame_data = frame_data
        self.metadata = metadata
        self.ignore_valid_flag = ignore_valid_flag
        
        self.gt_det = np.load(self.frame_data['GT_DET'], allow_pickle=True)
    
    def get_output_path(self, mode: str, camera: str = None):
        '''
        Generate output file path.
        
        Args:
            mode: visualization mode.
            camera: camera name (if applicable).
        
        Returns:
            output file path.
        '''
        if camera:
            return (
                f'{self.path}/simbev/viz/{mode}-{camera}/'
                f'SimBEV-scene-{self.scene_number:04d}-frame-{self.frame_number:04d}-{mode}-{camera}.jpg'
            )
        else:
            return (
                f'{self.path}/simbev/viz/{mode}/'
                f'SimBEV-scene-{self.scene_number:04d}-frame-{self.frame_number:04d}-{mode}.jpg'
            )


def visualize_rgb(ctx: VisualizationContext):
    '''
    Visualize RGB images with bounding boxes.
    
    Args:
        ctx: visualization context.
    '''
    gt_det = ctx.gt_det.copy()

    # Camera intrinsics.
    camera_intrinsics = np.eye(4, dtype=np.float32)
    
    camera_intrinsics[:3, :3] = ctx.metadata['camera_intrinsics']

    def process_rgb(camera):
        global2camera = get_global2sensor(ctx.frame_data, ctx.metadata, camera)

        # Global to image transformation.
        global2image = camera_intrinsics @ global2camera

        corners, labels = transform_bbox(gt_det, global2image, ctx.ignore_valid_flag)
        
        image = cv2.imread(ctx.frame_data['RGB-' + camera])

        visualize_image(ctx.get_output_path('RGB', camera), image, corners=corners, labels=labels)
    
    # Process all 6 cameras in parallel.
    with ThreadPoolExecutor(max_workers=6) as executor:
        executor.map(process_rgb, CAM_NAME)

def visualize_depth(ctx: VisualizationContext):
    '''Visualize depth images (parallel processing).'''
    
    def process_camera(camera):
        image = cv2.imread(ctx.frame_data['DPT-' + camera]).astype(np.float32)

        normalized_distance = (
            image[:, :, 0] + image[:, :, 1] * 256.0 + image[:, :, 2] * 256.0 * 256.0
        ) / (256.0 * 256.0 * 256.0 - 1)

        log_distance = 255 * np.log(256.0 * normalized_distance + 1) / np.log(257.0)

        cv2.imwrite(
            ctx.get_output_path('DPT', camera),
            log_distance.astype(np.uint8)
        )
    
    # Process all 6 cameras in parallel
    with ThreadPoolExecutor(max_workers=6) as executor:
        list(executor.map(process_camera, CAM_NAME))


def visualize_flow(ctx: VisualizationContext):
    '''Visualize optical flow.'''
    for camera in CAM_NAME:
        flow = np.load(ctx.frame_data['FLW-' + camera])['data']
        image = flow_to_color(flow)
        cv2.imwrite(ctx.get_output_path('FLW', camera), image)


def visualize_lidar(ctx: VisualizationContext):
    '''Visualize LiDAR point cloud (top-down view).'''
    point_cloud = np.load(ctx.frame_data['LIDAR'])['data']
    visualize_point_cloud_vectorized(ctx.get_output_path('LIDAR'), point_cloud)


def visualize_lidar_with_bbox(ctx: VisualizationContext):
    '''Visualize LiDAR with bounding boxes (top-down view).'''
    point_cloud = np.load(ctx.frame_data['LIDAR'])['data']
    gt_det = ctx.gt_det.copy()
    
    global2lidar = get_global2sensor(ctx.frame_data, ctx.metadata, 'LIDAR')
    corners, labels = transform_bbox(gt_det, global2lidar, ctx.ignore_valid_flag)

    visualize_point_cloud_vectorized(
        ctx.get_output_path('LIDARwBBOX'),
        point_cloud,
        corners=corners,
        labels=labels
    )


def visualize_lidar3d(ctx: VisualizationContext):
    '''Visualize LiDAR in 3D view.'''
    point_cloud = np.load(ctx.frame_data['LIDAR'])['data']
    
    lidar2image, camera_intrinsics = get_3d_view_transforms(ctx.metadata)
    
    distance = np.linalg.norm(point_cloud, axis=1)
    point_cloud_3d, canvas, indices, mask = project_to_3d_view(
        point_cloud, lidar2image, camera_intrinsics
    )
    
    distance = distance[indices][mask]
    color = compute_rainbow_colors(distance) * 255.0

    visualize_point_cloud_3d(
        ctx.get_output_path('LIDAR3D'),
        point_cloud_3d,
        canvas,
        color=color
    )


def visualize_lidar3d_with_bbox(ctx: VisualizationContext):
    '''Visualize LiDAR in 3D view with bounding boxes.'''
    point_cloud = np.load(ctx.frame_data['LIDAR'])['data']
    gt_det = ctx.gt_det.copy()
    
    global2lidar = get_global2sensor(ctx.frame_data, ctx.metadata, 'LIDAR')
    lidar2image, camera_intrinsics = get_3d_view_transforms(ctx.metadata)
    
    distance = np.linalg.norm(point_cloud, axis=1)
    point_cloud_3d, canvas, indices, mask = project_to_3d_view(
        point_cloud, lidar2image, camera_intrinsics
    )
    
    distance = distance[indices][mask]
    color = compute_rainbow_colors(distance) * 255.0
    
    global2image = lidar2image @ global2lidar
    corners, labels = transform_bbox(gt_det, global2image, ctx.ignore_valid_flag)

    visualize_point_cloud_3d(
        ctx.get_output_path('LIDAR3DwBBOX'),
        point_cloud_3d,
        canvas,
        corners=corners,
        labels=labels,
        color=color
    )


def visualize_semantic_lidar(ctx: VisualizationContext):
    '''Visualize semantic LiDAR (top-down view).'''
    data = np.load(ctx.frame_data['SEG-LIDAR'])['data']
    point_cloud = np.array([data['x'], data['y'], data['z']]).T
    labels = np.array(data['ObjTag'])
    label_color = LABEL_COLORS[labels]

    visualize_point_cloud_vectorized(
        ctx.get_output_path('SEG-LIDAR'),
        point_cloud,
        color=label_color
    )


def visualize_semantic_lidar3d(ctx: VisualizationContext):
    '''Visualize semantic LiDAR in 3D view.'''
    data = np.load(ctx.frame_data['SEG-LIDAR'])['data']
    point_cloud = np.array([data['x'], data['y'], data['z']]).T
    labels = np.array(data['ObjTag'])
    label_color = LABEL_COLORS[labels]
    
    lidar2image, camera_intrinsics = get_3d_view_transforms(ctx.metadata)
    point_cloud_3d, canvas, indices, mask = project_to_3d_view(
        point_cloud, lidar2image, camera_intrinsics
    )
    
    label_color = label_color[indices][mask]

    visualize_point_cloud_3d(
        ctx.get_output_path('SEG-LIDAR3D'),
        point_cloud_3d,
        canvas,
        color=(label_color * 255.0).astype(np.uint8)
    )


def load_radar_data(ctx: VisualizationContext):
    '''Load and transform radar data from all sensors.'''
    global2lidar = get_global2sensor(ctx.frame_data, ctx.metadata, 'LIDAR')
    
    point_cloud = []
    velocity = []
    
    for radar in RAD_NAME:
        # Radar to lidar transformation
        radar2lidar = np.eye(4, dtype=np.float32)
        radar2lidar[:3, :3] = Q(ctx.metadata[radar]['sensor2lidar_rotation']).rotation_matrix
        radar2lidar[:3, 3] = ctx.metadata[radar]['sensor2lidar_translation']

        radar_points = np.load(ctx.frame_data[radar])['data']
        velocity.append(radar_points[:, -1])
        radar_points = radar_points[:, :-1]

        # Convert spherical to Cartesian
        x = radar_points[:, 0] * np.cos(radar_points[:, 1]) * np.cos(radar_points[:, 2])
        y = radar_points[:, 0] * np.cos(radar_points[:, 1]) * np.sin(radar_points[:, 2])
        z = radar_points[:, 0] * np.sin(radar_points[:, 1])

        points = np.stack((x, y, z), axis=1)
        points_transformed = (
            radar2lidar @ np.append(points, np.ones((points.shape[0], 1)), 1).T
        )[:3].T

        point_cloud.append(points_transformed)
    
    point_cloud = np.concatenate(point_cloud, axis=0)
    velocity = np.concatenate(velocity, axis=0)
    
    # Compute velocity-based colors
    log_velocity = np.log(1.0 + np.abs(velocity))
    log_velocity_normalized = (
        log_velocity - log_velocity.min()
    ) / (
        log_velocity.max() - log_velocity.min() + 1e-6
    )

    color = np.c_[
        np.interp(log_velocity_normalized, RANGE, RAINBOW[:, 0]),
        np.interp(log_velocity_normalized, RANGE, RAINBOW[:, 1]),
        np.interp(log_velocity_normalized, RANGE, RAINBOW[:, 2])
    ]
    
    return point_cloud, color


def visualize_radar(ctx: VisualizationContext):
    '''Visualize radar point cloud (top-down view).'''
    point_cloud, color = load_radar_data(ctx)
    visualize_point_cloud(
        ctx.get_output_path('RADAR'),
        point_cloud,
        color=color,
        radius=128
    )


def visualize_radar_with_bbox(ctx: VisualizationContext):
    '''Visualize radar with bounding boxes (top-down view).'''
    point_cloud, color = load_radar_data(ctx)
    gt_det = ctx.gt_det.copy()
    
    global2lidar = get_global2sensor(ctx.frame_data, ctx.metadata, 'LIDAR')
    corners, labels = transform_bbox(gt_det, global2lidar)

    visualize_point_cloud(
        ctx.get_output_path('RADARwBBOX'),
        point_cloud,
        corners=corners,
        labels=labels,
        color=color,
        radius=128
    )


def visualize_radar3d(ctx: VisualizationContext):
    '''Visualize radar in 3D view.'''
    point_cloud, color = load_radar_data(ctx)
    
    lidar2image, camera_intrinsics = get_3d_view_transforms(ctx.metadata)
    point_cloud_3d, canvas, indices, mask = project_to_3d_view(
        point_cloud, lidar2image, camera_intrinsics
    )
    
    color = color[indices][mask]

    visualize_point_cloud_3d(
        ctx.get_output_path('RADAR3D'),
        point_cloud_3d,
        canvas,
        color=(color * 255.0).astype(np.uint8)
    )


def visualize_radar3d_with_bbox(ctx: VisualizationContext):
    '''Visualize radar in 3D view with bounding boxes.'''
    point_cloud, color = load_radar_data(ctx)
    gt_det = ctx.gt_det.copy()
    
    global2lidar = get_global2sensor(ctx.frame_data, ctx.metadata, 'LIDAR')
    lidar2image, camera_intrinsics = get_3d_view_transforms(ctx.metadata)
    
    point_cloud_3d, canvas, indices, mask = project_to_3d_view(
        point_cloud, lidar2image, camera_intrinsics
    )
    
    color = color[indices][mask]
    
    global2image = lidar2image @ global2lidar
    corners, labels = transform_bbox(gt_det, global2image)

    visualize_point_cloud_3d(
        ctx.get_output_path('RADAR3DwBBOX'),
        point_cloud_3d,
        canvas,
        corners=corners,
        labels=labels,
        color=(color * 255.0).astype(np.uint8)
    )
