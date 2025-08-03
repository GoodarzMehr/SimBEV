# Academic Software License: Copyright © 2025 Goodarz Mehr.

import os
import cv2
import json
import time
import flow_vis
import argparse
import traceback

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import colormaps as cm

from pyquaternion import Quaternion as Q


CAM_NAME = [
    'CAM_FRONT_LEFT',
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT',
    'CAM_BACK',
    'CAM_BACK_RIGHT'
]

RAD_NAME = ['RAD_LEFT', 'RAD_FRONT', 'RAD_RIGHT', 'RAD_BACK']

OBJECT_CLASSES = {
    7:  'traffic_light',
    8:  'traffic_sign',
    12: 'pedestrian',
    13: 'rider',
    14: 'car',
    15: 'truck',
    16: 'bus',
    18: 'motorcycle',
    19: 'bicycle'
}

SIMBEV_PALETTE = {
    'road': (196, 80, 196),
    'car': (0, 128, 240),
    'truck': (128, 240, 64),
    'bus': (0, 144, 0),
    'motorcycle': (240, 240, 0),
    'bicycle': (0, 240, 240),
    'rider': (240, 144, 0),
    'pedestrian': (240, 0, 0),
    'traffic_light': (240, 160, 0),
    'traffic_sign': (240, 0, 128)
}

RANGE = np.linspace(0.0, 1.0, 256)

RAINBOW = np.array(cm.get_cmap('rainbow')(RANGE))[:, :3]

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
]) / 255.0


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
    type=int,
    default=[-1],
    nargs='+',
    help='scene number(s) (default: -1, i.e. all scenes)'
)
argparser.add_argument(
    '-f', '--frame',
    type=int,
    default=[-1],
    nargs='+',
    help='frame number(s) (default: -1, i.e. all frames)'
)

args = argparser.parse_args()


def draw_bbox(canvas, corners, labels, bbox_color=None, thickness=1):
    '''
    Draw bounding boxes on the canvas.

    Args:
        canvas: canvas to draw on.
        corners: bounding box corners.
        labels: bounding box labels.
        bbox_color: bounding box color.
        thickness: bounding box line thickness.
    '''
    # Filter out bounding boxes that are behind the camera.
    indices = np.all(corners[..., 2] > 0, axis=1)
    corners = corners[indices]
    labels = labels[indices]

    # Sort bounding boxes by their distance to the camera.
    indices = np.argsort(-np.min(corners[..., 2], axis=1))
    corners = corners[indices]
    labels = labels[indices]

    # Find the pixels corresponding to bounding box corners.
    corners = corners.reshape(-1, 3)
    corners[:, 2] = np.clip(corners[:, 2], a_min=1e-5, a_max=1e5)
    corners[:, 0] /= corners[:, 2]
    corners[:, 1] /= corners[:, 2]

    corners = corners[..., :2].reshape(-1, 8, 2)

    # Draw bounding box lines.
    if corners is not None:
        for index in range(corners.shape[0]):
            name = labels[index]
            
            for start, end in [
                (0, 1),
                (0, 2),
                (0, 4),
                (1, 3),
                (1, 5),
                (2, 3),
                (2, 6),
                (3, 7),
                (4, 5),
                (4, 6),
                (5, 7),
                (6, 7)
            ]:
                cv2.line(
                    canvas,
                    corners[index, start].astype(int),
                    corners[index, end].astype(int),
                    bbox_color or SIMBEV_PALETTE[name],
                    thickness,
                    cv2.LINE_AA,
                )

        canvas = canvas.astype(np.uint8)
    
    return canvas


def visualize_image(fpath, image, corners=None, labels=None, bbox_color=None, thickness=2):
    '''
    Visualize image with bounding boxes.

    Args:
        fpath: file path to save the image.
        image: image to visualize.
        corners: bounding box corners.
        labels: bounding box labels.
        color: bounding box color.
        thickness: bounding box line thickness.
    '''
    canvas = image.copy()

    if corners is not None and corners.shape[0] > 0:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

        canvas = draw_bbox(canvas, corners, labels, bbox_color=bbox_color, thickness=thickness)
        
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    # Save the image.
    cv2.imwrite(fpath, canvas)


def visualize_point_cloud(
        fpath,
        point_cloud,
        corners=None,
        labels=None,
        color=None,
        xlim=(-72, 72),
        ylim=(-72, 72),
        radius=16,
        thickness=16
    ):
    '''
    Visualize point cloud from above with bounding boxes.

    Args:
        fpath: file path to save the image.
        point_cloud: point cloud to visualize.
        corners: bounding box corners.
        labels: bounding box labels.
        color: point cloud color(s).
        xlim: x-axis limits.
        ylim: y-axis limits.
        radius: display point radius.
        thickness: bounding box line thickness.
    '''
    fig = plt.figure(figsize=(xlim[1] - xlim[0], ylim[1] - ylim[0]))

    ax = plt.gca()

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    
    ax.set_aspect(1)
    ax.set_axis_off()

    # Filter out points that are outside the limits.
    mask = np.logical_and.reduce([
        point_cloud[:, 0] >= xlim[0],
        point_cloud[:, 0] < xlim[1],
        point_cloud[:, 1] >= ylim[0],
        point_cloud[:, 1] < ylim[1]
    ])

    point_cloud = point_cloud[mask]

    # Plot the point cloud.
    if point_cloud is not None:
        if color is None:
            plt.scatter(
                point_cloud[:, 0],
                point_cloud[:, 1],
                s=radius,
                c=np.log(np.linalg.norm(point_cloud, axis=1)),
                cmap='rainbow'
            )
        else:
            color = color[mask]

            plt.scatter(point_cloud[:, 0], point_cloud[:, 1], s=radius, c=color)
    
    # Draw bounding box lines.
    if corners is not None:
        coords = corners[:, [0, 2, 6, 4, 0], :2]

        for index in range(coords.shape[0]):
            name = labels[index]
            
            plt.plot(
                coords[index, :, 0],
                coords[index, :, 1],
                color=np.array(SIMBEV_PALETTE[name]) / 255.0,
                linewidth=thickness
            )
    
    # Save the image.
    fig.savefig(fpath, dpi=16, facecolor='white', format='jpg', bbox_inches='tight', pad_inches=0)
    
    plt.close()

def visualize_point_cloud_3d(
        fpath,
        point_cloud,
        canvas,
        corners=None,
        labels=None,
        color=None,
        bbox_color=None,
        thickness=1
    ):
    '''
    Visualize point cloud in 3D with bounding boxes.

    Args:
        fpath: file path to save the image.
        point_cloud: point cloud to visualize.
        canvas: canvas to draw on.
        corners: bounding box corners.
        labels: bounding box labels.
        color: point cloud color(s).
        thickness: bounding box line thickness.
    '''
    if color is not None:
        canvas[point_cloud[:, 1].astype(int), point_cloud[:, 0].astype(int), :] = color

    if corners is not None and corners.shape[0] > 0:
        canvas = draw_bbox(canvas, corners, labels, bbox_color=bbox_color, thickness=thickness)
    
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    cv2.imwrite(fpath, canvas)

def transform_bbox(gt_det, transform):
    '''
    Transform bounding boxes from the global coordinate system to the desired
    coordinate system.

    Args:
        gt_det: ground truth objects.
        transform: transformation matrix.
    
    Returns:
        corners: transformed bounding box corners.
        labels: bounding box labels.
    '''
    corners = []
    labels = []

    # Transform bounding boxes from the global coordinate system to the
    # desired coordinate system.
    for det_object in gt_det:
        if det_object['valid_flag']:
            for tag in det_object['semantic_tags']:
                if tag in OBJECT_CLASSES.keys():
                    global_bbox_corners = np.append(det_object['bounding_box'], np.ones((8, 1)), 1)
                    bbox_corners = (transform @ global_bbox_corners.T)[:3].T

                    corners.append(bbox_corners)
                    labels.append(OBJECT_CLASSES[tag])

    corners = np.array(corners)
    labels = np.array(labels)

    return corners, labels


def main(mode):
    try:
        print(f'Visualizing {mode}...')
        
        if mode == 'rgb':
            for camera in CAM_NAME:
                os.makedirs(f'{args.path}/simbev/viz/RGB-{camera}', exist_ok=True)
        
        if mode == 'depth':
            for camera in CAM_NAME:
                os.makedirs(f'{args.path}/simbev/viz/DPT-{camera}', exist_ok=True)
        
        if mode == 'flow':
            for camera in CAM_NAME:
                os.makedirs(f'{args.path}/simbev/viz/FLW-{camera}', exist_ok=True)

        if mode == 'lidar':
            os.makedirs(f'{args.path}/simbev/viz/LIDAR', exist_ok=True)
        
        if mode == 'lidar3d':
            os.makedirs(f'{args.path}/simbev/viz/LIDAR3D', exist_ok=True)
        
        if mode == 'lidar-with-bbox':
            os.makedirs(f'{args.path}/simbev/viz/LIDARwBBOX', exist_ok=True)
        
        if mode == 'lidar3d-with-bbox':
            os.makedirs(f'{args.path}/simbev/viz/LIDAR3DwBBOX', exist_ok=True)
        
        if mode == 'semantic-lidar':
            os.makedirs(f'{args.path}/simbev/viz/SEG-LIDAR', exist_ok=True)
        
        if mode == 'semantic-lidar3d':
            os.makedirs(f'{args.path}/simbev/viz/SEG-LIDAR3D', exist_ok=True)
        
        if mode == 'radar':
            os.makedirs(f'{args.path}/simbev/viz/RADAR', exist_ok=True)
        
        if mode == 'radar3d':
            os.makedirs(f'{args.path}/simbev/viz/RADAR3D', exist_ok=True)
        
        if mode == 'radar-with-bbox':
            os.makedirs(f'{args.path}/simbev/viz/RADARwBBOX', exist_ok=True)
        
        if mode == 'radar3d-with-bbox':
            os.makedirs(f'{args.path}/simbev/viz/RADAR3DwBBOX', exist_ok=True)
        
        for split in ['train', 'val', 'test']:
            if os.path.exists(f'{args.path}/simbev/infos/simbev_infos_{split}.json'):
                with open(f'{args.path}/simbev/infos/simbev_infos_{split}.json', 'r') as f:
                    infos = json.load(f)

                metadata = infos['metadata']

                # Get the list of scenes to visualize.
                if args.scene == [-1]:
                    scene_list = []

                    for scene in infos['data']:
                        scene_list.append(int(scene[-4:]))
                else:
                    scene_list = args.scene

                for scene_number in scene_list:
                    if f'scene_{scene_number:04d}' in infos['data']:
                        print(f'Visualizing scene {scene_number}...')

                        scene_data = infos['data'][f'scene_{scene_number:04d}']['scene_data']

                        if args.frame == [-1]:
                            frame_list = list(range(len(scene_data)))
                        else:
                            frame_list = [frame for frame in args.frame if frame < len(scene_data) and frame >= 0]

                        for frame_number in frame_list:
                            frame_data = scene_data[frame_number]

                            if mode == 'rgb' or 'bbox' in mode:
                                # Load object bounding boxes.
                                gt_det_path = frame_data['GT_DET']

                                gt_det = np.load(gt_det_path, allow_pickle=True)

                                # Ego to global transformation.
                                ego2global = np.eye(4).astype(np.float32)

                                ego2global[:3, :3] = Q(frame_data['ego2global_rotation']).rotation_matrix
                                ego2global[:3, 3] = frame_data['ego2global_translation']
                            
                            if 'bbox' in mode:
                                # Lidar to ego transformation.
                                lidar2ego = np.eye(4).astype(np.float32)

                                lidar2ego[:3, :3] = Q(metadata['LIDAR']['sensor2ego_rotation']).rotation_matrix
                                lidar2ego[:3, 3] = metadata['LIDAR']['sensor2ego_translation']

                                global2lidar = np.linalg.inv(ego2global @ lidar2ego)
                            
                            if '3d' in mode:
                                view2lidar = np.eye(4).astype(np.float32)

                                view2lidar[:3, :3] = Q([0.415627, -0.572061, 0.572061, -0.415627]).rotation_matrix
                                view2lidar[:3, 3] = [-40.0, 0.0, 12.0]

                                lidar2view = np.linalg.inv(view2lidar)

                                camera_intrinsics = np.eye(4).astype(np.float32)

                                camera_intrinsics[:3, :3] = metadata['camera_intrinsics']

                                lidar2image = camera_intrinsics @ lidar2view
                            
                            if mode == 'rgb':
                                for camera in CAM_NAME:
                                    # Camera to ego transformation.
                                    camera2ego = np.eye(4).astype(np.float32)

                                    camera2ego[:3, :3] = Q(metadata[camera]['sensor2ego_rotation']).rotation_matrix
                                    camera2ego[:3, 3] = metadata[camera]['sensor2ego_translation']

                                    # Camera intrinsics.
                                    camera_intrinsics = np.eye(4).astype(np.float32)

                                    camera_intrinsics[:3, :3] = metadata['camera_intrinsics']

                                    # Global to camera transformation.
                                    global2camera = np.linalg.inv(ego2global @ camera2ego)

                                    global2image = camera_intrinsics @ global2camera

                                    corners, labels = transform_bbox(gt_det, global2image)

                                    image = cv2.imread(frame_data['RGB-' + camera])

                                    visualize_image(
                                        f'{args.path}/simbev/viz/RGB-{camera}/SimBEV-scene-' \
                                            f'{scene_number:04d}-frame-{frame_number:04d}-RGB-{camera}.jpg',
                                        image,
                                        corners=corners,
                                        labels=labels
                                    )
                                
                            if mode == 'depth':
                                for camera in CAM_NAME:
                                    image = cv2.imread(frame_data['DPT-' + camera])

                                    normalized_distance = (
                                        image[:, :, 2] + image[:, :, 1] * 256.0 + image[:, :, 0] * 256.0 * 256.0
                                        ) / (256.0 * 256.0 * 256.0 - 1)

                                    log_distance = 255 * np.log(256.0 * normalized_distance + 1) / np.log(257.0)

                                    cv2.imwrite(
                                        f'{args.path}/simbev/viz/DPT-{camera}/SimBEV-scene-' \
                                            f'{scene_number:04d}-frame-{frame_number:04d}-DPT-{camera}.jpg',
                                        log_distance.astype(np.uint8)
                                    )
                            
                            if mode == 'flow':
                                for camera in CAM_NAME:
                                    flow = np.load(frame_data['FLW-' + camera])['data']

                                    image = flow_vis.flow_to_color(flow, convert_to_bgr=True)

                                    cv2.imwrite(
                                        f'{args.path}/simbev/viz/FLW-{camera}/SimBEV-scene-' \
                                            f'{scene_number:04d}-frame-{frame_number:04d}-FLW-{camera}.jpg',
                                        image
                                    )
                            
                            if mode in ['lidar', 'lidar3d', 'lidar-with-bbox', 'lidar3d-with-bbox']:
                                point_cloud = np.load(frame_data['LIDAR'])['data']

                                label_color = None
                            
                            if mode in ['semantic-lidar', 'semantic-lidar3d']:
                                data = np.load(frame_data['SEG-LIDAR'])['data']

                                point_cloud = np.array([data['x'], data['y'], data['z']]).T

                                labels = np.array(data['ObjTag'])

                                label_color = LABEL_COLORS[labels]
                            
                            if 'lidar3d' in mode:
                                distance = np.linalg.norm(point_cloud, axis=1)

                                point_cloud_3d = (
                                    lidar2image @ np.append(point_cloud, np.ones((point_cloud.shape[0], 1)), 1).T
                                )[:3].T
                                
                                # Filter out points that are behind the 3D view camera.
                                indices = point_cloud_3d[:, 2] > 0.0
                                point_cloud_3d = point_cloud_3d[indices]
                                distance = distance[indices]

                                if label_color is not None:
                                    label_color = label_color[indices]

                                # Find the pixels corresponding to the point cloud.
                                point_cloud_3d[:, 2] = np.clip(point_cloud_3d[:, 2], a_min=1e-5, a_max=1e5)
                                point_cloud_3d[:, 0] /= point_cloud_3d[:, 2]
                                point_cloud_3d[:, 1] /= point_cloud_3d[:, 2]

                                # Set the 3D view camera dimensions.
                                width = camera_intrinsics[0, 2] * 2
                                height = camera_intrinsics[1, 2] * 2

                                canvas = np.zeros((int(height), int(width), 3), dtype=np.uint8)
                                canvas[:] = (255, 255, 255)

                                mask = np.logical_and.reduce([
                                    point_cloud_3d[:, 0] >= 0,
                                    point_cloud_3d[:, 0] < width,
                                    point_cloud_3d[:, 1] >= 0,
                                    point_cloud_3d[:, 1] < height
                                ])
                                point_cloud_3d = point_cloud_3d[mask]
                                distance = distance[mask]

                                if label_color is not None:
                                    label_color = label_color[mask]

                                # Calculate point cloud colors.
                                log_distance = np.log(distance)

                                log_distance_normalized = (
                                    log_distance - log_distance.min()
                                ) / (
                                    log_distance.max() - log_distance.min() + 1e-6
                                )

                                color = np.c_[
                                    np.interp(log_distance_normalized, RANGE, RAINBOW[:, 0]),
                                    np.interp(log_distance_normalized, RANGE, RAINBOW[:, 1]),
                                    np.interp(log_distance_normalized, RANGE, RAINBOW[:, 2])
                                ] * 255.0
                            
                            if mode == 'lidar':
                                visualize_point_cloud(
                                    f'{args.path}/simbev/viz/LIDAR/' \
                                        f'SimBEV-scene-{scene_number:04d}-frame-{frame_number:04d}-LIDAR.jpg',
                                    point_cloud
                                )
                            
                            if mode == 'lidar3d':
                                visualize_point_cloud_3d(
                                    f'{args.path}/simbev/viz/LIDAR3D/' \
                                        f'SimBEV-scene-{scene_number:04d}-frame-{frame_number:04d}-LIDAR3D.jpg',
                                    point_cloud_3d,
                                    canvas,
                                    color=color
                                )
                            
                            if mode == 'lidar-with-bbox':
                                corners, labels = transform_bbox(gt_det, global2lidar)

                                visualize_point_cloud(
                                    f'{args.path}/simbev/viz/LIDARwBBOX/' \
                                        f'SimBEV-scene-{scene_number:04d}-frame-{frame_number:04d}-LIDARwBBOX.jpg',
                                    point_cloud,
                                    corners=corners,
                                    labels=labels
                                )
                            
                            if mode == 'lidar3d-with-bbox':
                                global2image = lidar2image @ global2lidar

                                corners, labels = transform_bbox(gt_det, global2image)

                                visualize_point_cloud_3d(
                                    f'{args.path}/simbev/viz/LIDAR3DwBBOX/' \
                                        f'SimBEV-scene-{scene_number:04d}-frame-{frame_number:04d}-LIDAR3DwBBOX.jpg',
                                    point_cloud_3d,
                                    canvas,
                                    corners=corners,
                                    labels=labels,
                                    color=color
                                )

                            if mode == 'semantic-lidar':
                                visualize_point_cloud(
                                    f'{args.path}/simbev/viz/SEG-LIDAR/' \
                                        f'SimBEV-scene-{scene_number:04d}-frame-{frame_number:04d}-SEG-LIDAR.jpg',
                                    point_cloud,
                                    color=label_color
                                )
                            
                            if mode == 'semantic-lidar3d':
                                visualize_point_cloud_3d(
                                    f'{args.path}/simbev/viz/SEG-LIDAR3D/' \
                                        f'SimBEV-scene-{scene_number:04d}-frame-{frame_number:04d}-SEG-LIDAR3D.jpg',
                                    point_cloud_3d,
                                    canvas,
                                    color=(label_color * 255.0).astype(np.uint8)
                                )
                            
                            if 'radar' in mode:
                                point_cloud = []
                                velocity = []
                                
                                for radar in RAD_NAME:
                                    # Radar to lidar transformation.
                                    radar2lidar = np.eye(4).astype(np.float32)

                                    radar2lidar[:3, :3] = Q(metadata[radar]['sensor2lidar_rotation']).rotation_matrix
                                    radar2lidar[:3, 3] = metadata[radar]['sensor2lidar_translation']

                                    radar_points = np.load(frame_data[radar])['data']

                                    velocity.append(radar_points[:, -1])

                                    radar_points = radar_points[:, :-1]

                                    # Convert the radar values of depth,
                                    # altitude angle, and azimuth angle to
                                    # x, y, and z coordinates.
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

                                # Calculate point cloud colors.
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
                            
                            if 'radar3d' in mode:
                                point_cloud_3d = (
                                    lidar2image @ np.append(point_cloud, np.ones((point_cloud.shape[0], 1)), 1).T
                                )[:3].T
                                
                                # Filter out points that are behind the 3D view camera.
                                indices = point_cloud_3d[:, 2] > 0.0
                                point_cloud_3d = point_cloud_3d[indices]
                                color = color[indices]

                                # Find the pixels corresponding to the point cloud.
                                point_cloud_3d[:, 2] = np.clip(point_cloud_3d[:, 2], a_min=1e-5, a_max=1e5)
                                point_cloud_3d[:, 0] /= point_cloud_3d[:, 2]
                                point_cloud_3d[:, 1] /= point_cloud_3d[:, 2]

                                # Set the 3D view camera dimensions.
                                width = camera_intrinsics[0, 2] * 2
                                height = camera_intrinsics[1, 2] * 2

                                canvas = np.zeros((int(height), int(width), 3), dtype=np.uint8)
                                canvas[:] = (255, 255, 255)

                                mask = np.logical_and.reduce([
                                    point_cloud_3d[:, 0] >= 0,
                                    point_cloud_3d[:, 0] < width,
                                    point_cloud_3d[:, 1] >= 0,
                                    point_cloud_3d[:, 1] < height
                                ])
                                point_cloud_3d = point_cloud_3d[mask]
                                color = color[mask]
                            
                            if mode == 'radar':
                                visualize_point_cloud(
                                    f'{args.path}/simbev/viz/RADAR/' \
                                        f'SimBEV-scene-{scene_number:04d}-frame-{frame_number:04d}-RADAR.jpg',
                                    point_cloud,
                                    color=color,
                                    radius=128
                                )
                            
                            if mode == 'radar3d':
                                visualize_point_cloud_3d(
                                    f'{args.path}/simbev/viz/RADAR3D/' \
                                        f'SimBEV-scene-{scene_number:04d}-frame-{frame_number:04d}-RADAR3D.jpg',
                                    point_cloud_3d,
                                    canvas,
                                    color=(color * 255.0).astype(np.uint8)
                                )
                            
                            if mode == 'radar-with-bbox':
                                corners, labels = transform_bbox(gt_det, global2lidar)

                                visualize_point_cloud(
                                    f'{args.path}/simbev/viz/RADARwBBOX/' \
                                        f'SimBEV-scene-{scene_number:04d}-frame-{frame_number:04d}-RADARwBBOX.jpg',
                                    point_cloud,
                                    corners=corners,
                                    labels=labels,
                                    color=color,
                                    radius=128
                                )

                            if mode == 'radar3d-with-bbox':
                                global2image = lidar2image @ global2lidar

                                corners, labels = transform_bbox(gt_det, global2image)

                                visualize_point_cloud_3d(
                                    f'{args.path}/simbev/viz/RADAR3DwBBOX/' \
                                        f'SimBEV-scene-{scene_number:04d}-frame-{frame_number:04d}-RADAR3DwBBOX.jpg',
                                    point_cloud_3d,
                                    canvas,
                                    corners=corners,
                                    labels=labels,
                                    color=(color * 255.0).astype(np.uint8)
                                )

    except Exception:
        print(traceback.format_exc())

        print('Killing the process...')

        time.sleep(3.0)


if __name__ == '__main__':
    try:
        os.makedirs(f'{args.path}/simbev/viz', exist_ok=True)

        if 'all' in args.mode:
            mode_list = [
                'rgb',
                'depth',
                'flow',
                'lidar',
                'lidar3d',
                'lidar-with-bbox',
                'lidar3d-with-bbox',
                'semantic-lidar',
                'semantic-lidar3d',
                'radar',
                'radar3d',
                'radar-with-bbox',
                'radar3d-with-bbox'
            ]
        else:
            mode_list = args.mode

        for mode in mode_list:
            main(mode)
    
    except KeyboardInterrupt:
        print('Killing the process...')

        time.sleep(3.0)
    
    finally:
        print('Done.')
