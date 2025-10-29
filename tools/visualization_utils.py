# Academic Software License: Copyright © 2025 Goodarz Mehr.

import cv2

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import colormaps as cm
from pyquaternion import Quaternion as Q


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


def parse_range_argument(arg_list: list) -> list:
    '''
    Parse range arguments like [1, 2, '4-6', 9, '10-12'] into a list of
    integers.
    
    Args:
        arg_list: list of integers and/or strings with ranges (e.g., '4-6')
    
    Returns:
        sorted list of unique integers.
    '''
    result = set()
    
    for item in arg_list:
        if isinstance(item, int):
            result.add(item)
        elif isinstance(item, str):
            if '-' in item:
                parts = item.split('-')
                
                if len(parts) == 2:
                    try:
                        start = int(parts[0])
                        end = int(parts[1])
                        
                        if start <= end:
                            result.update(range(start, end + 1))
                        else:
                            print(f'Warning: Invalid range "{item}" (start > end), skipping.')
                    
                    except ValueError:
                        print(f'Warning: Invalid range "{item}", skipping.')
                else:
                    print(f'Warning: Invalid range format "{item}", skipping.')
            else:
                try:
                    result.add(int(item))
                except ValueError:
                    print(f'Warning: Invalid value "{item}", skipping.')
        else:
            print(f'Warning: Unexpected type {type(item)} for value {item}, skipping.')
    
    return sorted(result)

def get_global2sensor(frame_data: dict, metadata: dict, sensor_name='LIDAR'):
    '''
    Get the global2sensor transformation matrix.
    
    Args:
        frame_data: data contained in the frame.
        metadata: dataset metadata.
        sensor_name: name of the sensor.
    '''
    # Ego to global transformation.
    ego2global = np.eye(4, dtype=np.float32)

    ego2global[:3, :3] = Q(frame_data['ego2global_rotation']).rotation_matrix
    ego2global[:3, 3] = frame_data['ego2global_translation']

    # Sensor to ego transformation.
    sensor2ego = np.eye(4, dtype=np.float32)

    sensor2ego[:3, :3] = Q(metadata[sensor_name]['sensor2ego_rotation']).rotation_matrix
    sensor2ego[:3, 3] = metadata[sensor_name]['sensor2ego_translation']

    # Global to sensor transformation.
    global2sensor = np.linalg.inv(ego2global @ sensor2ego)

    return global2sensor

def draw_bbox(canvas, corners, labels, bbox_color=None, thickness=1):
    '''Draw bounding boxes on the canvas.'''
    # Filter out bounding boxes that are behind the camera
    indices = np.all(corners[..., 2] > 0, axis=1)
    corners = corners[indices]
    labels = labels[indices]

    # Sort bounding boxes by their distance to the camera
    indices = np.argsort(-np.min(corners[..., 2], axis=1))
    corners = corners[indices]
    labels = labels[indices]

    # Find the pixels corresponding to bounding box corners
    corners = corners.reshape(-1, 3)
    corners[:, 2] = np.clip(corners[:, 2], a_min=1e-5, a_max=1e5)
    corners[:, 0] /= corners[:, 2]
    corners[:, 1] /= corners[:, 2]

    corners = corners[..., :2].reshape(-1, 8, 2)

    # Draw bounding box lines
    if corners is not None:
        for index in range(corners.shape[0]):
            name = labels[index]
            
            for start, end in [
                (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
                (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)
            ]:
                cv2.line(
                    canvas,
                    corners[index, start].astype(np.int32),
                    corners[index, end].astype(np.int32),
                    bbox_color or SIMBEV_PALETTE[name],
                    thickness,
                    cv2.LINE_AA,
                )

        canvas = canvas.astype(np.uint8)
    
    return canvas

def visualize_image(fpath, image, corners=None, labels=None, bbox_color=None, thickness=2):
    '''Visualize image with bounding boxes.'''
    canvas = image.copy()

    if corners is not None and corners.shape[0] > 0:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        canvas = draw_bbox(canvas, corners, labels, bbox_color=bbox_color, thickness=thickness)
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

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
    '''Visualize point cloud from above with bounding boxes.'''
    fig = plt.figure(figsize=(xlim[1] - xlim[0], ylim[1] - ylim[0]))
    ax = plt.gca()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect(1)
    ax.set_axis_off()

    # Filter out points that are outside the limits
    mask = np.logical_and.reduce([
        point_cloud[:, 0] >= xlim[0],
        point_cloud[:, 0] < xlim[1],
        point_cloud[:, 1] >= ylim[0],
        point_cloud[:, 1] < ylim[1]
    ])
    point_cloud = point_cloud[mask]

    # Plot the point cloud
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
    
    # Draw bounding box lines
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
    '''Visualize point cloud in 3D with bounding boxes.'''
    if color is not None:
        canvas[point_cloud[:, 1].astype(int), point_cloud[:, 0].astype(int), :] = color

    if corners is not None and corners.shape[0] > 0:
        canvas = draw_bbox(canvas, corners, labels, bbox_color=bbox_color, thickness=thickness)
    
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    cv2.imwrite(fpath, canvas)

def transform_bbox(gt_det, transform, ignore_valid_flag=False):
    '''Transform bounding boxes from global to desired coordinate system.'''
    corners = []
    labels = []

    for det_object in gt_det:
        if ignore_valid_flag or det_object['valid_flag']:
            for tag in det_object['semantic_tags']:
                if tag in OBJECT_CLASSES.keys():
                    global_bbox_corners = np.append(det_object['bounding_box'], np.ones((8, 1)), 1)
                    bbox_corners = (transform @ global_bbox_corners.T)[:3].T

                    corners.append(bbox_corners)
                    labels.append(OBJECT_CLASSES[tag])

    corners = np.array(corners)
    labels = np.array(labels)

    return corners, labels

def get_3d_view_transforms(metadata):
    '''Get 3D view camera transformations.'''
    view2lidar = np.eye(4, dtype=np.float32)
    view2lidar[:3, :3] = Q([0.415627, -0.572061, 0.572061, -0.415627]).rotation_matrix
    view2lidar[:3, 3] = [-40.0, 0.0, 12.0]

    lidar2view = np.linalg.inv(view2lidar)

    camera_intrinsics = np.eye(4, dtype=np.float32)
    camera_intrinsics[:3, :3] = metadata['camera_intrinsics']

    lidar2image = camera_intrinsics @ lidar2view

    return lidar2image, camera_intrinsics

def compute_rainbow_colors(values):
    '''Compute rainbow colors for given values.'''
    log_values = np.log(values)
    log_values_normalized = (
        log_values - log_values.min()
    ) / (
        log_values.max() - log_values.min() + 1e-6
    )

    color = np.c_[
        np.interp(log_values_normalized, RANGE, RAINBOW[:, 0]),
        np.interp(log_values_normalized, RANGE, RAINBOW[:, 1]),
        np.interp(log_values_normalized, RANGE, RAINBOW[:, 2])
    ]
    
    return color

def project_to_3d_view(point_cloud, lidar2image, camera_intrinsics):
    '''Project point cloud to 3D view and filter.'''
    point_cloud_3d = (
        lidar2image @ np.append(point_cloud, np.ones((point_cloud.shape[0], 1)), 1).T
    )[:3].T
    
    # Filter out points behind camera
    indices = point_cloud_3d[:, 2] > 0.0
    point_cloud_3d = point_cloud_3d[indices]

    # Project to image coordinates
    point_cloud_3d[:, 2] = np.clip(point_cloud_3d[:, 2], a_min=1e-5, a_max=1e5)
    point_cloud_3d[:, 0] /= point_cloud_3d[:, 2]
    point_cloud_3d[:, 1] /= point_cloud_3d[:, 2]

    # Get canvas dimensions
    width = camera_intrinsics[0, 2] * 2
    height = camera_intrinsics[1, 2] * 2

    # Create canvas
    canvas = np.zeros((int(height), int(width), 3), dtype=np.uint8)
    canvas[:] = (255, 255, 255)

    # Filter points within canvas bounds
    mask = np.logical_and.reduce([
        point_cloud_3d[:, 0] >= 0,
        point_cloud_3d[:, 0] < width,
        point_cloud_3d[:, 1] >= 0,
        point_cloud_3d[:, 1] < height
    ])
    point_cloud_3d = point_cloud_3d[mask]

    return point_cloud_3d, canvas, indices, mask