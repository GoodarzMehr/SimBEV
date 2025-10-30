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

def transform_bbox(gt_det: list, transform: np.ndarray, ignore_valid_flag: bool = False):
    '''
    Transform bounding boxes from the global coordinate system to the desired
    coordinate system.

    Args:
        gt_det: list of 3D object bounding boxes.
        transform: coordinate transformation matrix.
        ignore_valid_flag: whether to ignore valid_flag when transforming
            the bounding boxes.
    
    Returns:
        corners: array of transformed bounding box corners.
        labels: array of object class labels.
    '''
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

def visualize_image(
        fpath: str,
        image: np.ndarray,
        corners: np.ndarray = None,
        labels: np.ndarray = None,
        color: tuple = None,
        thickness: int = 1
    ):
    '''
    Visualize image with bounding boxes.

    Args:
        fpath: file path for saving the image.
        image: image to visualize.
        corners: array of bounding box corners.
        labels: array of bounding box labels.
        color: bounding box color.
        thickness: bounding box line thickness.
    '''
    canvas = image.copy()

    if corners is not None and corners.shape[0] > 0:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        
        canvas = draw_bbox(canvas, corners, labels, bbox_color=color, thickness=thickness)
        
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    cv2.imwrite(fpath, canvas)

def draw_bbox(
        canvas: np.ndarray,
        corners: np.ndarray,
        labels: np.ndarray,
        bbox_color: tuple = None,
        thickness: int = 1
    ):
    '''
    Draw bounding boxes on the canvas.

    Args:
        canvas: canvas to draw on.
        corners: array of bounding box corners.
        labels: array of bounding box labels.
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

    # Draw bounding box lines
    if corners is not None:
        for index in range(corners.shape[0]):
            name = labels[index]
            
            for start, end in [
                (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)
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

def flow_to_color(flow):
    '''
    Optical flow visualization.
    
    Args:
        flow: array of flow vectors.
    
    Returns:
        BGR image.
    '''
    # Compute the magnitude and angle of flow vectors.
    mag, ang = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])
    
    # Create HSV image
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)

    hsv[:, :, 0] = ang * 180 / (np.pi * 2)
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def visualize_point_cloud(
        fpath: str,
        point_cloud: np.ndarray,
        corners: np.ndarray = None,
        labels: np.ndarray = None,
        color: np.ndarray = None,
        xlim: tuple = (-80, 80),
        ylim: tuple = (-80, 80),
        radius: int = 16,
        thickness: int = 1
    ):
    '''
    Visualize point cloud with bounding boxes from above.

    Args:
        fpath: file path to save the image.
        point_cloud: point cloud to visualize.
        corners: array of bounding box corners.
        labels: array of bounding box labels.
        color: array of point cloud color(s).
        xlim: x-axis limits.
        ylim: y-axis limits.
        radius: display point radius.
        thickness: bounding box line thickness.
    '''
    pixels_per_meter = 16
    
    width = int((xlim[1] - xlim[0]) * pixels_per_meter)
    height = int((ylim[1] - ylim[0]) * pixels_per_meter)
    
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Remove the points outside image bounds.
    mask = (
        (point_cloud[:, 0] >= xlim[0]) & (point_cloud[:, 0] < xlim[1]) &
        (point_cloud[:, 1] >= ylim[0]) & (point_cloud[:, 1] < ylim[1])
    )

    point_cloud_filtered = point_cloud[mask]
    
    if len(point_cloud_filtered) > 0:
        # Convert to pixels.
        px = ((point_cloud_filtered[:, 0] - xlim[0]) * pixels_per_meter).astype(np.int32)
        py = ((ylim[1] - point_cloud_filtered[:, 1]) * pixels_per_meter).astype(np.int32)
        
        # Clip to the valid range.
        valid = (px >= 0) & (px < width) & (py >= 0) & (py < height)
        
        px = px[valid]
        py = py[valid]
        
        # Compute the colors.
        if color is None:
            distances = np.linalg.norm(point_cloud_filtered[valid], axis=1)
            
            log_distances = np.log(distances + 1e-6)
            
            log_normalized = (log_distances - log_distances.min()) / \
                (log_distances.max() - log_distances.min() + 1e-6)
            
            point_colors = np.c_[
                np.interp(log_normalized, RANGE, RAINBOW[:, 2]),
                np.interp(log_normalized, RANGE, RAINBOW[:, 1]),
                np.interp(log_normalized, RANGE, RAINBOW[:, 0])
            ] * 255
        else:
            color_filtered = color[mask][valid]
            
            if color_filtered.max() <= 1.0:
                color_filtered = color_filtered * 255
            
            point_colors = color_filtered[:, ::-1]
        
        point_colors = point_colors.astype(np.uint8)
        
        # Directly assign colors to pixels.
        point_radius = max(1, radius // 16)
        
        if point_radius == 1:
            # Direct pixel assignment (fastest).
            canvas[py, px] = point_colors
        else:
            # Draw circles (slower but supports larger points).
            for i in range(len(px)):
                cv2.circle(canvas, (px[i], py[i]), point_radius, point_colors[i].tolist(), -1)
    
    # Draw bounding boxes.
    if corners is not None:
        coords = corners[:, [0, 2, 6, 4], :2]
    
        for index in range(coords.shape[0]):
            name = labels[index]
            
            bbox_color = tuple([int(c) for c in SIMBEV_PALETTE[name]][::-1])
            
            # Convert to pixels.
            bbox_px = ((coords[index, :, 0] - xlim[0]) * pixels_per_meter).astype(np.int32)
            bbox_py = ((ylim[1] - coords[index, :, 1]) * pixels_per_meter).astype(np.int32)
            
            # Draw polygon (faster than individual lines).
            points = np.column_stack([bbox_px, bbox_py])
            
            cv2.polylines(
                canvas,
                [points],
                isClosed=True,
                color=bbox_color,
                thickness=max(1, thickness),
                lineType=cv2.LINE_AA
            )
    
    cv2.imwrite(fpath, canvas)

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