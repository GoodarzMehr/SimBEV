# Academic Software License: Copyright © 2026 Goodarz Mehr.

'''
SimBEV utility tools.
'''

import os
import cv2
import time
import carla
import psutil
import signal
import logging

import numpy as np

from tqdm import tqdm


def is_used(port: int) -> bool:
    '''
    Check whether or not a port is used.

    Args:
        port: port number.
    
    Returns:
        True if the port is being used, False otherwise.
    '''
    return port in [conn.laddr.port for conn in psutil.net_connections()]

def kill_all_servers():
    '''Kill all PIDs that start with CARLA.'''
    processes = [p for p in psutil.process_iter() if 'carla' in p.name().lower()]
    
    for process in processes:
        os.kill(process.pid, signal.SIGKILL)

def carla_vector_to_numpy(vector_list: list[carla.Vector3D]) -> np.ndarray:
    '''
    Convert a list of CARLA vectors to a NumPy array.

    Args:
        vector_list: list of CARLA vectors.
    
    Returns:
        vector_array: NumPy array of vectors.
    '''
    vector_array = np.zeros((len(vector_list), 3))

    for i, vector in enumerate(vector_list):
        vector_array[i, 0] = vector.x
        vector_array[i, 1] = vector.y
        vector_array[i, 2] = vector.z

    return vector_array

def carla_single_vector_to_numpy(vector: carla.Vector3D) -> np.ndarray:
    '''
    Convert a single CARLA vector to a NumPy array.

    Args:
        vector: CARLA vector.
    
    Returns:
        NumPy array of the vector.
    '''
    return np.array([vector.x, vector.y, vector.z])

def carla_single_rotation_to_numpy(rotation: carla.Rotation) -> np.ndarray:
    '''
    Convert a single CARLA rotation to a NumPy array.

    Args:
        rotation: CARLA rotation.
    
    Returns:
        NumPy array of the rotation.
    '''
    return np.array([rotation.roll, rotation.pitch, rotation.yaw])

def local_to_global(location: carla.Location, rotation: carla.Rotation) -> np.ndarray:
    '''
    Calculate the transformation from a local CARLA coordinate system to the
    global coordinate system. Local coordinates are first transformed from
    CARLA's left-handed coordinate system to a right-handed one. The global
    coordinate system is a right-handed system.

    Args:
        location: local coordinate system origin location.
        rotation: local coordinate system origin rotation.

    Returns:
        R: transformation matrix.
    '''
    x = location.x
    y = -location.y
    theta = -np.deg2rad(rotation.yaw)

    return np.array([[np.cos(theta), -np.sin(theta), x],
                     [np.sin(theta), np.cos(theta), y],
                     [0, 0, 1]])

def get_multi_polygon_mask(
        polygons: list[np.ndarray],
        ego_tf: carla.Transform,
        xDim: int,
        xRes: float,
        yDim: int = None,
        yRes: float = None
    ) -> np.ndarray:
    '''
    Calculate a BEV mask from the given polygons with global coordinates.

    Args:
        polygons: list of polygons to create a mask from.
        ego_tf: ego vehicle transform.
        xDim: BEV grid width.
        xRes: BEV grid width resolution.
        yDim: BEV grid height.
        yRes: BEV grid height resolution.
    
    Returns:
        mask: polygon mask.
    '''
    if yDim is None:
        yDim = xDim
    
    if yRes is None:
        yRes = xRes

    if len(polygons) == 0:
        return np.zeros((xDim, yDim), dtype=bool)

    ego_loc = ego_tf.location
    ego_rot = ego_tf.rotation
    
    # Calculate the transformation from the global coordinates to the ego
    # vehicle's local coordinates.
    R = np.linalg.inv(local_to_global(ego_loc, ego_rot))

    # Calculate the grid bounds.
    xLim = xDim * xRes / 2
    yLim = yDim * yRes / 2
    
    # Create an empty polygon mask.
    polygon_mask = np.zeros((xDim, yDim), dtype=np.uint8)

    # Process each polygon.
    for polygon in polygons:
        polygon[:, 2] = 1
        local_polygon = (R @ polygon.T)[:2].T

        # Convert to grid indices.
        grid_x = ((xLim - local_polygon[:, 1]) / xRes).astype(np.int32)
        grid_y = ((yLim - local_polygon[:, 0]) / yRes).astype(np.int32)

        # Create polygon points
        poly = np.column_stack([grid_x, grid_y])

        if len(poly) > 2:
            cv2.fillPoly(polygon_mask, [poly], 1)

    return polygon_mask.astype(bool)

def get_multi_line_mask(
        lines: list[np.ndarray],
        ego_tf: carla.Transform,
        xDim: int,
        xRes: float,
        yDim: int = None,
        yRes: float = None
    ) -> np.ndarray:
    '''
    Calculate a BEV mask from the given lines with global coordinates.

    Args:
        lines: list of lines to create a mask from.
        ego_tf: ego vehicle transform.
        xDim: BEV grid width.
        xRes: BEV grid width resolution.
        yDim: BEV grid height.
        yRes: BEV grid height resolution.
    
    Returns:
        mask: line mask.
    '''
    if yDim is None:
        yDim = xDim
    
    if yRes is None:
        yRes = xRes

    if len(lines) == 0:
        return np.zeros((xDim, yDim), dtype=bool)
    
    ego_loc = ego_tf.location
    ego_rot = ego_tf.rotation

    # Calculate the transformation from the global coordinates to the ego
    # vehicle's local coordinates.
    R = np.linalg.inv(local_to_global(ego_loc, ego_rot))

    # Calculate grid bounds.
    xLim = xDim * xRes / 2
    yLim = yDim * yRes / 2

    # Create an empty line mask.
    line_mask = np.zeros((xDim, yDim), dtype=np.uint8)

    lines = np.vstack(lines)
    
    lines[:, 2] = 1
    local_lines = (R @ lines.T)[:2].T

    # Convert to grid indices.
    grid_x = ((xLim - local_lines[:, 0]) / xRes).astype(np.int32)
    grid_y = ((yLim - local_lines[:, 1]) / yRes).astype(np.int32)

    # Remove the points that are outside the grid.
    boundary_mask = (grid_x < 0) | (grid_x >= xDim) | (grid_y < 0) | (grid_y >= yDim)

    line_mask[grid_x[~boundary_mask], grid_y[~boundary_mask]] = 1

    return line_mask.astype(bool)

def flow_to_color(flow: np.ndarray) -> np.ndarray:
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


class CustomTimer:
    '''
    Timer class that uses a performance counter if available, otherwise time
    in seconds.
    '''
    def __init__(self):
        try:
            self.timer = time.perf_counter
        except AttributeError:
            self.timer = time.time

    def time(self):
        return self.timer()


class TqdmLoggingHandler(logging.StreamHandler):
    '''Logging handler for tqdm progress bars.'''
    def emit(self, record):
        '''
        Emit a log record.

        Args:
            record: log record.
        '''
        try:
            msg = self.format(record)
            
            tqdm.write(msg)
            
            self.flush()
        
        except Exception:
            self.handleError(record)