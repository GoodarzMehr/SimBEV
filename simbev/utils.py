# Academic Software License: Copyright © 2025 Goodarz Mehr.

'''
SimBEV utility tools.
'''

import os
import time
import torch
import psutil
import signal

import numpy as np

def is_used(port):
    '''
    Check whether or not a port is used.

    Args:
        port: port number.
    
    Return:
        True if the port is being used, False otherwise.
    '''
    return port in [conn.laddr.port for conn in psutil.net_connections()]

def kill_all_servers():
    '''
    Kill all PIDs that start with CARLA.
    '''
    processes = [p for p in psutil.process_iter() if 'carla' in p.name().lower()]
    
    for process in processes:
        os.kill(process.pid, signal.SIGKILL)

def carla_vector_to_numpy(vector_list):
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

def carla_single_vector_to_numpy(vector):
    '''
    Convert a single CARLA vector to a NumPy array.

    Args:
        vector: CARLA vector.
    
    Returns:
        vector_array: NumPy array of the vector.
    '''

    return np.array([vector.x, vector.y, vector.z])

def carla_vector_to_torch(vector_list):
    '''
    Convert a list of CARLA vectors to a Torch tensor.

    Args:
        vector_list: list of CARLA vectors.
    
    Returns:
        vector_array: Torch tensor of vectors.
    '''
    vector_array = torch.zeros((len(vector_list), 3))

    for i, vector in enumerate(vector_list):
        vector_array[i, 0] = vector.x
        vector_array[i, 1] = vector.y
        vector_array[i, 2] = vector.z

    return vector_array

def local_to_global(location, rotation):
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

def get_road_mask(
        waypoints,
        lane_widths,
        ego_loc,
        ego_rot,
        xDim,
        xRes,
        yDim=None,
        yRes=None,
        device='cuda:0',
        dType=torch.float
    ):
    '''
    Calculate the road mask for the BEV grid using map-generated waypoints.

    Args:
        waypoints: array of map-generated waypoints.
        lane_widths: array of corresponding waypoint lane widths.
        ego_loc: ego vehicle location.
        ego_rot: ego vehicle rotation.
        xDim: BEV grid width.
        xRes: BEV grid width resolution.
        yDim: BEV grid height.
        yRes: BEV grid height resolution.
        device: device to use for computation, can be 'cpu' or 'cuda:i' where
            i is the GPU index.
        dType: data type to use for calculations.
    
    Returns:
        mask: BEV grid's road mask.
    '''
    if device == 'cpu':
        nSlice = 1
    elif dType == torch.float:
        nSlice = 8
    else:
        nSlice = 32

    if yDim is None:
        yDim = xDim
    
    if yRes is None:
        yRes = xRes
    
    if not isinstance(waypoints, torch.Tensor):
        waypoints = torch.from_numpy(waypoints).to(device, dType)
    else:
        waypoints = waypoints.to(device, dType)

    if not isinstance(lane_widths, torch.Tensor):
        lane_widths = torch.from_numpy(lane_widths).to(device, dType)
    else:
        lane_widths = lane_widths.to(device, dType)
    
    # Calculate the transformation from the global coordinate system to that
    # of the ego vehicle.
    R = torch.inverse(torch.from_numpy(local_to_global(ego_loc, ego_rot))).to(device, dType)

    # Flip the y coordinates because CARLA uses a left-handed coordinate
    # system and we use a right-handed one, and set the z coordinates to 1.
    waypoints[:, 1] *= -1
    waypoints[:, 2] = 1

    # Transform the waypoint coordinates into the ego vehicle's local
    # coordinate system.
    local_waypoints = (R @ waypoints.T)[:2].T

    # Calculate the center-point coordinates of the BEV grid cells.
    xLim = xDim * xRes / 2
    yLim = yDim * yRes / 2
    
    cxLim = xLim - xRes / 2
    cyLim = yLim - yRes / 2

    x = torch.linspace(cxLim, -cxLim, xDim, dtype=dType, device=device)
    y = torch.linspace(cyLim, -cyLim, yDim, dtype=dType, device=device)

    xx, yy = torch.meshgrid(x, y, indexing='ij')

    coordinates = torch.stack([xx, yy], dim=2).reshape(-1, 2)

    # Calculate the pair-wise distance between the center-point of each BEV
    # grid cell and each local waypoint. Then compare it to that waypoint's
    # lane width to determine if the center-point is inside a circle centered
    # at the waypoint with a diameter equal to the lane width.
    sliceW = (xDim * yDim) // nSlice
    nSlice = (xDim * yDim) // sliceW if (xDim * yDim) % sliceW == 0 else (xDim * yDim) // sliceW + 1

    for slice in range(nSlice):
        start = slice * sliceW
        end = (slice + 1) * sliceW if slice != nSlice - 1 else xDim * yDim

        dist_slice = torch.cdist(coordinates[start:end], local_waypoints)

        mask_slice = dist_slice < (lane_widths / 2)

        if slice == 0:
            mask = torch.any(mask_slice, dim=1)
        else:
            mask = torch.cat((mask, torch.any(mask_slice, dim=1)), dim=0)

    return mask.reshape(xDim, yDim).detach().cpu()


def get_object_mask(
        bbox,
        ego_loc,
        ego_rot,
        xDim,
        xRes,
        yDim=None,
        yRes=None,
        device='cuda:0',
        dType=torch.float
    ):
    '''
    Get a certain object's BEV grid mask using its bounding box coordinates.

    Args:
        bbox: object bounding box.
        ego_loc: ego vehicle location.
        ego_rot: ego vehicle rotation.
        xDim: BEV grid width.
        xRes: BEV grid width resolution.
        yDim: BEV grid height.
        yRes: BEV grid height resolution.
        device: device to use for computation, can be 'cpu' or 'cuda:i' where
            i is the GPU index.
        dType: data type to use for calculations.
    
    Returns:
        mask: the object's BEV grid mask.
    '''
    if yDim is None:
        yDim = xDim
    
    if yRes is None:
        yRes = xRes
    
    if not isinstance(bbox, torch.Tensor):
        bbox = torch.from_numpy(bbox).to(device, dType)
    else:
        bbox = bbox.to(device, dType)

    bbox[:, 2] = 1

    # Calculate the transformation from the global coordinate system to that
    # of the ego vehicle.
    R = torch.inverse(torch.from_numpy(local_to_global(ego_loc, ego_rot))).to(device, dType)

    # Transform the bounding box coordinates into the ego vehicle's local
    # coordinate system.
    local_bbox = (R @ bbox.T)[:2].T

    # Calculate the center-point coordinates of the BEV grid cells.
    xLim = xDim * xRes / 2
    yLim = yDim * yRes / 2

    cxLim = xLim - xRes / 2
    cyLim = yLim - yRes / 2

    x = torch.linspace(cxLim, -cxLim, xDim, dtype=dType, device=device)
    y = torch.linspace(cyLim, -cyLim, yDim, dtype=dType, device=device)

    xx, yy = torch.meshgrid(x, y, indexing='ij')

    coordinates = torch.stack([xx, yy], dim=2).reshape(-1, 2)

    # For each bounding box edge, find BEV grid cell center-points that are
    # on the right side of that edge. The object's BEV grid mask is the
    # intersection of these masks, or its complement.
    mask1 = is_on_right_side(coordinates[:, 0], coordinates[:, 1], local_bbox[0], local_bbox[2])
    mask2 = is_on_right_side(coordinates[:, 0], coordinates[:, 1], local_bbox[2], local_bbox[6])
    mask3 = is_on_right_side(coordinates[:, 0], coordinates[:, 1], local_bbox[6], local_bbox[4])
    mask4 = is_on_right_side(coordinates[:, 0], coordinates[:, 1], local_bbox[4], local_bbox[0])

    mask = (mask1 & mask2 & mask3 & mask4) | (~mask1 & ~mask2 & ~mask3 & ~mask4)

    return mask.reshape(xDim, yDim).detach().cpu()

def get_crosswalk_mask(
        crosswalk,
        ego_loc,
        ego_rot,
        xDim,
        xRes,
        yDim=None,
        yRes=None,
        device='cuda:0',
        dType=torch.float
    ):
    '''
    Get a crosswalk's BEV grid mask using the coordinates of its corners.

    Args:
        crosswalk: coordinates of the crosswalk's corners.
        ego_loc: ego vehicle location.
        ego_rot: ego vehicle rotation.
        xDim: BEV grid width.
        xRes: BEV grid width resolution.
        yDim: BEV grid height.
        yRes: BEV grid height resolution.
        device: device to use for computation, can be 'cpu' or 'cuda:i' where
            i is the GPU index.
        dType: data type to use for calculations.
    
    Returns:
        mask: the crosswalk's BEV grid mask.
    '''
    if yDim is None:
        yDim = xDim
    
    if yRes is None:
        yRes = xRes

    if not isinstance(crosswalk, torch.Tensor):
        crosswalk = torch.from_numpy(crosswalk).to(device, dType)
    else:
        crosswalk = crosswalk.to(device, dType)

    crosswalk[:, 2] = 1
    
    # Calculate the transformation from the global coordinate system to that
    # of the ego vehicle.
    R = torch.inverse(torch.from_numpy(local_to_global(ego_loc, ego_rot))).to(device, dType)

    # Transform the crosswalk coordinates into the ego vehicle's local
    # coordinate system.
    local_box = (R @ crosswalk.T)[:2].T

    # Calculate the center-point coordinates of the BEV grid cells.
    xLim = xDim * xRes / 2
    yLim = yDim * yRes / 2

    cxLim = xLim - xRes / 2
    cyLim = yLim - yRes / 2

    x = torch.linspace(cxLim, -cxLim, xDim, dtype=dType, device=device)
    y = torch.linspace(cyLim, -cyLim, yDim, dtype=dType, device=device)

    xx, yy = torch.meshgrid(x, y, indexing='ij')

    coordinates = torch.stack([xx, yy], dim=2).reshape(-1, 2)

    mask = torch.zeros(coordinates.shape[0], crosswalk.shape[0] - 1, device=device, dtype=torch.bool)

    # For each crosswalk edge, find BEV grid cell center-points that are
    # on the right side of that edge. The crosswalk's BEV grid mask is the
    # intersection of these masks, or its complement.
    for i in range(crosswalk.shape[0] - 1):
        mask[:, i] = is_on_right_side(coordinates[:, 0], coordinates[:, 1], local_box[i], local_box[i + 1])

    mask = torch.all(mask, dim=1) | torch.all(~mask, dim=1)

    return mask.reshape(xDim, yDim).detach().cpu()

def is_on_right_side(x, y, xy0, xy1):
    '''
    Determine if a point (or array of points) is on the right side of a line
    defined by two points.

    Args:
        x: x coordinate(s) of the point(s).
        y: y coordinate(s) of the point(s).
        xy0: first point of the line.
        xy1: second point of the line.

    Returns:
        mask: mask indicating if the point(s) are on the right side of the
            line defined by the two points.
    '''
    x0, y0 = xy0
    x1, y1 = xy1
    
    a = float(y1 - y0)
    b = float(x0 - x1)
    
    c = -a * x0 - b * y0
    
    return a * x + b * y + c >= 0

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