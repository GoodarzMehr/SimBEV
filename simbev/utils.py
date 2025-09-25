# Academic Software License: Copyright © 2025 Goodarz Mehr.

'''
SimBEV utility tools.
'''

import os
import cv2
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
    
    Returns:
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

    # coordinates = torch.stack([xx, yy], dim=2).reshape(-1, 2)

    # mask = torch.zeros(coordinates.shape[0], crosswalk.shape[0] - 1, device=device, dtype=torch.bool)

    # # For each crosswalk edge, find BEV grid cell center-points that are
    # # on the right side of that edge. The crosswalk's BEV grid mask is the
    # # intersection of these masks, or its complement.
    # for i in range(crosswalk.shape[0] - 1):
    #     mask[:, i] = is_on_right_side(coordinates[:, 0], coordinates[:, 1], local_box[i], local_box[i + 1])

    # mask = torch.all(mask, dim=1) | torch.all(~mask, dim=1)

    mask = point_in_polygon_vectorized(xx.flatten(), yy.flatten(), local_box)

    return mask.reshape(xDim, yDim).detach().cpu()

def point_in_polygon_vectorized(x, y, polygon):
    '''
    Vectorized point-in-polygon test using ray casting algorithm.
    
    Args:
        x: x coordinates of points to test
        y: y coordinates of points to test  
        polygon: polygon vertices as tensor of shape (n, 2)
    
    Returns:
        mask: boolean mask indicating points inside polygon
    '''
    n_points = x.shape[0]
    n_vertices = polygon.shape[0]
    
    # Prepare polygon edges
    x1 = polygon[:-1, 0].unsqueeze(0).expand(n_points, -1)  # (n_points, n_edges)
    y1 = polygon[:-1, 1].unsqueeze(0).expand(n_points, -1)
    x2 = polygon[1:, 0].unsqueeze(0).expand(n_points, -1)
    y2 = polygon[1:, 1].unsqueeze(0).expand(n_points, -1)
    
    # Expand point coordinates
    px = x.unsqueeze(1).expand(-1, n_vertices - 1)  # (n_points, n_edges)
    py = y.unsqueeze(1).expand(-1, n_vertices - 1)
    
    # Ray casting algorithm - count intersections
    cond1 = (y1 > py) != (y2 > py)
    cond2 = px < (x2 - x1) * (py - y1) / (y2 - y1) + x1
    
    intersections = (cond1 & cond2).sum(dim=1)
    
    return intersections % 2 == 1

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

def get_multiple_crosswalk_masks(
        crosswalks,  # List of crosswalk coordinate arrays
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
    Get multiple crosswalk BEV grid masks simultaneously.
    
    Args:
        crosswalks: list of crosswalk coordinate arrays
        ... (other args same as get_crosswalk_mask)
    
    Returns:
        masks: tensor of shape (n_crosswalks, xDim, yDim)
    '''
    if yDim is None:
        yDim = xDim
    
    if yRes is None:
        yRes = xRes

    if not crosswalks:
        return torch.zeros(0, xDim, yDim)

    # Transform all crosswalks to local coordinates
    R = torch.inverse(torch.from_numpy(local_to_global(ego_loc, ego_rot))).to(device, dType)
    
    local_crosswalks = []
    max_vertices = 0
    
    for crosswalk in crosswalks:
        if not isinstance(crosswalk, torch.Tensor):
            crosswalk = torch.from_numpy(crosswalk).to(device, dType)
        else:
            crosswalk = crosswalk.to(device, dType)
        
        crosswalk[:, 2] = 1
        local_box = (R @ crosswalk.T)[:2].T
        local_crosswalks.append(local_box)
        max_vertices = max(max_vertices, local_box.shape[0])

    # Create padded tensor for batch processing
    n_crosswalks = len(crosswalks)
    polygons_tensor = torch.zeros(n_crosswalks, max_vertices, 2, device=device, dtype=dType)
    
    for i, polygon in enumerate(local_crosswalks):
        n_verts = polygon.shape[0]
        polygons_tensor[i, :n_verts] = polygon
        # Pad shorter polygons by repeating the last vertex
        if n_verts < max_vertices:
            polygons_tensor[i, n_verts:] = polygon[-1].unsqueeze(0).expand(max_vertices - n_verts, -1)

    # Calculate grid coordinates
    xLim = xDim * xRes / 2
    yLim = yDim * yRes / 2
    cxLim = xLim - xRes / 2
    cyLim = yLim - yRes / 2

    x = torch.linspace(cxLim, -cxLim, xDim, dtype=dType, device=device)
    y = torch.linspace(cyLim, -cyLim, yDim, dtype=dType, device=device)
    xx, yy = torch.meshgrid(x, y, indexing='ij')

    # Test all crosswalks at once using the batched function
    masks = point_in_polygon_vectorized_batch(xx.flatten(), yy.flatten(), polygons_tensor)

    # Reshape individual masks
    individual_masks = masks.T.reshape(n_crosswalks, xDim, yDim)

    # Combine all masks into one using logical OR (union of all crosswalks)
    combined_mask = torch.any(individual_masks, dim=0).detach().cpu()
    
    # Reshape to (n_crosswalks, xDim, yDim)
    return combined_mask

def point_in_polygon_vectorized_batch(x, y, polygons_tensor):
    '''
    Ultra-fast batch processing for multiple polygons of similar size.
    
    Args:
        x: x coordinates of points to test (shape: n_points)
        y: y coordinates of points to test (shape: n_points)
        polygons_tensor: tensor of shape (n_polygons, max_vertices, 2)
                        Pad shorter polygons with duplicate last vertex
        max_vertices: maximum number of vertices across all polygons
    
    Returns:
        mask: boolean mask of shape (n_points, n_polygons)
    '''
    device = x.device
    n_points = x.shape[0]
    n_polygons, n_vertices, _ = polygons_tensor.shape
    n_edges = n_vertices - 1
    
    # Get all edge endpoints at once
    x1 = polygons_tensor[:, :-1, 0].T  # (n_edges, n_polygons)
    y1 = polygons_tensor[:, :-1, 1].T  # (n_edges, n_polygons)
    x2 = polygons_tensor[:, 1:, 0].T   # (n_edges, n_polygons)
    y2 = polygons_tensor[:, 1:, 1].T   # (n_edges, n_polygons)
    
    # Broadcast for all points: (n_points, n_edges, n_polygons)
    x1 = x1.unsqueeze(0).expand(n_points, -1, -1)
    y1 = y1.unsqueeze(0).expand(n_points, -1, -1)
    x2 = x2.unsqueeze(0).expand(n_points, -1, -1)
    y2 = y2.unsqueeze(0).expand(n_points, -1, -1)
    px = x.unsqueeze(1).unsqueeze(2).expand(-1, n_edges, n_polygons)
    py = y.unsqueeze(1).unsqueeze(2).expand(-1, n_edges, n_polygons)
    
    # Ray casting for all combinations
    cond1 = (y1 > py) != (y2 > py)
    
    dy = y2 - y1
    dy_safe = torch.where(torch.abs(dy) < 1e-10, torch.sign(dy) * 1e-10, dy)
    cond2 = px < x1 + (x2 - x1) * (py - y1) / dy_safe
    
    # Count intersections per polygon
    intersections = (cond1 & cond2).sum(dim=1)  # (n_points, n_polygons)
    
    return intersections % 2 == 1

def get_multiple_crosswalk_masks_cv2(
        crosswalks,
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
    Get multiple crosswalk BEV grid masks simultaneously using cv2.fillPoly.
    
    Args:
        crosswalks: list of crosswalk coordinate arrays
        ... (other args same as get_crosswalk_mask)
    
    Returns:
        mask: combined tensor of shape (xDim, yDim)
    '''
    if yDim is None:
        yDim = xDim
    
    if yRes is None:
        yRes = xRes

    if not crosswalks:
        return torch.zeros(xDim, yDim, dtype=torch.bool)

    # Transform all crosswalks to local coordinates
    R = torch.inverse(torch.from_numpy(local_to_global(ego_loc, ego_rot))).to(dType)
    
    # Calculate grid bounds
    xLim = xDim * xRes / 2
    yLim = yDim * yRes / 2
    
    # Create empty combined mask
    combined_mask = np.zeros((xDim, yDim), dtype=np.uint8)
    
    # Process each crosswalk
    all_polygons = []
    
    for crosswalk in crosswalks:
        if not isinstance(crosswalk, torch.Tensor):
            crosswalk = torch.from_numpy(crosswalk).to(dType)
        
        crosswalk[:, 2] = 1
        local_box = (R @ crosswalk.T)[:2].T

        # Convert to grid indices
        grid_x = ((xLim - local_box[:, 1]) / xRes).numpy().astype(np.int32)
        grid_y = ((yLim - local_box[:, 0]) / yRes).numpy().astype(np.int32)

        # Clip to grid bounds
        grid_x = np.clip(grid_x, 0, xDim - 1)
        grid_y = np.clip(grid_y, 0, yDim - 1)
        
        # Create polygon points
        polygon_points = np.column_stack([grid_x, grid_y]).reshape((-1, 1, 2))
        all_polygons.append(polygon_points)
    
    # Fill all polygons at once
    for poly in all_polygons:
        cv2.fillPoly(combined_mask, [poly], 1)

    return torch.from_numpy(combined_mask).bool()

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