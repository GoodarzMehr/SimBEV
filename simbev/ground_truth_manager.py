# Academic Software License: Copyright © 2025 Goodarz Mehr.

'''
Module that manages ground truth generation for CARLA simulation, including
BEV semantic segmentation, object detection bounding boxes, and HD map information.
'''

import json
import carla
import torch
import numpy as np

from utils import *
from skimage.morphology import binary_closing, binary_opening, binary_dilation


class GTManager:
    '''
    Ground Truth Manager class that handles the generation of various types
    of ground truth data including BEV semantic maps, object bounding boxes,
    and HD map information.

    Args:
        core: CarlaCore object instance that provides access to the CARLA
              world, vehicle, sensors, and configuration.
    '''
    
    def __init__(self, core):
        '''
        Initialize the GTManager with a reference to the CarlaCore object.
        
        Args:
            core: CarlaCore object instance
        '''
        self.core = core
        
        # Define traffic sign mapping (copied from carla_core.py)
        self.TRAFFIC_SIGN = {
            'highwaySign': 'highway',
            'pole_': 'billboard',
            'DoNoEnter': 'do_not_enter',
            'Interchange': 'interchange',
            'NoTrunLeft': 'no_left_turn',
            'busNoPark': 'bus_stop_no_parking',
            'cleanPet': 'clean_up_after_pet',
            'minPark': 'parking_time_limit',
            'noBicyc': 'no_bicycles',
            'noPark': 'no_parking',
            'noPed': 'no_pedestrians',
            'noStand': 'no_standing',
            'onlyCrossw': 'cross_only_at_crosswalk',
            'photo': 'photo_enforced',
            'reserved': 'reserved_parking',
            'school': 'school_zone',
            'stopPed': 'stop_for_pedestrians',
            'tow': 'tow_away_zone',
            'yieldPed': 'yield_to_pedestrians',
            'passenger': 'passenger_cars_only',
            'DiamondSignal': 'accident_ahead',
            'wildCrossing': 'animal_crossing',
            'AnimalCrossing': 'animal_crossing',
            'LaneReduceL': 'lane_reduction',
            'LaneReduct': 'lane_reduction',
            'MichiganLeft': 'michigan_left',
            'NoTurns': 'no_turns',
            'noTurn': 'no_turns',
            'OneWay': 'one_way',
            'oneWay': 'one_way',
            'stopSign': 'stop',
            'STOP_': 'stop',
            '_Stop': 'stop',
            '_Yield': 'yield',
            '_yield_': 'yield',
            '_Flag': 'street_name',
            '_letter': 'street_name',
            'SpeedLimiter': 'speed_limit',
            'SpeedSign': 'speed_limit',
            '_SpeedLimit': 'speed_limit',
            'SpeedLimit20': 'speed_limit_20',
            'SpeedLimit25': 'speed_limit_25',
            'SpeedLimit30': 'speed_limit_30',
            'SpeedLimit40': 'speed_limit_40',
            'SpeedLimit50': 'speed_limit_50',
            'SpeedLimit55': 'speed_limit_55',
            'SpeedLimit60': 'speed_limit_60',
            'SpeedLimit70': 'speed_limit_70',
            'SpeedLimit75': 'speed_limit_75',
            'SpeedLimit80': 'speed_limit_80',
            'SpeedLimit90': 'speed_limit_90',
            'SpeedLimit100': 'speed_limit_100',
            'SpeedLimit110': 'speed_limit_110',
            'SpeedLimit120': 'speed_limit_120',
            'SpeedLimit20_15': 'speed_limit_20_min_15',
            'SpeedLimit25_15': 'speed_limit_25_min_15',
            'SpeedLimit30_15': 'speed_limit_30_min_15',
            'SpeedLimit50_40': 'speed_limit_50_min_40',
            'SpeedLimit55_40': 'speed_limit_55_min_40',
            'SpeedLimit75_45': 'speed_limit_75_min_45',
        }

    def get_bev_gt(self):
        '''
        Get the BEV ground truth using the top and bottom semantic cameras and
        road waypoints.
        '''
        vehicle_transform = self.core.vehicle.get_transform()

        # Get the road mask from waypoints.
        wp_road_mask = get_road_mask(
            self.core.nwp_loc,
            self.core.nwp_lw,
            vehicle_transform.location,
            vehicle_transform.rotation,
            self.core.config['bev_dim'],
            self.core.config['bev_res'],
            device=f'cuda:{self.core.config["cuda_gpu"]}',
            dType=self.core.dType
        )

        # Get the road line mask from road line points.
        road_line_mask = get_road_mask(
            self.core.nlm_loc,
            self.core.nlm_lw * 4.8,
            vehicle_transform.location,
            vehicle_transform.rotation,
            self.core.config['bev_dim'],
            self.core.config['bev_res'],
            device=f'cuda:{self.core.config["cuda_gpu"]}',
            dType=self.core.dType
        ).detach().cpu().numpy().astype(bool)

        # Get the sidewalk mask from sidewalk points.
        wp_sidewalk_mask = get_road_mask(
            self.core.nsw_loc,
            self.core.nsw_lw,
            vehicle_transform.location,
            vehicle_transform.rotation,
            self.core.config['bev_dim'],
            self.core.config['bev_res'],
            device=f'cuda:{self.core.config["cuda_gpu"]}',
            dType=self.core.dType
        ).detach().cpu().numpy().astype(bool)

        # For Town06, get the sidewalk mask from sidewalk meshes.
        if self.core.map_name == 'Town06':
            mesh_sidewalk_mask = np.zeros((self.core.config['bev_dim'], self.core.config['bev_dim'])).astype(bool)

            all_sidewalks = [obj for obj in self.core.objects if '_Sidewalk_' in obj.name]

            for sidewalk in all_sidewalks:
                bbox = sidewalk.bounding_box.get_local_vertices()

                bbox = carla_vector_to_torch(bbox)

                bbox[:, 1] *= -1.0

                resulting_mask = get_object_mask(
                    bbox,
                    vehicle_transform.location,
                    vehicle_transform.rotation,
                    self.core.config['bev_dim'],
                    self.core.config['bev_res'],
                    device=f'cuda:{self.core.config["cuda_gpu"]}',
                    dType=self.core.dType
                )

                mesh_sidewalk_mask = np.logical_or(mesh_sidewalk_mask, resulting_mask).numpy().astype(bool)

        # Get the crosswalk mask from crosswalk locations.
        crosswalk_mask = np.zeros((self.core.config['bev_dim'], self.core.config['bev_dim'])).astype(bool)

        if self.core.map_name in ['Town03', 'Town04', 'Town05', 'Town06', 'Town07', 'Town10HD']:
            all_crosswalks = [obj for obj in self.core.objects if '_Crosswalk_' in obj.name]

            crosswalks = [cw for cw in all_crosswalks if not any(x in cw.name for x in self.core.bad_crosswalks)]

            for crosswalk in crosswalks:
                bbox = crosswalk.bounding_box.get_local_vertices()

                crosswalk_box = carla_vector_to_torch([bbox[0], bbox[2], bbox[6], bbox[4], bbox[0]])

                crosswalk_box[:, 1] *= -1.0

                resulting_mask = get_crosswalk_mask(
                    crosswalk_box,
                    vehicle_transform.location,
                    vehicle_transform.rotation,
                    self.core.config['bev_dim'],
                    self.core.config['bev_res'],
                    device=f'cuda:{self.core.config["cuda_gpu"]}',
                    dType=self.core.dType
                )

                crosswalk_mask = np.logical_or(crosswalk_mask, resulting_mask).numpy().astype(bool)
        else:
            for crosswalk in self.core.trimmed_crosswalks:
                crosswalk = carla_vector_to_torch(crosswalk)

                crosswalk[:, 1] *= -1.0

                resulting_mask = get_crosswalk_mask(
                    crosswalk,
                    vehicle_transform.location,
                    vehicle_transform.rotation,
                    self.core.config['bev_dim'],
                    self.core.config['bev_res'],
                    device=f'cuda:{self.core.config["cuda_gpu"]}',
                    dType=self.core.dType
                )

                crosswalk_mask = np.logical_or(crosswalk_mask, resulting_mask).numpy().astype(bool)

        # Get images from the top and bottom semantic cameras. Use the top
        # image along with the road mask from waypoints to create a road mask,
        # and use both images to create masks for cars, trucks, buses,
        # motorcycles, bicycles, riders, and pedestrians.
        top_bev_image = self.core.sensor_manager.semantic_bev_camera_list[0].save_queue.get(True, 10.0)
        bottom_bev_image = np.flip(self.core.sensor_manager.semantic_bev_camera_list[1].save_queue.get(True, 10.0), axis=0)

        bev_road_mask = np.logical_or(top_bev_image[:, :, 2] == 128, top_bev_image[:, :, 2] == 157)

        road_mask = binary_closing(np.logical_or(wp_road_mask, bev_road_mask))

        road_line_mask = binary_closing(road_line_mask, footprint=np.ones((3, 3)))
        
        if self.core.config['use_bev_for_sidewalks']:
            sidewalk_mask = binary_closing(np.logical_or(top_bev_image[:, :, 0] == 232, wp_sidewalk_mask))
        elif self.core.map_name == 'Town06':
            sidewalk_mask = binary_closing(np.logical_or(wp_sidewalk_mask, np.logical_and(
                mesh_sidewalk_mask, np.logical_or(top_bev_image[:, :, 0] == 232, bottom_bev_image[:, :, 0] == 232)
            )))
        else:
            sidewalk_mask = binary_closing(wp_sidewalk_mask)

        if self.core.map_name in ['Town12', 'Town13']:
            bev_crosswalk_mask = binary_opening(
                np.logical_or(top_bev_image[:, :, 2] == 157, crosswalk_mask),
                footprint=np.ones((3, 3))
            )
        elif self.core.map_name in ['Town15']:
            bev_crosswalk_mask = np.logical_or(
                np.logical_and(
                    binary_dilation(crosswalk_mask, footprint=np.ones((11, 11))),
                    np.logical_or(top_bev_image[:, :, 2] == 157, top_bev_image[:, :, 2] == 110),
                ),
                np.logical_and(crosswalk_mask, top_bev_image[:, :, 1] == 70)
            )
            bev_crosswalk_mask = binary_opening(binary_closing(bev_crosswalk_mask), footprint=np.ones((3, 3)))
            bev_crosswalk_mask = binary_closing(
                np.logical_or(bev_crosswalk_mask, ~road_mask), footprint=np.ones((4, 4))
            )
        else:
            bev_crosswalk_mask = crosswalk_mask

        crosswalk_mask = binary_closing(np.logical_and(bev_crosswalk_mask, road_mask))

        car_mask = np.logical_or(bottom_bev_image[:, :, 0] == 142, top_bev_image[:, :, 0] == 142)
        truck_mask = np.logical_or(
            np.logical_and(bottom_bev_image[:, :, 0] == 70, bottom_bev_image[:, :, 1] == 0),
            np.logical_and(top_bev_image[:, :, 0] == 70, top_bev_image[:, :, 1] == 0)
        )
        bus_mask = np.logical_or(
            np.logical_and(bottom_bev_image[:, :, 0] == 100, bottom_bev_image[:, :, 1] == 60),
            np.logical_and(top_bev_image[:, :, 0] == 100, top_bev_image[:, :, 1] == 60)
        )
        motorcycle_mask = np.logical_or(bottom_bev_image[:, :, 0] == 230, top_bev_image[:, :, 0] == 230)
        bicycle_mask = np.logical_or(bottom_bev_image[:, :, 0] == 32, top_bev_image[:, :, 0] == 32)
        rider_mask = np.logical_or(bottom_bev_image[:, :, 2] == 255, top_bev_image[:, :, 2] == 255)
        pedestrian_mask = np.logical_or(bottom_bev_image[:, :, 1] == 20, top_bev_image[:, :, 1] == 20)

        # Concatenate individual masks to get the ground truth.
        self.core.bev_gt = np.array([
            road_mask,
            road_line_mask,
            sidewalk_mask,
            crosswalk_mask,
            car_mask,
            truck_mask,
            bus_mask,
            motorcycle_mask,
            bicycle_mask,
            rider_mask,
            pedestrian_mask
        ])

    def get_bev_gt_alt(self):
        '''
        Get the BEV ground truth using object bounding boxes and road
        waypoints.
        '''
        vehicle_transform = self.core.vehicle.get_transform()

        # Find waypoints whose elevation difference with the ego vehicle is
        # less than a certain threshold.
        mask = (self.core.nwp_loc[:, 2] - vehicle_transform.location.z).abs() < 4.8
        
        # Get the road mask from waypoints.
        wp_road_mask = get_road_mask(
            self.core.nwp_loc[mask],
            self.core.nwp_lw[mask],
            vehicle_transform.location,
            vehicle_transform.rotation,
            self.core.config['bev_dim'],
            self.core.config['bev_res'],
            device=f'cuda:{self.core.config["cuda_gpu"]}',
            dType=self.core.dType
        )

        road_mask = binary_closing(wp_road_mask)

        # Find road line points whose elevation difference with the ego
        # vehicle is less than a certain threshold.
        mask = (self.core.nlm_loc[:, 2] - vehicle_transform.location.z).abs() < 4.8

        # Get the road line mask from road line points.
        road_line_mask = get_road_mask(
            self.core.nlm_loc[mask],
            self.core.nlm_lw[mask] * 4.8,
            vehicle_transform.location,
            vehicle_transform.rotation,
            self.core.config['bev_dim'],
            self.core.config['bev_res'],
            device=f'cuda:{self.core.config["cuda_gpu"]}',
            dType=self.core.dType
        ).detach().cpu().numpy().astype(bool)

        road_line_mask = binary_closing(road_line_mask, footprint=np.ones((3, 3)))

        # Find sidewalk points whose elevation difference with the ego vehicle
        # is less than a certain threshold.
        mask = (self.core.nsw_loc[:, 2] - vehicle_transform.location.z).abs() < 4.8

        # Get the sidewalk mask from sidewalk points.
        sidewalk_mask = get_road_mask(
            self.core.nsw_loc[mask],
            self.core.nsw_lw[mask],
            vehicle_transform.location,
            vehicle_transform.rotation,
            self.core.config['bev_dim'],
            self.core.config['bev_res'],
            device=f'cuda:{self.core.config["cuda_gpu"]}',
            dType=self.core.dType
        ).detach().cpu().numpy().astype(bool)

        sidewalk_mask = binary_closing(sidewalk_mask)

        # Get the crosswalk mask from crosswalk locations.
        crosswalk_mask = np.zeros((self.core.config['bev_dim'], self.core.config['bev_dim'])).astype(bool)

        if self.core.map_name in ['Town03', 'Town04', 'Town05', 'Town06', 'Town07', 'Town10HD']:
            all_crosswalks = [obj for obj in self.core.objects if '_Crosswalk_' in obj.name]

            crosswalks = [cw for cw in all_crosswalks if not any(x in cw.name for x in self.core.bad_crosswalks)]

            for crosswalk in crosswalks:
                bbox = crosswalk.bounding_box.get_local_vertices()

                if np.abs(bbox[0].z - vehicle_transform.location.z) < 4.8:
                    crosswalk_box = carla_vector_to_torch([bbox[0], bbox[2], bbox[6], bbox[4], bbox[0]])

                    crosswalk_box[:, 1] *= -1.0

                    resulting_mask = get_crosswalk_mask(
                        crosswalk_box,
                        vehicle_transform.location,
                        vehicle_transform.rotation,
                        self.core.config['bev_dim'],
                        self.core.config['bev_res'],
                        device=f'cuda:{self.core.config["cuda_gpu"]}',
                        dType=self.core.dType
                    )

                    crosswalk_mask = np.logical_or(crosswalk_mask, resulting_mask).numpy().astype(bool)
        else:
            for crosswalk in self.core.trimmed_crosswalks:
                if np.abs(crosswalk[0].z - vehicle_transform.location.z) < 4.8:
                    crosswalk = carla_vector_to_torch(crosswalk)

                    crosswalk[:, 1] *= -1.0

                    resulting_mask = get_crosswalk_mask(
                        crosswalk,
                        vehicle_transform.location,
                        vehicle_transform.rotation,
                        self.core.config['bev_dim'],
                        self.core.config['bev_res'],
                        device=f'cuda:{self.core.config["cuda_gpu"]}',
                        dType=self.core.dType
                    )

                    crosswalk_mask = np.logical_or(crosswalk_mask, resulting_mask).numpy().astype(bool)

        crosswalk_mask = binary_closing(np.logical_and(crosswalk_mask, road_mask))
        
        # Get object masks from bounding boxes.
        car_mask = np.zeros((self.core.config['bev_dim'], self.core.config['bev_dim'])).astype(bool)
        truck_mask = np.zeros((self.core.config['bev_dim'], self.core.config['bev_dim'])).astype(bool)
        bus_mask = np.zeros((self.core.config['bev_dim'], self.core.config['bev_dim'])).astype(bool)
        motorcycle_mask = np.zeros((self.core.config['bev_dim'], self.core.config['bev_dim'])).astype(bool)
        bicycle_mask = np.zeros((self.core.config['bev_dim'], self.core.config['bev_dim'])).astype(bool)
        rider_mask = np.zeros((self.core.config['bev_dim'], self.core.config['bev_dim'])).astype(bool)
        pedestrian_mask = np.zeros((self.core.config['bev_dim'], self.core.config['bev_dim'])).astype(bool)

        # Iterate over all bounding boxes and add the mask for each object to
        # the corresponding mask.
        for actor in self.core.actors:
            if any(x in actor['semantic_tags'] for x in [12, 13, 14, 15, 16, 18, 19]):
                if np.abs(actor['bounding_box'][:, 2] - vehicle_transform.location.z).max() < 4.8:
                    resulting_mask = get_object_mask(
                        actor['bounding_box'],
                        vehicle_transform.location,
                        vehicle_transform.rotation,
                        self.core.config['bev_dim'],
                        self.core.config['bev_res'],
                        device=f'cuda:{self.core.config["cuda_gpu"]}',
                        dType=self.core.dType
                    )
                    
                    if 12 in actor['semantic_tags']:
                        pedestrian_mask = np.logical_or(pedestrian_mask, resulting_mask).numpy().astype(bool)
                    if 13 in actor['semantic_tags']:
                        rider_mask = np.logical_or(rider_mask, resulting_mask).numpy().astype(bool)
                    if 14 in actor['semantic_tags']:
                        car_mask = np.logical_or(car_mask, resulting_mask).numpy().astype(bool)
                    if 15 in actor['semantic_tags']:
                        truck_mask = np.logical_or(truck_mask, resulting_mask).numpy().astype(bool)
                    if 16 in actor['semantic_tags']:
                        bus_mask = np.logical_or(bus_mask, resulting_mask).numpy().astype(bool)
                    if 18 in actor['semantic_tags']:
                        motorcycle_mask = np.logical_or(motorcycle_mask, resulting_mask).numpy().astype(bool)
                    if 19 in actor['semantic_tags']:
                        bicycle_mask = np.logical_or(bicycle_mask, resulting_mask).numpy().astype(bool)
        
        # Concatenate individual masks to get the ground truth.
        self.core.bev_gt = np.array([
            road_mask,
            road_line_mask,
            sidewalk_mask,
            crosswalk_mask,
            car_mask,
            truck_mask,
            bus_mask,
            motorcycle_mask,
            bicycle_mask,
            rider_mask,
            pedestrian_mask
        ])

    def get_bounding_boxes(self):
        '''
        Get the bounding box of actors (including traffic elements) in the
        scene that are within a certain radius of the ego vehicle.
        '''
        self.core.actors = []

        actor_list = self.core.world.get_actors()

        for actor in actor_list:
            actor_properties = {}
            actor_location = actor.get_location()

            if self.core.vehicle_location.distance(actor_location) < self.core.config['bbox_collection_radius'] \
                and all(x not in actor.type_id for x in ['spectator', 'sensor', 'controller']) \
                    and any(x in actor.semantic_tags for x in [12, 13, 14, 15, 16, 18, 19]):
                actor_properties['id'] = actor.id
                actor_properties['type'] = str(actor.type_id)
                actor_properties['is_alive'] = actor.is_alive
                actor_properties['is_active'] = actor.is_active
                actor_properties['is_dormant'] = actor.is_dormant
                actor_properties['parent'] = actor.parent.id if actor.parent is not None else None
                actor_properties['attributes'] = actor.attributes
                actor_properties['semantic_tags'] = actor.semantic_tags
                actor_properties['bounding_box'] = carla_vector_to_numpy(
                    actor.bounding_box.get_world_vertices(actor.get_transform())
                )
                actor_properties['linear_velocity'] = carla_single_vector_to_numpy(actor.get_velocity())
                actor_properties['angular_velocity'] = carla_single_vector_to_numpy(actor.get_angular_velocity())

                actor_properties['bounding_box'][:, 1] *= -1.0
                actor_properties['linear_velocity'][1] *= -1.0
                actor_properties['angular_velocity'][1:] *= -1.0

                self.core.actors.append(actor_properties)

            # Get traffic lights.
            if self.core.config['collect_traffic_light_bbox']:
                if isinstance(actor, carla.TrafficLight):
                    if self.core.vehicle_location.distance(actor_location) < self.core.config['bbox_collection_radius']:
                        bounding_boxes = actor.get_light_boxes()

                        for bounding_box in bounding_boxes:
                            if bounding_box.extent.z > 0.3:
                                actor_properties = {}

                                actor_properties['id'] = actor.id
                                actor_properties['type'] = str(actor.type_id)
                                actor_properties['is_alive'] = actor.is_alive
                                actor_properties['is_active'] = actor.is_active
                                actor_properties['is_dormant'] = actor.is_dormant
                                actor_properties['parent'] = actor.parent.id if actor.parent is not None else None
                                actor_properties['attributes'] = actor.attributes
                                actor_properties['semantic_tags'] = [7]

                                if self.core.map_name in ['Town12', 'Town13']:
                                    tile_bounding_box = carla_vector_to_numpy(bounding_box.get_local_vertices())

                                    x_difference = np.round((actor_location.x - bounding_box.location.x) / 1000.0) \
                                        * 1000.0
                                    y_difference = np.round((actor_location.y - bounding_box.location.y) / 1000.0) \
                                        * 1000.0

                                    actor_properties['bounding_box'] = tile_bounding_box + np.array([
                                        x_difference, y_difference, 0.0
                                    ])
                                else:
                                    actor_properties['bounding_box'] = carla_vector_to_numpy(
                                        bounding_box.get_local_vertices()
                                    )

                                actor_properties['linear_velocity'] = np.zeros(3)
                                actor_properties['angular_velocity'] = np.zeros(3)

                                actor_properties['bounding_box'][:, 1] *= -1.0

                                actor_properties['green_time'] = actor.get_green_time()
                                actor_properties['yellow_time'] = actor.get_yellow_time()
                                actor_properties['red_time'] = actor.get_red_time()
                                
                                actor_properties['state'] = str(actor.get_state())

                                actor_properties['opendrive_id'] = actor.get_opendrive_id()
                                actor_properties['pole_index'] = actor.get_pole_index()

                                self.core.actors.append(actor_properties)
        
        # Get the list of objects (props) in the scene, i.e. parked cars,
        # trucks, etc. that are within a certain radius of the ego vehicle.
        if self.core.config['collect_traffic_sign_bbox']:
            traffic_sign_list = self.core.world.get_environment_objects(carla.CityObjectLabel.TrafficSigns)
        
        car_list = self.core.world.get_environment_objects(carla.CityObjectLabel.Car)
        truck_list = self.core.world.get_environment_objects(carla.CityObjectLabel.Truck)
        bus_list = self.core.world.get_environment_objects(carla.CityObjectLabel.Bus)
        motorcycle_list = self.core.world.get_environment_objects(carla.CityObjectLabel.Motorcycle)
        bicycle_list = self.core.world.get_environment_objects(carla.CityObjectLabel.Bicycle)

        if self.core.config['collect_traffic_sign_bbox']:
            object_list = traffic_sign_list + car_list + truck_list + bus_list + motorcycle_list + bicycle_list
        else:
            object_list = car_list + truck_list + bus_list + motorcycle_list + bicycle_list

        for obj in object_list:
            object_properties = {}
            object_location = obj.transform.location if self.core.map_name not in ['Town12', 'Town13'] \
                else obj.bounding_box.location

            if self.core.vehicle_location.distance(object_location) < self.core.config['bbox_collection_radius']:
                object_properties['id'] = obj.id
                object_properties['type'] = str(obj.type)
                object_properties['is_alive'] = False
                object_properties['is_active'] = False
                object_properties['is_dormant'] = False
                object_properties['parent'] = None
                object_properties['attributes'] = {}
                object_properties['bounding_box'] = carla_vector_to_numpy(obj.bounding_box.get_local_vertices())
                object_properties['linear_velocity'] = np.zeros(3)
                object_properties['angular_velocity'] = np.zeros(3)

                object_properties['bounding_box'][:, 1] *= -1.0

                if obj.type == carla.CityObjectLabel.TrafficSigns:
                    object_properties['semantic_tags'] = [8]
                    object_properties['sign_type'] = ''

                    for sign in self.TRAFFIC_SIGN.keys():
                        if sign in obj.name:
                            object_properties['sign_type'] = self.TRAFFIC_SIGN[sign]

                    if self.core.map_name not in ['Town12', 'Town13', 'Town15'] and \
                        'speed_limit' in object_properties['sign_type']:
                        object_properties['sign_type'] = 'speed_limit'

                elif obj.type == carla.CityObjectLabel.Car:
                    object_properties['semantic_tags'] = [14]
                elif obj.type == carla.CityObjectLabel.Truck:
                    object_properties['semantic_tags'] = [15]
                elif obj.type == carla.CityObjectLabel.Bus:
                    object_properties['semantic_tags'] = [16]
                elif obj.type == carla.CityObjectLabel.Motorcycle:
                    object_properties['semantic_tags'] = [18]
                elif obj.type == carla.CityObjectLabel.Bicycle:
                    object_properties['semantic_tags'] = [19]
                else:
                    object_properties['semantic_tags'] = []

                self.core.actors.append(object_properties)

    def get_hd_map_info(self):
        '''
        Get HD map information from the waypoint at the vehicle's current location.
        '''
        self.core.hd_map_info = {}

        # Get the waypoint at the vehicle's current location.
        wp = self.core.map.get_waypoint(self.core.vehicle.get_location())

        self.core.hd_map_info['id'] = wp.id
        self.core.hd_map_info['s'] = wp.s
        self.core.hd_map_info['road_id'] = wp.road_id
        self.core.hd_map_info['section_id'] = wp.section_id
        self.core.hd_map_info['lane_id'] = wp.lane_id
        self.core.hd_map_info['lane_type'] = str(wp.lane_type)
        self.core.hd_map_info['lane_width'] = wp.lane_width
        self.core.hd_map_info['lane_change'] = str(wp.lane_change)

        self.core.hd_map_info['is_junction'] = wp.is_junction
        self.core.hd_map_info['junction_id'] = wp.junction_id if wp.is_junction else None
        self.core.hd_map_info['is_intersection'] = wp.is_intersection

        self.core.hd_map_info['left_lane_marking'] = {
            'type': str(wp.left_lane_marking.type),
            'width': wp.left_lane_marking.width,
            'color': str(wp.left_lane_marking.color),
            'lane_change': str(wp.left_lane_marking.lane_change)
        }

        self.core.hd_map_info['right_lane_marking'] = {
            'type': str(wp.right_lane_marking.type),
            'width': wp.right_lane_marking.width,
            'color': str(wp.right_lane_marking.color),
            'lane_change': str(wp.right_lane_marking.lane_change)
        }

        self.core.hd_map_info['transform'] = {
            'x': wp.transform.location.x,
            'y': -wp.transform.location.y,
            'z': wp.transform.location.z,
            'roll': wp.transform.rotation.roll,
            'pitch': -wp.transform.rotation.pitch,
            'yaw': -wp.transform.rotation.yaw
        }

        left_lane_wp = wp.get_left_lane()
        right_lane_wp = wp.get_right_lane()

        self.core.hd_map_info['left_lane'] = {}
        self.core.hd_map_info['right_lane'] = {}

        if left_lane_wp is not None:
            self.core.hd_map_info['left_lane']['id'] = left_lane_wp.id
            self.core.hd_map_info['left_lane']['s'] = left_lane_wp.s
            self.core.hd_map_info['left_lane']['road_id'] = left_lane_wp.road_id
            self.core.hd_map_info['left_lane']['section_id'] = left_lane_wp.section_id
            self.core.hd_map_info['left_lane']['lane_id'] = left_lane_wp.lane_id
            self.core.hd_map_info['left_lane']['lane_type'] = str(left_lane_wp.lane_type)
            self.core.hd_map_info['left_lane']['lane_width'] = left_lane_wp.lane_width
            self.core.hd_map_info['left_lane']['lane_change'] = str(left_lane_wp.lane_change)

        if right_lane_wp is not None:
            self.core.hd_map_info['right_lane']['id'] = right_lane_wp.id
            self.core.hd_map_info['right_lane']['s'] = right_lane_wp.s
            self.core.hd_map_info['right_lane']['road_id'] = right_lane_wp.road_id
            self.core.hd_map_info['right_lane']['section_id'] = right_lane_wp.section_id
            self.core.hd_map_info['right_lane']['lane_id'] = right_lane_wp.lane_id
            self.core.hd_map_info['right_lane']['lane_type'] = str(right_lane_wp.lane_type)
            self.core.hd_map_info['right_lane']['lane_width'] = right_lane_wp.lane_width
            self.core.hd_map_info['right_lane']['lane_change'] = str(right_lane_wp.lane_change)