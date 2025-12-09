'''
SimBEV: A Synthetic Multi-Task Multi-Sensor Driving Data Generation Tool

Copyright © 2025 Goodarz Mehr

Main package exports for SimBEV.
'''

__version__ = '3.0.0'
__author__ = 'Goodarz Mehr'
__email__ = 'goodarzm@vt.edu'

# Core simulation components
from .carla_core import CarlaCore
from .world_manager import WorldManager
from .sensor_manager import SensorManager
from .vehicle_manager import VehicleManager
from .scenario_manager import ScenarioManager 
from .ground_truth_manager import GTManager

# Sensor classes
from .sensors import (
    RGBCamera,
    SemanticCamera, 
    InstanceCamera,
    DepthCamera,
    FlowCamera,
    SemanticBEVCamera,
    Lidar,
    SemanticLidar,
    Radar,
    GNSS,
    IMU
)

# Utility functions
from .utils import *

__all__ = [
    # Version info
    '__version__',
    '__author__',
    '__email__',
    
    # Core components
    'CarlaCore',
    'WorldManager',
    'SensorManager', 
    'VehicleManager',
    'ScenarioManager',
    'GTManager',
    
    # Sensors
    'RGBCamera',
    'SemanticCamera',
    'InstanceCamera',
    'DepthCamera',
    'FlowCamera',
    'SemanticBEVCamera',
    'Lidar',
    'SemanticLidar',
    'Radar',
    'GNSS',
    'IMU',

    # Utilities
    'utils',
]
