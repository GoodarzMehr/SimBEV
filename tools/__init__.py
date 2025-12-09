'''
SimBEV Tools

Copyright © 2025 Goodarz Mehr

Post-processing and visualization utilities for SimBEV datasets.
'''

__version__ = '3.0.0'
__author__ = 'Goodarz Mehr'
__email__ = 'goodarzm@vt.edu'

# Note: Import functions only when needed to avoid triggering argparse at
# import time

# Utility functions
from .visualization_utils import *

# Handlers
from .visualization_handlers import *

# Interactive visualization
from .visualization_interactive import *

__all__ = [
    '__version__',
    '__author__',
    '__email__',
    'visualization_utils',
    'visualization_handlers',
    'visualization_interactive'
]
