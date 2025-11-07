# Academic Software License: Copyright © 2025 Goodarz Mehr.

import os
import json
import open3d as o3d
import numpy as np
from pyquaternion import Quaternion as Q
import open3d.visualization.gui as gui
import threading

# Color palette for bounding boxes
BBOX_COLORS = {
    'car': [0.0, 0.5, 0.94],
    'truck': [0.5, 0.94, 0.25],
    'bus': [0.0, 0.56, 0.0],
    'motorcycle': [0.94, 0.94, 0.0],
    'bicycle': [0.0, 0.94, 0.94],
    'rider': [0.94, 0.56, 0.0],
    'pedestrian': [0.94, 0.0, 0.0],
    'traffic_light': [0.94, 0.63, 0.0],
    'traffic_sign': [0.94, 0.0, 0.5]
}

RAD_NAME = ['RAD_LEFT', 'RAD_FRONT', 'RAD_RIGHT', 'RAD_BACK']


class LazyDataLoader:
    '''
    Lazy data loader that loads frame data on-demand.
    Implements caching to avoid reloading recently accessed frames.
    '''
    def __init__(self, path, metadata, cache_size=10):
        self.path = path
        self.metadata = metadata
        self.cache_size = cache_size
        self.cache = {}  # {(scene_idx, frame_idx, sensor_type): data}
        self.cache_order = []  # LRU cache
        
        # Pre-compute colormaps
        from matplotlib import colormaps as cm
        RANGE = np.linspace(0.0, 1.0, 256)
        self.RAINBOW = np.array(cm.get_cmap('rainbow')(RANGE))[:, :3]
        self.RANGE = RANGE
        
        # Label colors for semantic LiDAR
        self.LABEL_COLORS = np.array([
            (255, 255, 255), (128, 64, 128), (244, 35, 232), (70, 70, 70),
            (102, 102, 156), (190, 153, 153), (153, 153, 153), (250, 170, 30),
            (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180),
            (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
            (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32),
            (110, 190, 160), (170, 120, 50), (55, 90, 80), (45, 60, 150),
            (227, 227, 227), (81, 0, 81), (150, 100, 100), (230, 150, 140),
            (180, 165, 180), (180, 130, 70)
        ]) / 255.0
        
        # Load scene structure (just paths, not data)
        self.scenes_info = self._load_scene_structure()
    
    def _load_scene_structure(self):
        '''Load scene metadata and frame paths without loading actual data.'''
        scenes_info = []
        
        # Load from all splits
        for split in ['train', 'val', 'test']:
            info_path = f'{self.path}/simbev/infos/simbev_infos_{split}.json'
            
            if not os.path.exists(info_path):
                continue
            
            with open(info_path, 'r') as f:
                infos = json.load(f)
            
            # Store scene structure
            for scene_key, scene_value in infos['data'].items():
                scene_number = int(scene_key.split('_')[1])
                scene_data = scene_value['scene_data']
                
                scenes_info.append({
                    'scene_number': scene_number,
                    'frame_count': len(scene_data),
                    'frame_paths': scene_data,  # List of dicts with file paths
                    'split': split
                })
        
        # Sort by scene number
        scenes_info.sort(key=lambda x: x['scene_number'])
        
        return scenes_info
    
    def get_scene_count(self):
        '''Get total number of scenes.'''
        return len(self.scenes_info)
    
    def get_frame_count(self, scene_idx):
        '''Get number of frames in a scene.'''
        return self.scenes_info[scene_idx]['frame_count']
    
    def get_scene_number(self, scene_idx):
        '''Get scene number for display.'''
        return self.scenes_info[scene_idx]['scene_number']
    
    def _add_to_cache(self, key, data):
        '''Add data to LRU cache.'''
        # Remove oldest if cache is full
        if len(self.cache) >= self.cache_size:
            oldest_key = self.cache_order.pop(0)
            del self.cache[oldest_key]
        
        self.cache[key] = data
        self.cache_order.append(key)
    
    def _get_from_cache(self, key):
        '''Get data from cache and update LRU order.'''
        if key in self.cache:
            # Move to end (most recently used)
            self.cache_order.remove(key)
            self.cache_order.append(key)
            return self.cache[key]
        return None
    
    def load_frame(self, scene_idx, frame_idx, sensor_type):
        '''
        Load data for a specific frame and sensor type.
        
        Args:
            scene_idx: scene index (0-based)
            frame_idx: frame index within scene (0-based)
            sensor_type: 'lidar', 'semantic-lidar', or 'radar'
        
        Returns:
            dict with 'points', 'colors', 'bboxes'
        '''
        # Check cache first
        cache_key = (scene_idx, frame_idx, sensor_type)
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
        
        # Load from disk
        scene_info = self.scenes_info[scene_idx]
        frame_data = scene_info['frame_paths'][frame_idx]
        
        from tools.visualization_utils import get_global2sensor, transform_bbox
        
        # Load bounding boxes (shared across all sensors)
        gt_det = np.load(frame_data['GT_DET'], allow_pickle=True)
        global2lidar = get_global2sensor(frame_data, self.metadata, 'LIDAR')
        corners, labels = transform_bbox(gt_det, global2lidar, False)
        
        bboxes = [
            {'corners': c, 'label': l} for c, l in zip(corners, labels)
        ]
        
        if sensor_type == 'lidar':
            if 'LIDAR' in frame_data:
                data = self._load_lidar(frame_data, bboxes)
            else:
                data = {'points': np.empty((0, 3)), 'colors': None, 'bboxes': bboxes}
        elif sensor_type == 'semantic-lidar':
            if 'SEG-LIDAR' in frame_data:
                data = self._load_semantic_lidar(frame_data, bboxes)
            else:
                data = {'points': np.empty((0, 3)), 'colors': None, 'bboxes': bboxes}
        elif sensor_type == 'radar':
            if all(radar in frame_data for radar in RAD_NAME):
                data = self._load_radar(frame_data, bboxes)
            else:
                data = {'points': np.empty((0, 3)), 'colors': None, 'bboxes': bboxes}
        else:
            raise ValueError(f"Unknown sensor type: {sensor_type}")
        
        # Add to cache
        self._add_to_cache(cache_key, data)
        
        return data
    
    def _load_lidar(self, frame_data, bboxes):
        '''Load LiDAR point cloud data.'''
        point_cloud = np.load(frame_data['LIDAR'])['data']
        
        # Compute distance-based colors
        distances = np.linalg.norm(point_cloud, axis=1)
        log_distances = np.log(distances + 1e-6)
        log_normalized = (log_distances - log_distances.min()) / \
            (log_distances.max() - log_distances.min() + 1e-6)
        
        colors = np.c_[
            np.interp(log_normalized, self.RANGE, self.RAINBOW[:, 0]),
            np.interp(log_normalized, self.RANGE, self.RAINBOW[:, 1]),
            np.interp(log_normalized, self.RANGE, self.RAINBOW[:, 2])
        ]
        
        return {
            'points': point_cloud,
            'colors': colors,
            'bboxes': bboxes
        }
    
    def _load_semantic_lidar(self, frame_data, bboxes):
        '''Load semantic LiDAR point cloud data.'''
        data = np.load(frame_data['SEG-LIDAR'])['data']
        point_cloud = np.array([data['x'], data['y'], data['z']]).T
        seg_labels = np.array(data['ObjTag'])
        colors = self.LABEL_COLORS[seg_labels]
        
        return {
            'points': point_cloud,
            'colors': colors,
            'bboxes': bboxes
        }
    
    def _load_radar(self, frame_data, bboxes):
        '''Load radar point cloud data.'''
        point_cloud_list = []
        velocity_list = []
        
        for radar in RAD_NAME:
            radar2lidar = np.eye(4, dtype=np.float32)
            radar2lidar[:3, :3] = Q(self.metadata[radar]['sensor2lidar_rotation']).rotation_matrix
            radar2lidar[:3, 3] = self.metadata[radar]['sensor2lidar_translation']
            
            radar_points = np.load(frame_data[radar])['data']
            velocity_list.append(radar_points[:, -1])
            radar_points = radar_points[:, :-1]
            
            # Spherical to Cartesian
            x = radar_points[:, 0] * np.cos(radar_points[:, 1]) * np.cos(radar_points[:, 2])
            y = radar_points[:, 0] * np.cos(radar_points[:, 1]) * np.sin(radar_points[:, 2])
            z = radar_points[:, 0] * np.sin(radar_points[:, 1])
            
            points = np.stack((x, y, z), axis=1)
            points_transformed = (radar2lidar @ np.append(points, np.ones((points.shape[0], 1)), 1).T)[:3].T
            
            point_cloud_list.append(points_transformed)
        
        point_cloud = np.concatenate(point_cloud_list, axis=0)
        velocity = np.concatenate(velocity_list, axis=0)
        
        # Velocity-based colors
        log_velocity = np.log(1.0 + np.abs(velocity))
        log_velocity_normalized = (log_velocity - log_velocity.min()) / \
            (log_velocity.max() - log_velocity.min() + 1e-6)
        
        colors = np.c_[
            np.interp(log_velocity_normalized, self.RANGE, self.RAINBOW[:, 0]),
            np.interp(log_velocity_normalized, self.RANGE, self.RAINBOW[:, 1]),
            np.interp(log_velocity_normalized, self.RANGE, self.RAINBOW[:, 2])
        ]
        
        return {
            'points': point_cloud,
            'colors': colors,
            'bboxes': bboxes
        }


class AdvancedVisualizer:
    '''
    Advanced interactive visualizer with GUI slider and controls.
    Uses lazy loading for memory efficiency.
    
    Args:
        title: window title.
        point_size: point cloud rendering size.
    '''
    def __init__(self, data_loader, title='SimBEV Interactive Viewer', point_size=2.0):
        self.data_loader = data_loader
        
        self.app = gui.Application.instance
        self.app.initialize()
        
        self.window = self.app.create_window(title, 1920, 1080)
        
        # State
        self.current_scene = 0
        self.max_scene = data_loader.get_scene_count() - 1
        self.current_frame = 0
        self.max_frame = data_loader.get_frame_count(0) - 1
        self.show_bbox = True
        self.sensor_type = 'lidar'
        self.point_size = point_size
        
        # Create 3D scene widget
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = o3d.visualization.rendering.Open3DScene(
            self.window.renderer
        )
        
        # Set up scene rendering
        self.scene_widget.scene.set_background([0.1, 0.1, 0.1, 1.0])
        
        # Create control panel
        self._create_control_panel()
        
        # Layout
        self.window.set_on_layout(self._on_layout)
        self.window.add_child(self.scene_widget)
        self.window.add_child(self.panel)
        
        # Add coordinate frame
        self._add_coordinate_frame()
        
        # Load initial frame
        self._update_frame()
        
        # Setup camera
        bounds = self.scene_widget.scene.bounding_box
        self.scene_widget.setup_camera(60, bounds, bounds.get_center())
    
    def _add_coordinate_frame(self):
        '''Add coordinate frame to scene.'''
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=3.0, origin=[0, 0, 0]
        )
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        self.scene_widget.scene.add_geometry("coordinate_frame", coord_frame, mat)
    
    def _create_control_panel(self):
        '''Create UI control panel with sliders, checkboxes, and radio buttons.'''
        em = self.window.theme.font_size
        
        self.panel = gui.Vert(0, gui.Margins(em, em, em, em))
        
        # Sensor type selection
        self.panel.add_child(gui.Label("Sensor Type:"))
        
        self.sensor_radio = gui.RadioButton(gui.RadioButton.VERT)
        self.sensor_radio.set_items(["LiDAR", "Semantic LiDAR", "Radar"])
        self.sensor_radio.selected_index = 0
        self.sensor_radio.set_on_selection_changed(self._on_sensor_changed)
        self.panel.add_child(self.sensor_radio)
        
        self.panel.add_fixed(em)
        
        # Scene slider
        self.scene_label = gui.Label(f"Scene: {self.data_loader.get_scene_number(0):04d} (1/{self.max_scene + 1})")
        self.panel.add_child(self.scene_label)
        
        self.scene_slider = gui.Slider(gui.Slider.INT)
        self.scene_slider.set_limits(0, self.max_scene)
        self.scene_slider.set_on_value_changed(self._on_scene_slider_changed)
        self.panel.add_child(self.scene_slider)
        
        # Scene navigation buttons
        scene_button_layout = gui.Horiz()
        self.prev_scene_button = gui.Button("Prev Scene")
        self.prev_scene_button.set_on_clicked(self._on_prev_scene_clicked)
        scene_button_layout.add_child(self.prev_scene_button)
        
        self.next_scene_button = gui.Button("Next Scene")
        self.next_scene_button.set_on_clicked(self._on_next_scene_clicked)
        scene_button_layout.add_child(self.next_scene_button)
        
        self.panel.add_child(scene_button_layout)
        
        self.panel.add_fixed(em)
        
        # Frame slider
        self.frame_label = gui.Label(f"Frame: 1/{self.max_frame + 1}")
        self.panel.add_child(self.frame_label)
        
        self.frame_slider = gui.Slider(gui.Slider.INT)
        self.frame_slider.set_limits(0, self.max_frame)
        self.frame_slider.set_on_value_changed(self._on_frame_slider_changed)
        self.panel.add_child(self.frame_slider)
        
        # Frame navigation buttons
        frame_button_layout = gui.Horiz()
        self.prev_frame_button = gui.Button("Prev Frame")
        self.prev_frame_button.set_on_clicked(self._on_prev_frame_clicked)
        frame_button_layout.add_child(self.prev_frame_button)
        
        self.next_frame_button = gui.Button("Next Frame")
        self.next_frame_button.set_on_clicked(self._on_next_frame_clicked)
        frame_button_layout.add_child(self.next_frame_button)
        
        self.panel.add_child(frame_button_layout)
        
        self.panel.add_fixed(em)
        
        # Bounding box toggle
        self.bbox_checkbox = gui.Checkbox("Show Bounding Boxes")
        self.bbox_checkbox.checked = True
        self.bbox_checkbox.set_on_checked(self._on_bbox_toggle)
        self.panel.add_child(self.bbox_checkbox)
        
        self.panel.add_fixed(em)
        
        # Info label
        self.info_label = gui.Label("")
        self.panel.add_child(self.info_label)
        
        self.panel.add_fixed(em)
        
        # Cache info label
        self.cache_label = gui.Label("")
        self.panel.add_child(self.cache_label)
        
        self.panel.add_fixed(em)
        
        # Reset view button
        self.reset_button = gui.Button("Reset Camera")
        self.reset_button.set_on_clicked(self._on_reset_view)
        self.panel.add_child(self.reset_button)
    
    def _on_layout(self, layout_context):
        '''Handle window layout.'''
        r = self.window.content_rect
        
        # Panel width (right side)
        panel_width = 300
        
        # Scene takes remaining space
        self.scene_widget.frame = gui.Rect(
            r.x, r.y, r.width - panel_width, r.height
        )
        
        # Panel on right side
        self.panel.frame = gui.Rect(
            r.get_right() - panel_width, r.y, panel_width, r.height
        )
    
    def _on_sensor_changed(self, index):
        '''Handle sensor type radio button change.'''
        sensor_map = {0: 'lidar', 1: 'semantic-lidar', 2: 'radar'}
        self.sensor_type = sensor_map[index]
        
        # Update point size based on sensor
        if self.sensor_type == 'lidar':
            self.point_size = 2.0
        elif self.sensor_type == 'semantic-lidar':
            self.point_size = 3.0
        elif self.sensor_type == 'radar':
            self.point_size = 5.0
        
        self._update_frame()
    
    def _on_scene_slider_changed(self, value):
        '''Handle scene slider value change.'''
        self.current_scene = int(value)
        self.current_frame = 0  # Reset to first frame of new scene
        
        # Update frame slider limits
        self.max_frame = self.data_loader.get_frame_count(self.current_scene) - 1
        self.frame_slider.set_limits(0, self.max_frame)
        self.frame_slider.int_value = 0
        
        self._update_frame()
    
    def _on_frame_slider_changed(self, value):
        '''Handle frame slider value change.'''
        self.current_frame = int(value)
        self._update_frame()
    
    def _on_bbox_toggle(self, checked):
        '''Handle bbox checkbox toggle.'''
        self.show_bbox = checked
        self._update_frame()
    
    def _on_prev_scene_clicked(self):
        '''Navigate to previous scene.'''
        if self.current_scene > 0:
            self.current_scene -= 1
            self.scene_slider.int_value = self.current_scene
            self._on_scene_slider_changed(self.current_scene)
    
    def _on_next_scene_clicked(self):
        '''Navigate to next scene.'''
        if self.current_scene < self.max_scene:
            self.current_scene += 1
            self.scene_slider.int_value = self.current_scene
            self._on_scene_slider_changed(self.current_scene)
    
    def _on_prev_frame_clicked(self):
        '''Navigate to previous frame.'''
        if self.current_frame > 0:
            self.current_frame -= 1
            self.frame_slider.int_value = self.current_frame
            self._update_frame()
    
    def _on_next_frame_clicked(self):
        '''Navigate to next frame.'''
        if self.current_frame < self.max_frame:
            self.current_frame += 1
            self.frame_slider.int_value = self.current_frame
            self._update_frame()
    
    def _on_reset_view(self):
        '''Reset camera view.'''
        bounds = self.scene_widget.scene.bounding_box
        self.scene_widget.setup_camera(60, bounds, bounds.get_center())
    
    def _update_frame(self):
        '''Update visualization to current scene and frame (loads data on-demand).'''
        # Load data for current frame and sensor
        sensor_data = self.data_loader.load_frame(
            self.current_scene,
            self.current_frame,
            self.sensor_type
        )
        
        # Update point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(sensor_data['points'])
        
        if 'colors' in sensor_data and sensor_data['colors'] is not None:
            pcd.colors = o3d.utility.Vector3dVector(sensor_data['colors'])
        else:
            colors = np.tile([0.5, 0.5, 0.5], (len(sensor_data['points']), 1))
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Set point cloud material
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = self.point_size
        
        # Remove old point cloud if exists
        if self.scene_widget.scene.has_geometry("point_cloud"):
            self.scene_widget.scene.remove_geometry("point_cloud")
        
        self.scene_widget.scene.add_geometry("point_cloud", pcd, mat)
        
        # Update bounding boxes
        self._update_bboxes(sensor_data.get('bboxes', []))
        
        # Update labels
        scene_number = self.data_loader.get_scene_number(self.current_scene)
        self.scene_label.text = f"Scene: {scene_number:04d} ({self.current_scene + 1}/{self.max_scene + 1})"
        self.frame_label.text = f"Frame: {self.current_frame + 1}/{self.max_frame + 1}"
        self.info_label.text = (
            f"Sensor: {self.sensor_type.replace('-', ' ').title()}\n"
            f"Points: {len(sensor_data['points'])}\n"
            f"BBoxes: {len(sensor_data.get('bboxes', []))}"
        )
        
        # Update cache info
        cache_size = len(self.data_loader.cache)
        self.cache_label.text = f"Cache: {cache_size}/{self.data_loader.cache_size} frames"

        # Prefetch adjacent frames
        # self._prefetch_adjacent_frames()
    
    def _update_bboxes(self, bboxes):
        '''Update bounding box visualization.'''
        # Remove old bounding boxes
        for i in range(100):
            if self.scene_widget.scene.has_geometry(f"bbox_{i}"):
                self.scene_widget.scene.remove_geometry(f"bbox_{i}")
        
        if not self.show_bbox or not bboxes:
            return
        
        # Add new bounding boxes
        for idx, bbox in enumerate(bboxes):
            line_set = self._create_bbox_lineset(
                bbox['corners'],
                bbox.get('label', 'car')
            )
            
            mat = o3d.visualization.rendering.MaterialRecord()
            mat.shader = "unlitLine"
            mat.line_width = 2.0
            
            self.scene_widget.scene.add_geometry(f"bbox_{idx}", line_set, mat)
    
    def _create_bbox_lineset(self, corners, label):
        '''Create Open3D LineSet for bounding box.'''
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Front face
            [4, 5], [5, 6], [6, 7], [7, 4],  # Back face
            [0, 4], [1, 5], [2, 6], [3, 7],  # Connecting edges
        ]
        
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corners)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        
        color = BBOX_COLORS.get(label, [1.0, 1.0, 1.0])
        colors = [color for _ in range(len(lines))]
        line_set.colors = o3d.utility.Vector3dVector(colors)
        
        return line_set
    
    def run(self):
        '''Start interactive visualization.'''
        gui.Application.instance.run()
    
    def _prefetch_adjacent_frames(self):
        '''Prefetch next and previous frames in background.'''
        
        def prefetch():
            # Prefetch next frame
            if self.current_frame < self.max_frame:
                self.data_loader.load_frame(
                    self.current_scene,
                    self.current_frame + 1,
                    self.sensor_type
                )
            
            # Prefetch previous frame
            if self.current_frame > 0:
                self.data_loader.load_frame(
                    self.current_scene,
                    self.current_frame - 1,
                    self.sensor_type
                )
        
        # Run in background thread
        threading.Thread(target=prefetch, daemon=True).start()


def visualize_interactive(ctx):
    '''
    Unified interactive visualization for all sensor types with lazy loading.
    '''
    # Load metadata
    metadata = None
    for split in ['train', 'val', 'test']:
        info_path = f'{ctx.path}/simbev/infos/simbev_infos_{split}.json'
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                infos = json.load(f)
            metadata = infos['metadata']
            break
    
    if metadata is None:
        print("Error: Could not load metadata.")
        return
    
    # Create lazy data loader with cache
    print("Initializing data loader...")
    data_loader = LazyDataLoader(ctx.path, metadata, cache_size=160)
    
    scene_count = data_loader.get_scene_count()
    if scene_count == 0:
        print("No scenes found to visualize.")
        return
    
    print(f"Found {scene_count} scene(s).")
    print(f"Cache size: {data_loader.cache_size} frames (adjustable)")
    
    # Create and run visualizer
    visualizer = AdvancedVisualizer(
        data_loader=data_loader,
        title='SimBEV Interactive Viewer',
        point_size=2.0
    )
    visualizer.run()