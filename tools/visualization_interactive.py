# Academic Software License: Copyright © 2025 Goodarz Mehr.

import os
import time
import json
import threading

import numpy as np
import open3d as o3d

import open3d.visualization.gui as gui

from pyquaternion import Quaternion as Q

from .visualization_utils import RANGE, RAINBOW
from .visualization_handlers import RAD_NAME, LABEL_COLORS

from .visualization_utils import get_global2sensor, transform_bbox


BBOX_COLORS = {
    'car': [0.0, 0.5, 0.9375],
    'truck': [0.5, 0.9375, 0.25],
    'bus': [0.0, 0.5625, 0.0],
    'motorcycle': [0.9375, 0.9375, 0.0],
    'bicycle': [0.0, 0.9375, 0.9375],
    'rider': [0.9375, 0.5625, 0.0],
    'pedestrian': [0.9375, 0.0, 0.0],
    'traffic_light': [0.9375, 0.625, 0.0],
    'traffic_sign': [0.9375, 0.0, 0.5]
}

RAD_NAME = ['RAD_LEFT', 'RAD_FRONT', 'RAD_RIGHT', 'RAD_BACK']


class VizDataLoader:
    '''
    Data loader that loads point cloud and bounding box data for
    interactive visualization and implements caching to avoid reloading
    recently accessed frames.

    Args:
        path: root directory of the dataset.
        metadata: dataset metadata.
        cache_size: number of frames to cache.
        ignore_valid_flag: whether to ignore the valid_flag of object bounding
            boxes.
    '''
    def __init__(self, path: str, metadata: dict, cache_size: int = 10, ignore_valid_flag: bool = False):
        self._path = path
        self._metadata = metadata
        self.cache_size = cache_size
        self._ignore_valid_flag = ignore_valid_flag

        self.cache = {}  # {(scene, frame, sensor_type): data}
        
        self._cache_order = []
        
        # Load scene structure (just paths, not data).
        self._scene_info = self._load_scene_structure()

        # Track prefetch threads
        self._prefetch_thread = None
        self._prefetch_stop = False
    
    def _load_scene_structure(self):
        '''
        Load scene metadata and frame paths without loading the actual data.
        
        Returns:
            List of scene information dictionaries.
        '''
        scene_info = []
        
        for split in ['train', 'val', 'test']:
            info_path = f'{self._path}/simbev/infos/simbev_infos_{split}.json'
            
            if not os.path.exists(info_path):
                continue
            
            with open(info_path, 'r') as f:
                infos = json.load(f)

            for key, value in infos['data'].items():
                scene_number = int(key.split('_')[1])
                scene_data = value['scene_data']

                scene_info.append({
                    'scene_number': scene_number,
                    'frame_count': len(scene_data),
                    'frame_paths': scene_data,
                    'split': split
                })
        
        scene_info.sort(key=lambda x: x['scene_number'])
        
        return scene_info
    
    def get_scene_count(self):
        '''Get the total number of scenes.'''
        return len(self._scene_info)
    
    def get_frame_count(self, idx: int):
        '''
        Get the number of frames in a scene.
        
        Args:
            idx: scene index (0-based).
        
        Returns:
            Number of frames in the scene.
        '''
        return self._scene_info[idx]['frame_count']
    
    def get_scene_number(self, idx: int):
        '''
        Get the scene number.
        
        Args:
            idx: scene index (0-based).
        
        Returns:
            Scene number.
        '''
        return self._scene_info[idx]['scene_number']
    
    def _add_to_cache(self, key: tuple, data: dict):
        '''
        Add data to cache.

        Args:
            key: cache key.
            data: data to cache.
        '''
        # Remove the oldest if cache is full.
        if len(self.cache) >= self.cache_size:
            try:
                oldest_key = self._cache_order.pop(0)
                
                del self.cache[oldest_key]
            except KeyError:
                print('Warning: Tried to delete a non-existing cache key.')

        self.cache[key] = data
        
        self._cache_order.append(key)
    
    def _get_from_cache(self, key: tuple):
        '''
        Get data from cache and update the cache order.
        
        Args:
            key: cache key.

        Returns:
            Cached data.
        '''
        if key in self.cache:
            # Move to end (most recently used)
            self._cache_order.remove(key)
            self._cache_order.append(key)
            
            return self.cache[key]
        
        return None

    def load_frame(self, scene: int, frame: int, sensor_type: str):
        '''
        Load data for a specific frame and sensor type.
        
        Args:
            scene: scene index (0-based).
            frame: frame index within scene (0-based).
            sensor_type: 'lidar', 'semantic-lidar', or 'radar'.

        Returns:
            Dictionary with 'points', 'colors', and 'bboxes'.
        '''
        # Check cache first.
        cache_key = (scene, frame, sensor_type)
        
        cached_data = self._get_from_cache(cache_key)
        
        if cached_data is not None:
            return cached_data
        
        # Load from disk.
        scene_info = self._scene_info[scene]
        
        frame_data = scene_info['frame_paths'][frame]
        
        # Load bounding boxes (shared across all sensors).
        gt_det = np.load(frame_data['GT_DET'], allow_pickle=True)
        
        global2lidar = get_global2sensor(frame_data, self._metadata, 'LIDAR')

        corners, labels = transform_bbox(gt_det, global2lidar, self._ignore_valid_flag)

        bboxes = [{'corners': c, 'label': l} for c, l in zip(corners, labels)]
        
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
        
        # Add to cache.
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
            np.interp(log_normalized, RANGE, RAINBOW[:, 0]),
            np.interp(log_normalized, RANGE, RAINBOW[:, 1]),
            np.interp(log_normalized, RANGE, RAINBOW[:, 2])
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
        colors = LABEL_COLORS[seg_labels]
        
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
            radar2lidar[:3, :3] = Q(self._metadata[radar]['sensor2lidar_rotation']).rotation_matrix
            radar2lidar[:3, 3] = self._metadata[radar]['sensor2lidar_translation']
            
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
            np.interp(log_velocity_normalized, RANGE, RAINBOW[:, 0]),
            np.interp(log_velocity_normalized, RANGE, RAINBOW[:, 1]),
            np.interp(log_velocity_normalized, RANGE, RAINBOW[:, 2])
        ]
        
        return {
            'points': point_cloud,
            'colors': colors,
            'bboxes': bboxes
        }
    
    def prefetch_scene(self, scene_idx: int, sensor_type: str, progress_callback=None):
        '''
        Prefetch all frames for a specific scene and sensor type.
        
        Args:
            scene_idx: scene index (0-based).
            sensor_type: 'lidar', 'semantic-lidar', or 'radar'.
            progress_callback: optional callback function(current, total) for progress updates.
        '''
        scene_info = self._scene_info[scene_idx]
        frame_count = scene_info['frame_count']
        
        print(f"Prefetching scene {self.get_scene_number(scene_idx):04d}, sensor: {sensor_type} ({frame_count} frames)...")
        
        for frame_idx in range(frame_count):
            # Check if we should stop prefetching
            if self._prefetch_stop:
                print("Prefetch cancelled.")
                self._prefetch_stop = False
                return
            
            # Load frame (will use cache if already loaded)
            cache_key = (scene_idx, frame_idx, sensor_type)
            if cache_key not in self.cache:
                self.load_frame(scene_idx, frame_idx, sensor_type)
            
            # Report progress
            if progress_callback:
                progress_callback(frame_idx + 1, frame_count)
        
        print(f"Prefetch complete for scene {self.get_scene_number(scene_idx):04d}, sensor: {sensor_type}")
    
    def prefetch_scene_async(self, scene_idx: int, sensor_type: str, progress_callback=None, completion_callback=None):
        '''
        Prefetch all frames for a scene and sensor type in background thread.
        
        Args:
            scene_idx: scene index (0-based).
            sensor_type: 'lidar', 'semantic-lidar', or 'radar'.
            progress_callback: optional callback function(current, total) for progress updates.
            completion_callback: optional callback function() called when prefetch completes.
        '''
        # Stop any existing prefetch
        self.stop_prefetch()
        
        def prefetch_worker():
            try:
                self.prefetch_scene(scene_idx, sensor_type, progress_callback)
                if completion_callback:
                    completion_callback()
            except Exception as e:
                print(f"Prefetch error: {e}")
        
        self._prefetch_thread = threading.Thread(target=prefetch_worker, daemon=True)
        self._prefetch_thread.start()
    
    def stop_prefetch(self):
        '''Stop any ongoing prefetch operation.'''
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_stop = True
            self._prefetch_thread.join(timeout=1.0)
            self._prefetch_thread = None


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
        self.is_playing = False  # Playback state
        self.play_speed = 30  # FPS for playback

        # Prefetch state
        self.is_prefetching = False
        self.prefetch_progress = 0
        self.prefetch_total = 0
        
        # Create 3D scene widget
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = o3d.visualization.rendering.Open3DScene(
            self.window.renderer
        )
        
        # Set up scene rendering
        self.scene_widget.scene.set_background([0.1, 0.1, 0.1, 1.0])
        
        # Register keyboard callbacks for Open3D controls
        self.scene_widget.set_on_key(self._on_key_event)
        
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

        # Start prefetching initial scene
        self._start_prefetch()
        
        print("\n=== Keyboard Controls ===")
        print("+ / =    : Increase point size")
        print("- / _    : Decrease point size")
        print("Space    : Toggle bounding boxes")
        print("Left     : Previous frame")
        print("Right    : Next frame")
        print("Up       : Previous scene")
        print("Down     : Next scene")
        print("T        : Top-down view")
        print("V        : Perspective view (3D)")
        print("P        : Play/Pause animation")
        print("========================\n")
    
    def _on_key_event(self, event):
        '''Handle keyboard events.'''
        # Return True if event is handled, False otherwise
        if event.type == gui.KeyEvent.DOWN:
            # Increase point size
            if event.key == ord('+') or event.key == ord('='):
                self.point_size = min(self.point_size + 1.0, 20.0)
                self._update_point_size()
                print(f"Point size: {self.point_size}")
                return gui.Widget.EventCallbackResult.HANDLED
            
            # Decrease point size
            elif event.key == ord('-') or event.key == ord('_'):
                self.point_size = max(self.point_size - 1.0, 1.0)
                self._update_point_size()
                print(f"Point size: {self.point_size}")
                return gui.Widget.EventCallbackResult.HANDLED
            
            # Toggle bounding boxes
            elif event.key == gui.KeyName.SPACE:
                self.show_bbox = not self.show_bbox
                self.bbox_checkbox.checked = self.show_bbox
                print(f"Bounding boxes: {'ON' if self.show_bbox else 'OFF'}")
                self._update_frame()
                return gui.Widget.EventCallbackResult.HANDLED
            
            # Previous frame
            elif event.key == gui.KeyName.LEFT:
                self._on_prev_frame_clicked()
                return gui.Widget.EventCallbackResult.HANDLED
            
            # Next frame
            elif event.key == gui.KeyName.RIGHT:
                self._on_next_frame_clicked()
                return gui.Widget.EventCallbackResult.HANDLED
            
            # Previous scene
            elif event.key == gui.KeyName.UP:
                self._on_prev_scene_clicked()
                return gui.Widget.EventCallbackResult.HANDLED
            
            # Next scene
            elif event.key == gui.KeyName.DOWN:
                self._on_next_scene_clicked()
                return gui.Widget.EventCallbackResult.HANDLED
            
            # Top-down view
            elif event.key == ord('T') or event.key == ord('t'):
                self._on_topdown_view()
                return gui.Widget.EventCallbackResult.HANDLED
            
            # Perspective view
            elif event.key == ord('V') or event.key == ord('v'):
                self._on_perspective_view()
                return gui.Widget.EventCallbackResult.HANDLED
            
            # Play/Pause
            elif event.key == ord('P') or event.key == ord('p'):
                self._toggle_playback()
                return gui.Widget.EventCallbackResult.HANDLED
        
        return gui.Widget.EventCallbackResult.IGNORED
    
    def _update_point_size(self):
        '''Update point cloud rendering size by reloading current frame.'''
        # Update UI labels
        self.point_size_slider.double_value = self.point_size
        self.point_size_label.text = f"{self.point_size:.1f}"
        
        # Reload current frame with new point size
        # This will use cached data, so it's fast
        self._update_frame()
    
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
        
        # Playback controls
        playback_layout = gui.Horiz()
        
        self.play_button = gui.Button("▶ Play")
        self.play_button.set_on_clicked(self._on_play_clicked)
        playback_layout.add_child(self.play_button)
        
        self.panel.add_child(playback_layout)
        
        self.panel.add_fixed(em / 2)
        
        # Playback speed slider
        self.panel.add_child(gui.Label("Playback Speed (FPS):"))
        
        self.speed_slider = gui.Slider(gui.Slider.INT)
        self.speed_slider.set_limits(1, 60)
        self.speed_slider.int_value = self.play_speed
        self.speed_slider.set_on_value_changed(self._on_speed_changed)
        self.panel.add_child(self.speed_slider)
        
        self.speed_label = gui.Label(f"{self.play_speed} FPS")
        self.panel.add_child(self.speed_label)
        
        self.panel.add_fixed(em)
        
        # Bounding box toggle
        self.bbox_checkbox = gui.Checkbox("Show Bounding Boxes")
        self.bbox_checkbox.checked = True
        self.bbox_checkbox.set_on_checked(self._on_bbox_toggle)
        self.panel.add_child(self.bbox_checkbox)
        
        self.panel.add_fixed(em)
        
        # Point size control
        self.panel.add_child(gui.Label("Point Size:"))
        
        self.point_size_slider = gui.Slider(gui.Slider.DOUBLE)
        self.point_size_slider.set_limits(1.0, 20.0)
        self.point_size_slider.double_value = self.point_size
        self.point_size_slider.set_on_value_changed(self._on_point_size_slider_changed)
        self.panel.add_child(self.point_size_slider)
        
        self.point_size_label = gui.Label(f"{self.point_size:.1f}")
        self.panel.add_child(self.point_size_label)
        
        self.panel.add_fixed(em)
        
        # Info label
        self.info_label = gui.Label("")
        self.panel.add_child(self.info_label)
        
        self.panel.add_fixed(em)
        
        # Cache info label
        self.cache_label = gui.Label("")
        self.panel.add_child(self.cache_label)
        
        self.panel.add_fixed(em)
        
        # Prefetch progress label
        self.prefetch_label = gui.Label("")
        self.panel.add_child(self.prefetch_label)
        
        self.panel.add_fixed(em)
        
        # Camera view buttons
        self.panel.add_child(gui.Label("Camera View:"))
        
        view_button_layout = gui.Horiz()
        
        self.topdown_button = gui.Button("Top-Down")
        self.topdown_button.set_on_clicked(self._on_topdown_view)
        view_button_layout.add_child(self.topdown_button)
        
        self.perspective_button = gui.Button("Perspective")
        self.perspective_button.set_on_clicked(self._on_perspective_view)
        view_button_layout.add_child(self.perspective_button)
        
        self.panel.add_child(view_button_layout)
    
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
        
        # Update default point size based on sensor
        if self.sensor_type == 'lidar':
            default_size = 2.0
        elif self.sensor_type == 'semantic-lidar':
            default_size = 3.0
        elif self.sensor_type == 'radar':
            default_size = 5.0
        
        self.point_size = default_size
        self.point_size_slider.double_value = default_size
        self.point_size_label.text = f"{default_size:.1f}"
        
        self._update_frame()
        
        # Start prefetching new sensor type
        self._start_prefetch()
    
    def _on_point_size_slider_changed(self, value):
        '''Handle point size slider change.'''
        self.point_size = value
        self.point_size_label.text = f"{value:.1f}"
        self._update_point_size()
    
    def _on_speed_changed(self, value):
        '''Handle playback speed slider change.'''
        self.play_speed = int(value)
        self.speed_label.text = f"{self.play_speed} FPS"
    
    def _on_play_clicked(self):
        '''Handle play button click.'''
        self._toggle_playback()
    
    def _toggle_playback(self):
        '''Toggle playback on/off.'''
        self.is_playing = not self.is_playing
        
        if self.is_playing:
            self.play_button.text = "⏸ Pause"
            print("Playback started")
            self._start_playback()
        else:
            self.play_button.text = "▶ Play"
            print("Playback stopped")
    
    def _start_playback(self):
        '''Start automatic frame advancement.'''
        def play_loop():
            while self.is_playing:
                # Calculate delay based on FPS
                delay = 1.0 / self.play_speed
                time.sleep(delay)
                
                # Advance to next frame
                if self.current_frame < self.max_frame:
                    # Use post_to_main_thread to safely update GUI from background thread
                    gui.Application.instance.post_to_main_thread(
                        self.window,
                        lambda: self._advance_frame()
                    )
                else:
                    # End of scene - stop playback or loop
                    gui.Application.instance.post_to_main_thread(
                        self.window,
                        lambda: self._stop_playback()
                    )
                    break
        
        # Start playback in background thread
        threading.Thread(target=play_loop, daemon=True).start()
    
    def _advance_frame(self):
        '''Advance to next frame (called from playback thread).'''
        if self.current_frame < self.max_frame:
            self.current_frame += 1
            self.frame_slider.int_value = self.current_frame
            self._update_frame()
    
    def _loop_playback(self):
        '''Loop back to first frame.'''
        self.current_frame = 0
        self.frame_slider.int_value = 0
        self._update_frame()
    
    def _stop_playback(self):
        '''Stop playback.'''
        self.is_playing = False
        self.play_button.text = "▶ Play"
        print("Playback finished")
    
    def _on_scene_slider_changed(self, value):
        '''Handle scene slider value change.'''
        self.current_scene = int(value)
        self.current_frame = 0  # Reset to first frame of new scene
        
        # Update frame slider limits
        self.max_frame = self.data_loader.get_frame_count(self.current_scene) - 1
        self.frame_slider.set_limits(0, self.max_frame)
        self.frame_slider.int_value = 0
        
        self._update_frame()
        
        # Start prefetching new scene
        self._start_prefetch()
    
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
    
    def _on_topdown_view(self):
        '''Set camera to top-down view (bird's eye).'''
        # Get bounding box center
        bounds = self.scene_widget.scene.bounding_box
        center = bounds.get_center()
        
        # Calculate extent for appropriate zoom
        extent = bounds.get_extent()
        max_extent = max(extent[0], extent[1])
        
        # Set camera looking down from above
        # Position: directly above center at appropriate height
        eye = [center[0], center[1], max_extent * 1.5]
        look_at = center
        up = [0, 1, 0]  # Y-axis is up in top-down view
        
        self.scene_widget.look_at(look_at, eye, up)
        print("Camera view: Top-Down")

    def _on_perspective_view(self):
        '''Set camera to 3D perspective view from behind the car.'''
        # Get bounding box center
        bounds = self.scene_widget.scene.bounding_box
        center = bounds.get_center()
        
        # Calculate extent for appropriate distance
        extent = bounds.get_extent()
        max_extent = max(extent[0], extent[1], extent[2])
        
        # Set camera behind and above the car
        # Position: behind (negative Y), elevated (positive Z), centered (X)
        # Assuming car faces forward (positive Y direction)
        distance = max_extent * 2.0
        eye = [center[0], center[1] - distance, center[2] + distance * 0.5]
        look_at = center
        up = [0, 0, 1]  # Z-axis is up in 3D perspective
        
        self.scene_widget.look_at(look_at, eye, up)
        print("Camera view: Perspective (Behind Car)")
    
    def _start_prefetch(self):
        '''Prefetch all sensor types for current scene.'''
        self.is_prefetching = True
        
        sensor_types = ['lidar', 'semantic-lidar', 'radar']
        total_frames = len(sensor_types) * (self.max_frame + 1)
        completed_frames = [0]
        
        def progress_callback(current, total):
            completed_frames[0] += 1
            self.prefetch_progress = completed_frames[0]
            self.prefetch_total = total_frames
            
            gui.Application.instance.post_to_main_thread(
                self.window,
                lambda: self._update_prefetch_label()
            )
        
        def sensor_complete():
            if completed_frames[0] >= total_frames:
                self.is_prefetching = False
                gui.Application.instance.post_to_main_thread(
                    self.window,
                    lambda: self._update_prefetch_label()
                )
        
        # Prefetch each sensor type
        for sensor in sensor_types:
            self.data_loader.prefetch_scene_async(
                self.current_scene,
                sensor,
                progress_callback,
                sensor_complete
            )
    
    def _update_prefetch_label(self):
        '''Update prefetch progress label.'''
        if self.is_prefetching:
            percent = (self.prefetch_progress / self.prefetch_total * 100) if self.prefetch_total > 0 else 0
            self.prefetch_label.text = f"Prefetching: {self.prefetch_progress}/{self.prefetch_total} ({percent:.0f}%)"
        else:
            self.prefetch_label.text = "Prefetch: Complete ✓"
    
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
            [0, 1], [1, 3], [3, 2], [2, 0],  # Front face
            [4, 5], [5, 7], [7, 6], [6, 4],  # Back face
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
    data_loader = VizDataLoader(ctx.path, metadata, cache_size=1600, ignore_valid_flag=ctx.ignore_valid_flag)
    
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