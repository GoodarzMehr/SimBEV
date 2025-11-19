# Academic Software License: Copyright © 2025 Goodarz Mehr.

import os
import time
import json
import threading

import numpy as np
import open3d as o3d

import open3d.visualization.gui as gui

from pyquaternion import Quaternion as Q

from concurrent.futures import ThreadPoolExecutor

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


class VizDataLoader:
    '''
    Data loader that loads point cloud and bounding box data for interactive
    visualization.

    Args:
        path: root directory of the dataset.
        metadata: dataset metadata.
        ignore_valid_flag: whether to ignore the valid_flag of object bounding
            boxes.
        max_workers: number of workers for loading data in parallel.
        max_cache_size: maximum number of scenes to keep in cache.
    '''
    def __init__(
        self,
        path: str,
        metadata: dict,
        ignore_valid_flag: bool = False,
        max_workers: int = 8,
        max_cached_scenes: int = 3
    ):
        self._path = path
        self._metadata = metadata
        self._ignore_valid_flag = ignore_valid_flag
        self._max_workers = max_workers
        self._max_cached_scenes = max_cached_scenes
        
        # Scene cache: {scene: {sensor_type: [frame_data, ...]}}
        self._scene_cache = {}
        
        # Track scene access order for LRU eviction.
        self._scene_access_order = []
        
        # Load scene structure (just paths, not data).
        self._scene_info = self._load_scene_structure()
        
        # Executor for parallel loading.
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def _load_scene_structure(self) -> list:
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
    
    def get_scene_count(self) -> int:
        '''Get the total number of scenes.'''
        return len(self._scene_info)
    
    def get_frame_count(self, scene: int) -> int:
        '''
        Get the number of frames in a scene.
        
        Args:
            scene: scene index (0-based).
        
        Returns:
            Number of frames in the scene.
        '''
        return self._scene_info[scene]['frame_count']
    
    def get_scene_number(self, scene: int) -> int:
        '''
        Get the scene number.
        
        Args:
            scene: scene index (0-based).
        
        Returns:
            Scene number.
        '''
        return self._scene_info[scene]['scene_number']
    
    def is_scene_loaded(self, scene: int) -> bool:
        '''
        Check if the scene is loaded in cache.
        
        Args:
            scene: scene index (0-based).

        Returns:
            True if scene is loaded in cache, False otherwise.
        '''
        return scene in self._scene_cache
    
    def _evict_oldest_scene(self):
        '''Evict the least recently used scene from cache.'''
        if self._scene_access_order:
            # Get the oldest scene (first in the list).
            oldest_scene = self._scene_access_order[0]
            
            self.unload_scene(oldest_scene)
    
    def _mark_scene_accessed(self, scene: int):
        '''
        Mark a scene as recently accessed (move to the end of the LRU list).
        
        Args:
            scene: scene index (0-based).
        '''
        if scene in self._scene_access_order:
            self._scene_access_order.remove(scene)

        self._scene_access_order.append(scene)

    def load_scene(self, scene: int, progress_callback=None) -> bool:
        '''
        Load the entire scene (all frames, all sensors) into cache.
        Automatically manage cache size by evicting the least recently used
        (LRU) scenes.
        
        Args:
            scene: scene index (0-based).
            progress_callback: optional callback(current, total, message) for progress updates.
        
        Returns:
            True if loaded successfully, False otherwise.
        '''
        if scene in self._scene_cache:
            self._mark_scene_accessed(scene)
            
            if progress_callback:
                progress_callback(1, 1, "Scene already loaded.")
            
            return True
        
        # Evict the oldest scene if the cache is full.
        if len(self._scene_cache) >= self._max_cached_scenes:
            self._evict_oldest_scene()

        print(f"\nLoading scene {self.get_scene_number(scene):04d}...")

        start = time.perf_counter()

        scene_info = self._scene_info[scene]
        
        frame_count = scene_info['frame_count']
        
        sensor_types = ['lidar', 'semantic-lidar', 'radar']
        
        # Initialize the cache for this scene.
        scene_data = {sensor_type: [None] * frame_count for sensor_type in sensor_types}
        
        # Create tasks for parallel loading
        tasks = []
        
        total_tasks = frame_count * len(sensor_types)
        
        for sensor_type in sensor_types:
            for frame in range(frame_count):
                task = self._executor.submit(self._load_single_frame, scene, frame, sensor_type)
                
                tasks.append((task, frame, sensor_type))

        # Wait for all tasks to complete and update cache.
        completed = 0
        
        for task, frame, sensor_type in tasks:
            try:
                frame_data = task.result()
                
                scene_data[sensor_type][frame] = frame_data
                
                completed += 1
                
                if progress_callback and completed % 10 == 0:
                    progress_callback(completed, total_tasks, f'Loading: {completed}/{total_tasks}')
            
            except Exception as e:
                print(f'Error while loading scene {scene}, frame {frame}, sensor {sensor_type}: {e}')
                
                scene_data[sensor_type][frame] = {'points': np.empty((0, 3)), 'colors': None, 'bboxes': []}
        
        # Store in cache.
        self._scene_cache[scene] = scene_data
        
        self._mark_scene_accessed(scene)
        
        elapsed = time.perf_counter() - start
        
        print(f'Scene {self.get_scene_number(scene):04d} loaded in {elapsed:.2f} s.')
        
        if progress_callback:
            progress_callback(total_tasks, total_tasks, "Scene loaded.")
        
        return True
    
    def _load_single_frame(self, scene: int, frame: int, sensor_type: str) -> dict:
        '''
        Worker for loading a single frame for a specific sensor type.

        Args:
            scene: scene index (0-based).
            frame: frame index (0-based).
            sensor_type: type of sensor data to load.
        
        Returns:
            Frame data dictionary with 'points', 'colors', 'bboxes'.
        '''
        scene_info = self._scene_info[scene]
        
        frame_data = scene_info['frame_paths'][frame]
        
        # Load bounding boxes.
        gt_det = np.load(frame_data['GT_DET'], allow_pickle=True)
        
        global2lidar = get_global2sensor(frame_data, self._metadata, 'LIDAR')
        
        corners, labels = transform_bbox(gt_det, global2lidar, self._ignore_valid_flag)
        
        bboxes = [{'corners': c, 'label': l} for c, l in zip(corners, labels)]
        
        if sensor_type == 'lidar':
            if 'LIDAR' in frame_data:
                return self._load_lidar(frame_data, bboxes)
            else:
                return {'points': np.empty((0, 3)), 'colors': None, 'bboxes': bboxes}
        elif sensor_type == 'semantic-lidar':
            if 'SEG-LIDAR' in frame_data:
                return self._load_semantic_lidar(frame_data, bboxes)
            else:
                return {'points': np.empty((0, 3)), 'colors': None, 'bboxes': bboxes}
        elif sensor_type == 'radar':
            if all(radar in frame_data for radar in RAD_NAME):
                return self._load_radar(frame_data, bboxes)
            else:
                return {'points': np.empty((0, 3)), 'colors': None, 'bboxes': bboxes}
        else:
            raise ValueError(f'Unknown sensor type: {sensor_type}')
    
    def _load_lidar(self, frame_data: dict, bboxes: list) -> dict:
        '''
        Load lidar point clouds.
        
        Args:
            frame_data: dictionary of frame data.
            bboxes: list of bounding boxes.
        
        Returns:
            Dictionary with 'points', 'colors', 'bboxes'.
        '''
        point_cloud = np.load(frame_data['LIDAR'])['data']
        
        # Compute distance-based colors
        distances = np.linalg.norm(point_cloud, axis=1)
        
        log_distances = np.log(distances + 1e-6)
        
        log_normalized = (log_distances - log_distances.min()) / (log_distances.max() - log_distances.min() + 1e-6)
        
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
    
    def _load_semantic_lidar(self, frame_data: dict, bboxes: list) -> dict:
        '''
        Load semantic lidar point clouds.
        
        Args:
            frame_data: dictionary of frame data.
            bboxes: list of bounding boxes.
        
        Returns:
            Dictionary with 'points', 'colors', 'bboxes'.
        '''
        data = np.load(frame_data['SEG-LIDAR'])['data']
        
        point_cloud = np.array([data['x'], data['y'], data['z']]).T
        
        seg_labels = np.array(data['ObjTag'])
        
        colors = LABEL_COLORS[seg_labels]
        
        return {
            'points': point_cloud,
            'colors': colors,
            'bboxes': bboxes
        }
    
    def _load_radar(self, frame_data: dict, bboxes: list) -> dict:
        '''
        Load radar point clouds.
        
        Args:
            frame_data: dictionary of frame data.
            bboxes: list of bounding boxes.
        
        Returns:
            Dictionary with 'points', 'colors', 'bboxes'.
        '''
        point_cloud_list = []
        velocity_list = []
        
        for radar in RAD_NAME:
            radar2lidar = np.eye(4, dtype=np.float32)
            
            radar2lidar[:3, :3] = Q(self._metadata[radar]['sensor2lidar_rotation']).rotation_matrix
            radar2lidar[:3, 3] = self._metadata[radar]['sensor2lidar_translation']
            
            radar_points = np.load(frame_data[radar])['data']
            
            velocity_list.append(radar_points[:, -1])
            
            radar_points = radar_points[:, :-1]
            
            # Transform depth, altitude, and azimuth data to x, y, and z.
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
    
    def get_frame(self, scene: int, frame: int, sensor_type: str) -> dict:
        '''
        Get frame data from cache. Scene must be loaded first.
        
        Args:
            scene: scene index (0-based).
            frame: frame index (0-based).
            sensor_type: 'lidar', 'semantic-lidar', or 'radar'.
        
        Returns:
            Frame data dictionary with 'points', 'colors', 'bboxes'.
        '''
        if scene not in self._scene_cache:
            raise RuntimeError(f'Scene {scene} not loaded. Call load_scene() first.')
        
        self._mark_scene_accessed(scene)
        
        return self._scene_cache[scene][sensor_type][frame]
    
    def unload_scene(self, scene: int):
        '''
        Unload scene from cache to free up memory.

        Args:
            scene: scene index (0-based).
        '''
        if scene in self._scene_cache:
            del self._scene_cache[scene]
            
            if scene in self._scene_access_order:
                self._scene_access_order.remove(scene)
            
            print(f'Scene {self.get_scene_number(scene):04d} unloaded from cache.')
    
    def get_cache_info(self):
        '''Get information about the current state of the cache.'''
        return {
            'cached_scenes': len(self._scene_cache),
            'max_cached_scenes': self._max_cached_scenes,
            'cached_scene_indices': list(self._scene_cache.keys()),
            'access_order': self._scene_access_order.copy()
        }
    
    def cleanup(self):
        '''Clean up resources.'''
        self._executor.shutdown(wait=False)
        self._scene_cache.clear()
        self._scene_access_order.clear()


class AdvancedVisualizer:
    '''
    Advanced interactive visualizer with GUI slider and controls.
    Loads entire scenes on demand with automatic cache management.
    Uses Open3D's tick event for smooth animation on the main thread.
    
    Args:
        data_loader: VizDataLoader instance.
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
        self.is_playing = False
        self.play_speed = 10  # Start with 10 FPS
        
        # Loading state
        self.is_loading = False
        self.load_progress = 0
        self.load_total = 0
        
        # Playback timing (no threads needed!)
        self._last_frame_time = time.perf_counter()
        self._bbox_count = 0
        
        # Create 3D scene widget
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = o3d.visualization.rendering.Open3DScene(
            self.window.renderer
        )
        
        # Set up scene rendering
        self.scene_widget.scene.set_background([0.1, 0.1, 0.1, 1.0])
        
        # Register keyboard callbacks
        self.scene_widget.set_on_key(self._on_key_event)
        
        # Create control panel
        self._create_control_panel()
        
        # Layout
        self.window.set_on_layout(self._on_layout)
        self.window.add_child(self.scene_widget)
        self.window.add_child(self.panel)
        
        # Add coordinate frame
        self._add_coordinate_frame()
        
        # Set up animation callback - runs every frame on main thread!
        self.window.set_on_tick_event(self._on_tick)
        
        # Load initial scene and display first frame
        self._load_and_display_scene(0)
        
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
    
    def _on_tick(self):
        '''
        Called every frame by the GUI event loop.
        Perfect for animation - runs on main thread!
        No threading issues possible!
        '''
        if not self.is_playing:
            return True  # Continue event loop
        
        # Calculate if we should advance frame based on play speed
        current_time = time.perf_counter()
        elapsed = current_time - self._last_frame_time
        frame_time = 1.0 / self.play_speed
        
        if elapsed >= frame_time:
            self._last_frame_time = current_time
            
            # Advance frame
            if self.current_frame < self.max_frame:
                self.current_frame += 1
                self.frame_slider.int_value = self.current_frame
                self._update_frame()
            else:
                # Reached end of sequence
                self._stop_playback()
        
        return True  # Continue event loop
    
    def _load_and_display_scene(self, scene: int):
        '''Load a scene and display the first frame.'''
        if self.data_loader.is_scene_loaded(scene):
            # Already loaded, just update display
            self._update_frame()
        else:
            # Need to load scene
            self.is_loading = True
            self._update_loading_label("Loading scene...")
            
            def progress_callback(current, total, message):
                self.load_progress = current
                self.load_total = total
                gui.Application.instance.post_to_main_thread(
                    self.window,
                    lambda: self._update_loading_label(message)
                )
            
            # Load in background thread (only for file I/O)
            def load_worker():
                success = self.data_loader.load_scene(scene, progress_callback)
                
                # Update UI on main thread
                gui.Application.instance.post_to_main_thread(
                    self.window,
                    lambda: self._on_scene_loaded(success)
                )
            
            threading.Thread(target=load_worker, daemon=True).start()
    
    def _on_scene_loaded(self, success: bool):
        '''Called when scene loading completes.'''
        self.is_loading = False
        
        if success:
            self._update_loading_label("Scene loaded ✓")
            self._update_cache_status()
            self._update_frame()
            
            # Setup camera for new scene
            bounds = self.scene_widget.scene.bounding_box
            self.scene_widget.setup_camera(60, bounds, bounds.get_center())
        else:
            self._update_loading_label("Failed to load scene")
    
    def _update_loading_label(self, message: str):
        '''Update loading status label.'''
        if self.is_loading:
            percent = (self.load_progress / self.load_total * 100) if self.load_total > 0 else 0
            self.loading_label.text = f"{message} ({percent:.0f}%)"
        else:
            self.loading_label.text = message
    
    def _on_key_event(self, event):
        '''Handle keyboard events.'''
        if event.type == gui.KeyEvent.DOWN:
            if event.key == ord('+') or event.key == ord('='):
                self.point_size = min(self.point_size + 1.0, 20.0)
                self._update_point_size()
                print(f"Point size: {self.point_size}")
                return gui.Widget.EventCallbackResult.HANDLED
            
            elif event.key == ord('-') or event.key == ord('_'):
                self.point_size = max(self.point_size - 1.0, 1.0)
                self._update_point_size()
                print(f"Point size: {self.point_size}")
                return gui.Widget.EventCallbackResult.HANDLED
            
            elif event.key == gui.KeyName.SPACE:
                self.show_bbox = not self.show_bbox
                self.bbox_checkbox.checked = self.show_bbox
                print(f"Bounding boxes: {'ON' if self.show_bbox else 'OFF'}")
                self._update_frame()
                return gui.Widget.EventCallbackResult.HANDLED
            
            elif event.key == gui.KeyName.LEFT:
                self._on_prev_frame_clicked()
                return gui.Widget.EventCallbackResult.HANDLED
            
            elif event.key == gui.KeyName.RIGHT:
                self._on_next_frame_clicked()
                return gui.Widget.EventCallbackResult.HANDLED
            
            elif event.key == gui.KeyName.UP:
                self._on_prev_scene_clicked()
                return gui.Widget.EventCallbackResult.HANDLED
            
            elif event.key == gui.KeyName.DOWN:
                self._on_next_scene_clicked()
                return gui.Widget.EventCallbackResult.HANDLED
            
            elif event.key == ord('T') or event.key == ord('t'):
                self._on_topdown_view()
                return gui.Widget.EventCallbackResult.HANDLED
            
            elif event.key == ord('V') or event.key == ord('v'):
                self._on_perspective_view()
                return gui.Widget.EventCallbackResult.HANDLED
            
            elif event.key == ord('P') or event.key == ord('p'):
                self._toggle_playback()
                return gui.Widget.EventCallbackResult.HANDLED
        
        return gui.Widget.EventCallbackResult.IGNORED
    
    def _update_point_size(self):
        '''Update point cloud rendering size by reloading current frame.'''
        self.point_size_slider.double_value = self.point_size
        self.point_size_label.text = f"{self.point_size:.1f}"
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
        self.speed_slider.set_limits(1, 30)
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
        
        # Loading status label
        self.loading_label = gui.Label("")
        self.panel.add_child(self.loading_label)
        
        self.panel.add_fixed(em)
        
        # Cache status label
        self.cache_status_label = gui.Label("")
        self.panel.add_child(self.cache_status_label)
        self._update_cache_status()
        
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
    
    def _update_cache_status(self):
        '''Update cache status label.'''
        cache_info = self.data_loader.get_cache_info()
        cached = cache_info['cached_scenes']
        max_cached = cache_info['max_cached_scenes']
        
        # Show which scenes are cached
        if cache_info['cached_scene_indices']:
            scene_numbers = [self.data_loader.get_scene_number(idx) 
                           for idx in cache_info['cached_scene_indices']]
            scene_str = ', '.join([f"{num:04d}" for num in scene_numbers])
            self.cache_status_label.text = (
                f"Cache: {cached}/{max_cached} scenes\n"
                f"Loaded: {scene_str}"
            )
        else:
            self.cache_status_label.text = f"Cache: {cached}/{max_cached} scenes"
    
    def _on_layout(self, layout_context):
        '''Handle window layout.'''
        r = self.window.content_rect
        panel_width = 300
        
        self.scene_widget.frame = gui.Rect(
            r.x, r.y, r.width - panel_width, r.height
        )
        
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
        '''Toggle playback on/off - simple, no threads!'''
        self.is_playing = not self.is_playing
        
        if self.is_playing:
            self.play_button.text = "⏸ Pause"
            self._last_frame_time = time.perf_counter()
            print("Playback started")
        else:
            self.play_button.text = "▶ Play"
            print("Playback stopped")
    
    def _stop_playback(self):
        '''Stop playback - called when reaching end of sequence.'''
        self.is_playing = False
        self.play_button.text = "▶ Play"
        print("Playback finished")
    
    def _on_scene_slider_changed(self, value):
        '''Handle scene slider value change.'''
        new_scene = int(value)
        if new_scene != self.current_scene:
            # Stop playback before changing scenes
            if self.is_playing:
                self.is_playing = False
                self.play_button.text = "▶ Play"
            
            self.current_scene = new_scene
            self.current_frame = 0
            
            # Update frame slider limits
            self.max_frame = self.data_loader.get_frame_count(self.current_scene) - 1
            self.frame_slider.set_limits(0, self.max_frame)
            self.frame_slider.int_value = 0
            
            # Load and display new scene
            self._load_and_display_scene(self.current_scene)
    
    def _on_frame_slider_changed(self, value):
        '''Handle frame slider value change.'''
        # Stop playback when manually scrubbing
        if self.is_playing:
            self.is_playing = False
            self.play_button.text = "▶ Play"
        
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
        '''Set camera to top-down view.'''
        bounds = self.scene_widget.scene.bounding_box
        center = bounds.get_center()
        extent = bounds.get_extent()
        max_extent = max(extent[0], extent[1])
        
        eye = [center[0], center[1], max_extent * 1.5]
        look_at = center
        up = [0, 1, 0]
        
        self.scene_widget.look_at(look_at, eye, up)
        print("Camera view: Top-Down")

    def _on_perspective_view(self):
        '''Set camera to 3D perspective view.'''
        bounds = self.scene_widget.scene.bounding_box
        center = bounds.get_center()
        extent = bounds.get_extent()
        max_extent = max(extent[0], extent[1], extent[2])
        
        distance = max_extent * 2.0
        eye = [center[0], center[1] - distance, center[2] + distance * 0.5]
        look_at = center
        up = [0, 0, 1]
        
        self.scene_widget.look_at(look_at, eye, up)
        print("Camera view: Perspective (Behind Car)")
    
    def _update_frame(self):
        '''
        Update visualization to current scene and frame.
        Runs on main thread - no locks needed!
        '''
        if self.is_loading:
            return
        
        try:
            # Get data from cache (scene must be loaded)
            sensor_data = self.data_loader.get_frame(
                self.current_scene,
                self.current_frame,
                self.sensor_type
            )
        except RuntimeError as e:
            print(f"Error: {e}")
            return
        except Exception as e:
            print(f"Unexpected error: {e}")
            return
        
        # Validate data
        if sensor_data is None or 'points' not in sensor_data:
            return
        
        try:
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
            
            # Remove old point cloud
            if self.scene_widget.scene.has_geometry("point_cloud"):
                self.scene_widget.scene.remove_geometry("point_cloud")
            
            self.scene_widget.scene.add_geometry("point_cloud", pcd, mat)
            
            # Update bounding boxes
            self._update_bboxes(sensor_data.get('bboxes', []))
            
        except Exception as e:
            print(f"Error updating geometry: {e}")
            return
        
        # Update labels
        scene_number = self.data_loader.get_scene_number(self.current_scene)
        self.scene_label.text = f"Scene: {scene_number:04d} ({self.current_scene + 1}/{self.max_scene + 1})"
        self.frame_label.text = f"Frame: {self.current_frame + 1}/{self.max_frame + 1}"
        self.info_label.text = (
            f"Sensor: {self.sensor_type.replace('-', ' ').title()}\n"
            f"Points: {len(sensor_data['points'])}\n"
            f"BBoxes: {len(sensor_data.get('bboxes', []))}"
        )
        
        # Update cache status
        self._update_cache_status()
    
    def _update_bboxes(self, bboxes):
        '''Update bounding box visualization.'''
        if not hasattr(self, '_bbox_count'):
            self._bbox_count = 0
        
        # Remove old bboxes
        for i in range(self._bbox_count):
            bbox_name = f"bbox_{i}"
            try:
                if self.scene_widget.scene.has_geometry(bbox_name):
                    self.scene_widget.scene.remove_geometry(bbox_name)
            except:
                pass  # Silently ignore removal errors
        
        self._bbox_count = 0
        
        if not self.show_bbox or not bboxes:
            return
        
        # Add new bounding boxes
        for idx, bbox in enumerate(bboxes):
            try:
                line_set = self._create_bbox_lineset(
                    bbox['corners'],
                    bbox.get('label', 'car')
                )
                
                mat = o3d.visualization.rendering.MaterialRecord()
                mat.shader = "unlitLine"
                mat.line_width = 2.0
                
                self.scene_widget.scene.add_geometry(f"bbox_{idx}", line_set, mat)
                self._bbox_count += 1
                
            except Exception as e:
                print(f"Error adding bbox {idx}: {e}")
                continue
    
    def _create_bbox_lineset(self, corners, label):
        '''Create Open3D LineSet for bounding box.'''
        lines = [
            [0, 1], [1, 3], [3, 2], [2, 0],
            [4, 5], [5, 7], [7, 6], [6, 4],
            [0, 4], [1, 5], [2, 6], [3, 7],
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
        
        # Cleanup on exit
        self.data_loader.cleanup()


def visualize_interactive(ctx):
    '''Unified interactive visualization for all sensor types.'''
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
    
    # Create data loader with 3-scene cache
    print("Initializing data loader...")
    data_loader = VizDataLoader(
        ctx.path, 
        metadata, 
        ignore_valid_flag=ctx.ignore_valid_flag,
        max_workers=16,  # Adjust based on your system
        max_cached_scenes=5  # Keep at most 5 scenes in memory
    )
    
    scene_count = data_loader.get_scene_count()
    if scene_count == 0:
        print("No scenes found to visualize.")
        return
    
    print(f"Found {scene_count} scene(s).")
    print(f"Cache limit: 3 scenes")
    
    # Create and run visualizer
    visualizer = AdvancedVisualizer(
        data_loader=data_loader,
        title='SimBEV Interactive Viewer',
        point_size=2.0
    )
    visualizer.run()