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
        self.max_cached_scenes = max_cached_scenes
        
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
                progress_callback(1, 1, 'Scene already loaded.')
            
            return True
        
        # Evict the oldest scene if the cache is full.
        if len(self._scene_cache) >= self.max_cached_scenes:
            self._evict_oldest_scene()

        print(f'\nLoading scene {self.get_scene_number(scene):04d}...')

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
            progress_callback(total_tasks, total_tasks, 'Scene loaded.')
        
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
            'max_cached_scenes': self.max_cached_scenes,
            'cached_scene_indices': list(self._scene_cache.keys()),
            'access_order': self._scene_access_order.copy()
        }
    
    def cleanup(self):
        '''Clean up resources.'''
        self._executor.shutdown(wait=False)
        self._scene_cache.clear()
        self._scene_access_order.clear()


class InteractiveVisualizer:
    '''
    Interactive Open3D visualizer with GUI controls that loads scenes on
    demand.
    
    Args:
        data_loader: VizDataLoader instance.
        title: window title.
        point_size: point cloud rendering size.
    '''
    def __init__(self, data_loader: VizDataLoader, title: str = 'SimBEV Interactive Viewer', point_size: float = 2.0):
        self._data_loader = data_loader
        self._point_size = point_size
        self._lidar_point_size = point_size
        self._semantic_lidar_point_size = point_size
        self._radar_point_size = point_size

        print('\n===== Keyboard Controls =====')
        print('+/=   : Increase point size')
        print('-/_   : Decrease point size')
        print('Space : Play/Pause animation')
        print('Left  : Previous frame')
        print('Right : Next frame')
        print('Up    : Previous sensor type')
        print('Down  : Next sensor type')
        print('E     : Bird\'s-eye view')
        print('T     : Tracker view')
        print('L     : Left view ')
        print('R     : Right view')
        print('F     : Front view')
        print('B     : Back view')
        print('X     : Toggle bounding boxes')
        print('=============================\n')
        
        self._app = gui.Application.instance
        self._app.initialize()
        
        self._window = self._app.create_window(title, 3840, 2160)
        
        # State
        self._current_scene = 0
        self._max_scene = data_loader.get_scene_count() - 1
        
        self._current_frame = 0
        self._max_frame = data_loader.get_frame_count(0) - 1

        self._play_speed = 20

        self._is_playing = False

        self._sensor_type = 'lidar'
        
        self._show_bbox = True

        self._bbox_count = 0

        # Loading state
        self._is_loading = False
        
        self._load_progress = 0
        self._load_total = 0
        
        # Playback
        self._last_frame_time = time.perf_counter()

        self._playback_loop = False
        
        # Create 3D scene widget.
        self._scene_widget = gui.SceneWidget()

        self._scene_widget.scene = o3d.visualization.rendering.Open3DScene(self._window.renderer)
        
        # Set up scene rendering.
        self._scene_widget.scene.set_background([0.0, 0.0, 0.0, 1.0])
        
        # Register keyboard callbacks.
        self._scene_widget.set_on_key(self._on_key_event)
        
        # Create control panel.
        self._create_control_panel()
        
        # Layout.
        self._window.set_on_layout(self._on_layout)
        self._window.add_child(self._scene_widget)
        self._window.add_child(self._panel)
        
        # Add coordinate frame.
        self._add_coordinate_frame()
        
        # Set up animation callback.
        self._window.set_on_tick_event(self._on_tick)
        
        # Load initial scene and display the first frame.
        self._load_and_display_scene(0)
    
    def _on_tick(self) -> bool:
        '''
        Animation callback for playback.

        Returns:
            True to continue the event loop, False to stop.
        '''
        if not self._is_playing:
            return True
        
        # Calculate if we should advance one frame based on playback speed.
        frame_time = 1.0 / self._play_speed
        
        elapsed = time.perf_counter() - self._last_frame_time
        
        if elapsed >= frame_time:
            self._last_frame_time = time.perf_counter()
            
            # Advance one frame.
            if self._current_frame < self._max_frame:
                self._current_frame += 1
                
                self._frame_slider.int_value = self._current_frame
                
                self._update_frame()
            else:
                self._loop_playback() if self._playback_loop else self._stop_playback()
        
        return True
    
    def _load_and_display_scene(self, scene: int):
        '''
        Load the scene and display the first frame.
        
        Args:
            scene: scene index (0-based).
        '''
        if self._data_loader.is_scene_loaded(scene):
            self._update_frame()
        else:
            self._is_loading = True
            
            self._update_loading_label('Loading scene...')
            
            def progress_callback(current, total, message):
                self._load_progress = current
                self._load_total = total
                
                gui.Application.instance.post_to_main_thread(
                    self._window,
                    lambda: self._update_loading_label(message)
                )
            
            # Load in the background.
            def load_worker():
                success = self._data_loader.load_scene(scene, progress_callback)
                
                # Update UI on the main thread.
                gui.Application.instance.post_to_main_thread(self._window, lambda: self._on_scene_loaded(success))
            
            threading.Thread(target=load_worker, daemon=True).start()
    
    def _on_scene_loaded(self, success: bool):
        '''
        Called when scene loading completes.
        
        Args:
            success: whether the scene was loaded successfully.
        '''
        self._is_loading = False
        
        if success:
            self._update_loading_label('Scene loaded.')
            self._update_cache_status()
            self._update_frame()
            
            # Setup the camera for the new scene.
            bounds = self._scene_widget.scene.bounding_box
            
            self._scene_widget.setup_camera(60, bounds, bounds.get_center())
        else:
            self._update_loading_label('Failed to load scene.')
    
    def _update_loading_label(self, message: str):
        '''
        Update loading status label.
        
        Args:
            message: status message.
        '''
        if self._is_loading:
            percent = (self._load_progress / self._load_total * 100) if self._load_total > 0 else 0
            
            self._loading_label.text = f'{message} ({percent:.0f}%)'
        else:
            self._loading_label.text = message
    
    def _on_key_event(self, event):
        '''
        Handle keyboard events.
        
        Args:
            event: key event.
        
        Returns:
            Callback result indicating if the event was handled.
        '''
        if event.type == gui.KeyEvent.DOWN:
            if event.key == ord('+') or event.key == ord('='):
                self._point_size = min(self._point_size + 1.0, 20.0)

                if self._sensor_type == 'lidar':
                    self._lidar_point_size = self._point_size
                elif self._sensor_type == 'semantic-lidar':
                    self._semantic_lidar_point_size = self._point_size
                elif self._sensor_type == 'radar':
                    self._radar_point_size = self._point_size
                
                self._update_point_size()
                
                return gui.Widget.EventCallbackResult.HANDLED
            
            elif event.key == ord('-') or event.key == ord('_'):
                self._point_size = max(self._point_size - 1.0, 1.0)

                if self._sensor_type == 'lidar':
                    self._lidar_point_size = self._point_size
                elif self._sensor_type == 'semantic-lidar':
                    self._semantic_lidar_point_size = self._point_size
                elif self._sensor_type == 'radar':
                    self._radar_point_size = self._point_size
                
                self._update_point_size()
                
                return gui.Widget.EventCallbackResult.HANDLED
            
            elif event.key == ord('X') or event.key == ord('x'):
                self._show_bbox = not self._show_bbox
                self._bbox_checkbox.checked = self._show_bbox
                
                self._update_frame()
                
                return gui.Widget.EventCallbackResult.HANDLED
            
            elif event.key == gui.KeyName.LEFT:
                self._on_prev_frame_clicked()
                
                return gui.Widget.EventCallbackResult.HANDLED
            
            elif event.key == gui.KeyName.RIGHT:
                self._on_next_frame_clicked()
                
                return gui.Widget.EventCallbackResult.HANDLED
            
            elif event.key == gui.KeyName.UP:
                self._sensor_radio.selected_index = (self._sensor_radio.selected_index - 1) % 3
                
                self._on_sensor_changed(self._sensor_radio.selected_index)
                
                return gui.Widget.EventCallbackResult.HANDLED
            
            elif event.key == gui.KeyName.DOWN:
                self._sensor_radio.selected_index = (self._sensor_radio.selected_index + 1) % 3
                
                self._on_sensor_changed(self._sensor_radio.selected_index)
                
                return gui.Widget.EventCallbackResult.HANDLED
            
            elif event.key == ord('E') or event.key == ord('e'):
                self._on_bev_view()
                
                return gui.Widget.EventCallbackResult.HANDLED
            
            elif event.key == ord('T') or event.key == ord('t'):
                self._on_tracker_view()
                
                return gui.Widget.EventCallbackResult.HANDLED
            
            elif event.key == ord('L') or event.key == ord('l'):
                self._on_left_view()
                
                return gui.Widget.EventCallbackResult.HANDLED
            
            elif event.key == ord('R') or event.key == ord('r'):
                self._on_right_view()
                
                return gui.Widget.EventCallbackResult.HANDLED
            
            elif event.key == ord('F') or event.key == ord('f'):
                self._on_front_view()
                
                return gui.Widget.EventCallbackResult.HANDLED
            
            elif event.key == ord('B') or event.key == ord('b'):
                self._on_back_view()
                
                return gui.Widget.EventCallbackResult.HANDLED
            
            elif event.key == gui.KeyName.SPACE:
                self._toggle_playback()
                
                return gui.Widget.EventCallbackResult.HANDLED
        
        return gui.Widget.EventCallbackResult.IGNORED
    
    def _update_point_size(self):
        '''Update point cloud rendering size.'''
        self._point_size_slider.double_value = self._point_size
        
        self._update_frame()
    
    def _add_coordinate_frame(self):
        '''Add coordinate frame to the scene.'''
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = 'defaultUnlit'
        
        self._scene_widget.scene.add_geometry('coordinate_frame', coord_frame, mat)
    
    def _create_control_panel(self):
        '''Create UI control panel.'''
        em = self._window.theme.font_size
        
        self._panel = gui.Vert(0, gui.Margins(em, em, em, em))
        
        # Sensor type selection.
        self._panel.add_child(gui.Label('Sensor Type:'))
        
        self._sensor_radio = gui.RadioButton(gui.RadioButton.VERT)
        self._sensor_radio.set_items(['Lidar', 'Semantic Lidar', 'Radar'])
        self._sensor_radio.selected_index = 0
        self._sensor_radio.set_on_selection_changed(self._on_sensor_changed)
        
        self._panel.add_child(self._sensor_radio)
        self._panel.add_fixed(2 * em)
        
        # Scene slider.
        self._scene_label = gui.Label(f'Scene: {self._data_loader.get_scene_number(0):04d} (1/{self._max_scene + 1})')
        
        self._panel.add_child(self._scene_label)
        
        self._scene_slider = gui.Slider(gui.Slider.INT)
        self._scene_slider.set_limits(0, self._max_scene)
        self._scene_slider.set_on_value_changed(self._on_scene_slider_changed)
        
        self._panel.add_child(self._scene_slider)
        self._panel.add_fixed(0.2 * em)
        
        # Scene navigation buttons.
        scene_button_layout = gui.Horiz()
        scene_button_layout.add_stretch()
        
        self._prev_scene_button = gui.Button('<')
        self._prev_scene_button.set_on_clicked(self._on_prev_scene_clicked)
        
        scene_button_layout.add_child(self._prev_scene_button)
        
        self._next_scene_button = gui.Button('>')
        self._next_scene_button.set_on_clicked(self._on_next_scene_clicked)
        
        scene_button_layout.add_child(self._next_scene_button)
        scene_button_layout.add_stretch()
        
        self._panel.add_child(scene_button_layout)
        self._panel.add_fixed(em)
        
        # Frame slider.
        self._frame_label = gui.Label(f'Frame: 1/{self._max_frame + 1}')
        
        self._panel.add_child(self._frame_label)
        
        self._frame_slider = gui.Slider(gui.Slider.INT)
        self._frame_slider.set_limits(0, self._max_frame)
        self._frame_slider.set_on_value_changed(self._on_frame_slider_changed)
        
        self._panel.add_child(self._frame_slider)
        self._panel.add_fixed(0.2 * em)
        
        # Frame navigation buttons.
        frame_button_layout = gui.Horiz()
        frame_button_layout.add_stretch()
        
        self._prev_frame_button = gui.Button('<')
        self._prev_frame_button.set_on_clicked(self._on_prev_frame_clicked)
        
        frame_button_layout.add_child(self._prev_frame_button)
        
        self._next_frame_button = gui.Button('>')
        self._next_frame_button.set_on_clicked(self._on_next_frame_clicked)
        
        frame_button_layout.add_child(self._next_frame_button)
        frame_button_layout.add_stretch()
        
        self._panel.add_child(frame_button_layout)
        self._panel.add_fixed(2 * em)
        
        # Playback controls.
        playback_layout = gui.Horiz()
        
        self._play_button = gui.Button('Play')
        self._play_button.background_color = gui.Color(0.1, 0.8, 0.1)
        self._play_button.set_on_clicked(self._on_play_clicked)
        
        playback_layout.add_child(self._play_button)
        playback_layout.add_fixed(em)

        # Playback loop checkbox.
        self._loop_checkbox = gui.Checkbox('Loop Playback')
        self._loop_checkbox.checked = False
        self._loop_checkbox.set_on_checked(self._on_loop_toggle)
        
        playback_layout.add_child(self._loop_checkbox)
        
        self._panel.add_child(playback_layout)
        self._panel.add_fixed(em)
        
        # Playback speed slider.
        self._panel.add_child(gui.Label('Playback Speed (FPS):'))
        
        self._speed_slider = gui.Slider(gui.Slider.INT)
        self._speed_slider.set_limits(1, 30)
        self._speed_slider.int_value = self._play_speed
        self._speed_slider.set_on_value_changed(self._on_speed_changed)
        
        self._panel.add_child(self._speed_slider)
        self._panel.add_fixed(2 * em)
        
        # Bounding box toggle.
        self._bbox_checkbox = gui.Checkbox('Show Bounding Boxes')
        self._bbox_checkbox.checked = True
        self._bbox_checkbox.set_on_checked(self._on_bbox_toggle)
        
        self._panel.add_child(self._bbox_checkbox)
        self._panel.add_fixed(2 * em)
        
        # Point size control.
        self._panel.add_child(gui.Label('Point Size:'))
        
        self._point_size_slider = gui.Slider(gui.Slider.DOUBLE)
        self._point_size_slider.set_limits(1.0, 20.0)
        self._point_size_slider.double_value = self._point_size
        self._point_size_slider.set_on_value_changed(self._on_point_size_slider_changed)
        
        self._panel.add_child(self._point_size_slider)
        self._panel.add_fixed(em)
        
        # Info label.
        self._info_label = gui.Label('')
        
        self._panel.add_child(self._info_label)
        self._panel.add_fixed(2 * em)
        
        # Loading status label
        self._loading_label = gui.Label('')
        
        self._panel.add_child(self._loading_label)
        self._panel.add_fixed(em)
        
        # Cache status label
        self._cache_status_label = gui.Label('')
        
        self._panel.add_child(self._cache_status_label)
        
        self._update_cache_status()
        
        self._panel.add_fixed(2 * em)
        
        # Camera view buttons
        self._panel.add_child(gui.Label('Camera View:'))
        
        camera_button_layout = gui.Horiz()
        camera_button_layout.add_stretch()
        
        view_button_layout = gui.Vert()
        
        self._bev_button = gui.Button('BEV')
        self._bev_button.set_on_clicked(self._on_bev_view)
        self._bev_button.horizontal_padding_em = 4.0
        
        view_button_layout.add_child(self._bev_button)
        view_button_layout.add_fixed(0.2 * em)
        
        self._tracker_button = gui.Button('Tracker')
        self._tracker_button.set_on_clicked(self._on_tracker_view)
        
        view_button_layout.add_child(self._tracker_button)
        view_button_layout.add_fixed(0.2 * em)

        self._left_button = gui.Button('Left')
        self._left_button.set_on_clicked(self._on_left_view)
        
        view_button_layout.add_child(self._left_button)
        view_button_layout.add_fixed(0.2 * em)

        self._right_button = gui.Button('Right')
        self._right_button.set_on_clicked(self._on_right_view)
        
        view_button_layout.add_child(self._right_button)
        view_button_layout.add_fixed(0.2 * em)

        self._front_button = gui.Button('Front')
        self._front_button.set_on_clicked(self._on_front_view)
        
        view_button_layout.add_child(self._front_button)
        view_button_layout.add_fixed(0.2 * em)

        self._back_button = gui.Button('Back')
        self._back_button.set_on_clicked(self._on_back_view)
        
        view_button_layout.add_child(self._back_button)
        
        camera_button_layout.add_child(view_button_layout)
        camera_button_layout.add_stretch()
        
        self._panel.add_child(camera_button_layout)
    
    def _update_cache_status(self):
        '''Update cache status label.'''
        cache_info = self._data_loader.get_cache_info()
        
        cached = cache_info['cached_scenes']
        max_cached = cache_info['max_cached_scenes']
        
        # Show which scenes are cached.
        if cache_info['cached_scene_indices']:
            scene_numbers = [self._data_loader.get_scene_number(idx) for idx in cache_info['cached_scene_indices']]
            
            scene_str = ', '.join([f"{num:04d}" for num in scene_numbers])
            
            self._cache_status_label.text = (f'Cache: {cached}/{max_cached} scenes\nLoaded: {scene_str}')
        else:
            self._cache_status_label.text = f'Cache: {cached}/{max_cached} scenes'
    
    def _on_layout(self, layout_context):
        '''Handle window layout.'''
        r = self._window.content_rect
        
        panel_width = 640
        
        self._scene_widget.frame = gui.Rect(r.x, r.y, r.width - panel_width, r.height)
        
        self._panel.frame = gui.Rect(r.get_right() - panel_width, r.y, panel_width, r.height)
    
    def _on_sensor_changed(self, index):
        '''
        Handle sensor type radio button change.
        
        Args:
            index: selected index.
        '''
        sensor_map = {0: 'lidar', 1: 'semantic-lidar', 2: 'radar'}
        
        self._sensor_type = sensor_map[index]
        
        # Update point size based on sensor.
        if self._sensor_type == 'lidar':
            self._point_size = self._lidar_point_size
        elif self._sensor_type == 'semantic-lidar':
            self._point_size = self._semantic_lidar_point_size
        elif self._sensor_type == 'radar':
            self._point_size = self._radar_point_size
        
        self._point_size_slider.double_value = self._point_size
        
        self._update_frame()
    
    def _on_point_size_slider_changed(self, value):
        '''
        Handle point size slider change.
        
        Args:
            value: point size value.
        '''
        self._point_size = value
        
        if self._sensor_type == 'lidar':
            self._lidar_point_size = value
        elif self._sensor_type == 'semantic-lidar':
            self._semantic_lidar_point_size = value
        elif self._sensor_type == 'radar':
            self._radar_point_size = value
        
        self._update_point_size()
    
    def _on_speed_changed(self, value):
        '''
        Handle playback speed slider change.
        
        Args:
            value: playback speed value.
        '''
        self._play_speed = int(value)
    
    def _on_play_clicked(self):
        '''Handle play button click.'''
        self._toggle_playback()
    
    def _toggle_playback(self):
        '''Toggle playback state.'''
        self._is_playing = not self._is_playing
        
        if self._is_playing:
            self._play_button.text = 'Pause'
            self._play_button.background_color = gui.Color(0.8, 0.1, 0.1)
            
            self._last_frame_time = time.perf_counter()
        else:
            self._play_button.text = 'Play'
            self._play_button.background_color = gui.Color(0.1, 0.8, 0.1)
    
    def _stop_playback(self):
        '''Stop playback.'''
        self._is_playing = False
        
        self._play_button.text = 'Play'
        self._play_button.background_color = gui.Color(0.1, 0.8, 0.1)
    
    def _loop_playback(self):
        '''Loop playback from start to end.'''
        self._current_frame = 0
        
        self._frame_slider.int_value = 0
    
    def _on_scene_slider_changed(self, value):
        '''
        Handle scene slider value change.
        
        Args:
            value: scene index value.
        '''
        new_scene = int(value)
        
        if new_scene != self._current_scene:
            # Stop playback before changing scenes.
            if self._is_playing:
                self._is_playing = False
                
                self._play_button.text = 'Play'
                self._play_button.background_color = gui.Color(0.1, 0.8, 0.1)
            
            self._current_scene = new_scene
            self._current_frame = 0
            
            # Update frame slider limits.
            self._max_frame = self._data_loader.get_frame_count(self._current_scene) - 1
            
            self._frame_slider.set_limits(0, self._max_frame)
            self._frame_slider.int_value = 0
            
            # Load and display the new scene.
            self._load_and_display_scene(self._current_scene)
    
    def _on_frame_slider_changed(self, value):
        '''
        Handle frame slider value change.
        
        Args:
            value: frame index value.
        '''
        # Stop playback when manually scrubbing.
        if self._is_playing:
            self._is_playing = False
            
            self._play_button.text = 'Play'
            self._play_button.background_color = gui.Color(0.1, 0.8, 0.1)
        
        self._current_frame = int(value)
        
        self._update_frame()
    
    def _on_bbox_toggle(self, checked):
        '''
        Handle bounding box checkbox toggle.
        
        Args:
            checked: whether bounding box display is enabled.
        '''
        self._show_bbox = checked

        self._update_frame()
    
    def _on_loop_toggle(self, checked):
        '''
        Handle loop playback checkbox toggle.
        
        Args:
            checked: whether loop playback is enabled.
        '''
        self._playback_loop = checked
    
    def _on_prev_scene_clicked(self):
        '''Navigate to the previous scene.'''
        if self._current_scene > 0:
            self._on_scene_slider_changed(self._current_scene - 1)
            
            self._scene_slider.int_value = self._current_scene
            
    def _on_next_scene_clicked(self):
        '''Navigate to the next scene.'''
        if self._current_scene < self._max_scene:
            self._on_scene_slider_changed(self._current_scene + 1)
            
            self._scene_slider.int_value = self._current_scene
            
    def _on_prev_frame_clicked(self):
        '''Navigate to the previous frame.'''
        if self._current_frame > 0:
            self._current_frame -= 1
            
            self._frame_slider.int_value = self._current_frame
            
            self._update_frame()
    
    def _on_next_frame_clicked(self):
        '''Navigate to the next frame.'''
        if self._current_frame < self._max_frame:
            self._current_frame += 1
            
            self._frame_slider.int_value = self._current_frame
            
            self._update_frame()
    
    def _on_bev_view(self):
        '''Set camera to bird's-eye view.'''
        bounds = self._scene_widget.scene.bounding_box
        
        center = bounds.get_center()
        extent = bounds.get_extent()
        
        max_extent = max(extent[0], extent[1])
        
        self._scene_widget.look_at(center, [center[0], center[1], max_extent], [0, 1, 0])

    def _on_tracker_view(self):
        '''Set camera to ego tracker view.'''
        distance = 8.0
        
        self._scene_widget.look_at([0, 0, 0], [-2.0 * distance, 0, 0.5 * distance], [0, 0, 1])
    
    def _on_left_view(self):
        '''Set camera to left side view.'''
        self._scene_widget.look_at([0, 0, 0], [0, -2.0, 0], [0, 0, 1])
    
    def _on_right_view(self):
        '''Set camera to right side view.'''
        self._scene_widget.look_at([0, 0, 0], [0, 2.0, 0], [0, 0, 1])
    
    def _on_front_view(self):
        '''Set camera to front view.'''
        self._scene_widget.look_at([0, 0, 0], [-2.0, 0, 0], [0, 0, 1])
    
    def _on_back_view(self):
        '''Set camera to back view.'''
        self._scene_widget.look_at([0, 0, 0], [2.0, 0, 0], [0, 0, 1])
    
    def _update_frame(self):
        '''Update visualization to the current scene and frame.'''
        if self._is_loading:
            return
        
        try:
            sensor_data = self._data_loader.get_frame(self._current_scene, self._current_frame, self._sensor_type)
        
        except Exception as e:
            print(f'Unexpected error: {e}')
            
            return
        
        # Validate the data.
        if sensor_data is None or 'points' not in sensor_data:
            return
        
        try:
            # Update the point cloud.
            pcd = o3d.geometry.PointCloud()
            
            pcd.points = o3d.utility.Vector3dVector(sensor_data['points'])
            
            if 'colors' in sensor_data and sensor_data['colors'] is not None:
                pcd.colors = o3d.utility.Vector3dVector(sensor_data['colors'])
            else:
                colors = np.tile([0.8, 0.8, 0.8], (len(sensor_data['points']), 1))
                
                pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # Set point cloud material.
            mat = o3d.visualization.rendering.MaterialRecord()
            
            mat.shader = 'defaultUnlit'
            mat.point_size = self._point_size
            
            # Remove old point cloud.
            if self._scene_widget.scene.has_geometry('point_cloud'):
                self._scene_widget.scene.remove_geometry('point_cloud')
            
            self._scene_widget.scene.add_geometry('point_cloud', pcd, mat)
            
            # Update bounding boxes.
            self._update_bboxes(sensor_data.get('bboxes', []))
            
        except Exception as e:
            print(f'Error updating geometry: {e}')
            
            return
        
        # Update labels.
        scene_number = self._data_loader.get_scene_number(self._current_scene)
        
        self._scene_label.text = f'Scene: {scene_number:04d} ({self._current_scene + 1}/{self._max_scene + 1})'
        self._frame_label.text = f'Frame: {self._current_frame + 1}/{self._max_frame + 1}'
        self._info_label.text = (
            f'Points: {len(sensor_data["points"])}\n'
            f'Bounding Boxes: {len(sensor_data.get("bboxes", []))}'
        )
        
        # Update cache status.
        self._update_cache_status()
    
    def _update_bboxes(self, bboxes):
        '''
        Update bounding box visualization.
        
        Args:
            bboxes: list of bounding boxes.
        '''
        if not hasattr(self, '_bbox_count'):
            self._bbox_count = 0
        
        # Remove old bounding boxes.
        for i in range(self._bbox_count):
            bbox_name = f'bbox_{i}'
            
            try:
                if self._scene_widget.scene.has_geometry(bbox_name):
                    self._scene_widget.scene.remove_geometry(bbox_name)
            
            except:
                pass
        
        self._bbox_count = 0
        
        if not self._show_bbox or not bboxes:
            return
        
        # Add new bounding boxes.
        for idx, bbox in enumerate(bboxes):
            try:
                line_set = self._create_bbox_lineset(bbox['corners'], bbox.get('label', 'car'))
                
                mat = o3d.visualization.rendering.MaterialRecord()
                
                mat.shader = 'unlitLine'
                mat.line_width = 2.0
                
                self._scene_widget.scene.add_geometry(f'bbox_{idx}', line_set, mat)
                
                self._bbox_count += 1
                
            except Exception as e:
                print(f'Error adding bounding box {idx}: {e}')
                
                continue
    
    def _create_bbox_lineset(self, corners, label):
        '''Create Open3D LineSet for bounding box.
        
        Args:
            corners: corners of the bounding box.
            label: label of the bounding box.
        
        Returns:
            Open3D LineSet representing the bounding box.
        '''
        lines = [
            [0, 1], [1, 3], [3, 2], [2, 0], # front face
            [4, 5], [5, 7], [7, 6], [6, 4], # back face
            [0, 4], [1, 5], [2, 6], [3, 7], # sides
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
        
        # Clean up on exit.
        self._data_loader.cleanup()


def visualize_interactive(ctx):
    '''
    Unified interactive visualizer for all sensors with point cloud data.
    
    Args:
        ctx: visualization context.
    '''
    metadata = None
    
    for split in ['train', 'val', 'test']:
        info_path = f'{ctx.path}/simbev/infos/simbev_infos_{split}.json'
        
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                infos = json.load(f)
            
            metadata = infos['metadata']
            
            break
    
    if metadata is None:
        print('Error: Could not load metadata.')
        
        return
    
    # Create data loader with cache.
    print('Initializing data loader...')
    
    data_loader = VizDataLoader(
        ctx.path, 
        metadata, 
        ignore_valid_flag=ctx.ignore_valid_flag,
        max_workers=16,
        max_cached_scenes=5
    )
    
    scene_count = data_loader.get_scene_count()
    
    if scene_count == 0:
        print('No scenes found to visualize.')
        
        return
    
    print(f'Found {scene_count} scene(s).')
    
    # Create and run the visualizer.
    visualizer = InteractiveVisualizer(data_loader=data_loader, title='SimBEV Interactive Viewer', point_size=2.0)
    
    visualizer.run()