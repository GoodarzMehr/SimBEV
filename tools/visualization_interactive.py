# Academic Software License: Copyright © 2025 Goodarz Mehr.

import os
import json
import open3d as o3d
import numpy as np
from pyquaternion import Quaternion as Q
import open3d.visualization.gui as gui

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


class AdvancedVisualizer:
    '''
    Advanced interactive visualizer with GUI slider and controls.
    
    Args:
        title: window title.
        point_size: point cloud rendering size.
    '''
    def __init__(self, title='SimBEV Advanced Viewer', point_size=2.0):
        self.app = gui.Application.instance
        self.app.initialize()
        
        self.window = self.app.create_window(title, 1920, 1080)
        
        # State
        self.current_frame = 0
        self.max_frame = 0
        self.show_bbox = True
        self.frames_data = []
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
    
    def _create_control_panel(self):
        '''Create UI control panel with slider and checkbox.'''
        em = self.window.theme.font_size
        
        self.panel = gui.Vert(0, gui.Margins(em, em, em, em))
        
        # Frame slider
        self.frame_label = gui.Label("Frame: 0/0")
        self.panel.add_child(self.frame_label)
        
        self.frame_slider = gui.Slider(gui.Slider.INT)
        self.frame_slider.set_limits(0, 100)
        self.frame_slider.set_on_value_changed(self._on_slider_changed)
        self.panel.add_child(self.frame_slider)
        
        self.panel.add_fixed(em)  # Spacing
        
        # Bounding box toggle
        self.bbox_checkbox = gui.Checkbox("Show Bounding Boxes")
        self.bbox_checkbox.checked = True
        self.bbox_checkbox.set_on_checked(self._on_bbox_toggle)
        self.panel.add_child(self.bbox_checkbox)
        
        self.panel.add_fixed(em)
        
        # Info label
        self.info_label = gui.Label("")
        self.panel.add_child(self.info_label)
        
        # Navigation buttons
        button_layout = gui.Horiz()
        
        self.prev_button = gui.Button("Previous")
        self.prev_button.set_on_clicked(self._on_prev_clicked)
        button_layout.add_child(self.prev_button)
        
        self.next_button = gui.Button("Next")
        self.next_button.set_on_clicked(self._on_next_clicked)
        button_layout.add_child(self.next_button)
        
        self.panel.add_child(button_layout)
        
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
    
    def _on_slider_changed(self, value):
        '''Handle slider value change.'''
        self.current_frame = int(value)
        self._update_frame()
    
    def _on_bbox_toggle(self, checked):
        '''Handle bbox checkbox toggle.'''
        self.show_bbox = checked
        self._update_frame()
    
    def _on_prev_clicked(self):
        '''Navigate to previous frame.'''
        if self.current_frame > 0:
            self.current_frame -= 1
            self.frame_slider.int_value = self.current_frame
            self._update_frame()
    
    def _on_next_clicked(self):
        '''Navigate to next frame.'''
        if self.current_frame < self.max_frame:
            self.current_frame += 1
            self.frame_slider.int_value = self.current_frame
            self._update_frame()
    
    def _on_reset_view(self):
        '''Reset camera view.'''
        bounds = self.scene_widget.scene.bounding_box
        self.scene_widget.setup_camera(60, bounds, bounds.get_center())
    
    def set_frames(self, frames_data):
        '''
        Set frame data for visualization.
        
        Args:
            frames_data: list of dicts with keys:
                - 'points': (N, 3) array
                - 'colors': (N, 3) array (optional)
                - 'bboxes': list of bbox dicts (optional)
        '''
        self.frames_data = frames_data
        self.max_frame = len(frames_data) - 1
        self.current_frame = 0
        
        # Update slider range
        self.frame_slider.set_limits(0, self.max_frame)
        self.frame_slider.int_value = 0
        
        # Add coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=3.0, origin=[0, 0, 0]
        )
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        self.scene_widget.scene.add_geometry("coordinate_frame", coord_frame, mat)
        
        # Update to first frame
        self._update_frame()
        
        # Setup camera
        bounds = self.scene_widget.scene.bounding_box
        self.scene_widget.setup_camera(60, bounds, bounds.get_center())
    
    def _update_frame(self):
        '''Update visualization to current frame.'''
        if not self.frames_data:
            return
        
        frame_data = self.frames_data[self.current_frame]
        
        # Update point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(frame_data['points'])
        
        if 'colors' in frame_data and frame_data['colors'] is not None:
            pcd.colors = o3d.utility.Vector3dVector(frame_data['colors'])
        else:
            colors = np.tile([0.5, 0.5, 0.5], (len(frame_data['points']), 1))
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
        self._update_bboxes(frame_data.get('bboxes', []))
        
        # Update labels
        self.frame_label.text = f"Frame: {self.current_frame + 1}/{self.max_frame + 1}"
        self.info_label.text = (
            f"Points: {len(frame_data['points'])}\n"
            f"BBoxes: {len(frame_data.get('bboxes', []))}"
        )
    
    def _update_bboxes(self, bboxes):
        '''Update bounding box visualization.'''
        # Remove old bounding boxes
        for i in range(100):  # Remove up to 100 old bboxes
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


def visualize_lidar_live(ctx):
    '''Interactive LiDAR visualization with temporal navigation.'''
    # Collect all frames for this scene
    frames_data = []
    
    # Get scene data
    scene_key = f'scene_{ctx.scene_number:04d}'
    
    # Load info file to get all frames in scene
    for split in ['train', 'val', 'test']:
        info_path = f'{ctx.path}/simbev/infos/simbev_infos_{split}.json'
        
        if not os.path.exists(info_path):
            continue
        
        with open(info_path, 'r') as f:
            infos = json.load(f)
        
        if scene_key in infos['data']:
            scene_data = infos['data'][scene_key]['scene_data']
            metadata = infos['metadata']
            
            # Load each frame
            for frame_idx, frame_data in enumerate(scene_data):
                point_cloud = np.load(frame_data['LIDAR'])['data']
                
                # Compute colors based on distance
                distances = np.linalg.norm(point_cloud, axis=1)
                log_distances = np.log(distances + 1e-6)
                log_normalized = (log_distances - log_distances.min()) / \
                    (log_distances.max() - log_distances.min() + 1e-6)
                
                # Use rainbow colormap
                from matplotlib import colormaps as cm
                RANGE = np.linspace(0.0, 1.0, 256)
                RAINBOW = np.array(cm.get_cmap('rainbow')(RANGE))[:, :3]
                
                colors = np.c_[
                    np.interp(log_normalized, RANGE, RAINBOW[:, 0]),
                    np.interp(log_normalized, RANGE, RAINBOW[:, 1]),
                    np.interp(log_normalized, RANGE, RAINBOW[:, 2])
                ]
                
                # Load bounding boxes
                gt_det = np.load(frame_data['GT_DET'], allow_pickle=True)
                
                from tools.visualization_utils import get_global2sensor, transform_bbox
                
                global2lidar = get_global2sensor(frame_data, metadata, 'LIDAR')
                corners, labels = transform_bbox(gt_det, global2lidar, ctx.ignore_valid_flag)
                
                # Convert to bbox format for Open3D
                bboxes = []
                for corner, label in zip(corners, labels):
                    bboxes.append({
                        'corners': corner,
                        'label': label
                    })
                
                frames_data.append({
                    'points': point_cloud,
                    'colors': colors,
                    'bboxes': bboxes
                })
            
            # Visualize
            visualizer = AdvancedVisualizer(
                title=f'SimBEV LiDAR - Scene {ctx.scene_number:04d}',
                point_size=2.0
            )
            visualizer.set_frames(frames_data)
            visualizer.run()
            
            break


def visualize_semantic_lidar_live(ctx):
    '''Interactive semantic LiDAR visualization.'''
    # Label colors (from LABEL_COLORS in visualization_handlers.py)
    LABEL_COLORS = np.array([
        (255, 255, 255), (128, 64, 128), (244, 35, 232), (70, 70, 70),
        (102, 102, 156), (190, 153, 153), (153, 153, 153), (250, 170, 30),
        (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180),
        (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
        (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32),
        (110, 190, 160), (170, 120, 50), (55, 90, 80), (45, 60, 150),
        (227, 227, 227), (81, 0, 81), (150, 100, 100), (230, 150, 140),
        (180, 165, 180), (180, 130, 70)
    ]) / 255.0
    
    frames_data = []
    scene_key = f'scene_{ctx.scene_number:04d}'
    
    for split in ['train', 'val', 'test']:
        info_path = f'{ctx.path}/simbev/infos/simbev_infos_{split}.json'
        
        if not os.path.exists(info_path):
            continue
        
        with open(info_path, 'r') as f:
            infos = json.load(f)
        
        if scene_key in infos['data']:
            scene_data = infos['data'][scene_key]['scene_data']
            metadata = infos['metadata']
            
            for frame_idx, frame_data in enumerate(scene_data):
                data = np.load(frame_data['SEG-LIDAR'])['data']
                point_cloud = np.array([data['x'], data['y'], data['z']]).T
                labels = np.array(data['ObjTag'])
                colors = LABEL_COLORS[labels]
                
                # Load bounding boxes
                gt_det = np.load(frame_data['GT_DET'], allow_pickle=True)
                
                from tools.visualization_utils import get_global2sensor, transform_bbox
                
                global2lidar = get_global2sensor(frame_data, metadata, 'LIDAR')
                corners, bbox_labels = transform_bbox(gt_det, global2lidar, ctx.ignore_valid_flag)
                
                bboxes = []
                for corner, label in zip(corners, bbox_labels):
                    bboxes.append({'corners': corner, 'label': label})
                
                frames_data.append({
                    'points': point_cloud,
                    'colors': colors,
                    'bboxes': bboxes
                })
            
            visualizer = AdvancedVisualizer(
                title=f'SimBEV Semantic LiDAR - Scene {ctx.scene_number:04d}',
                point_size=3.0
            )
            visualizer.set_frames(frames_data)
            visualizer.run()
            
            break


def visualize_radar_live(ctx):
    '''Interactive radar visualization.'''
    frames_data = []
    scene_key = f'scene_{ctx.scene_number:04d}'
    
    RAD_NAME = ['RAD_LEFT', 'RAD_FRONT', 'RAD_RIGHT', 'RAD_BACK']
    
    for split in ['train', 'val', 'test']:
        info_path = f'{ctx.path}/simbev/infos/simbev_infos_{split}.json'
        
        if not os.path.exists(info_path):
            continue
        
        with open(info_path, 'r') as f:
            infos = json.load(f)
        
        if scene_key in infos['data']:
            scene_data = infos['data'][scene_key]['scene_data']
            metadata = infos['metadata']
            
            for frame_idx, frame_data in enumerate(scene_data):
                point_cloud = []
                velocity = []
                
                for radar in RAD_NAME:
                    radar2lidar = np.eye(4, dtype=np.float32)
                    radar2lidar[:3, :3] = Q(metadata[radar]['sensor2lidar_rotation']).rotation_matrix
                    radar2lidar[:3, 3] = metadata[radar]['sensor2lidar_translation']
                    
                    radar_points = np.load(frame_data[radar])['data']
                    velocity.append(radar_points[:, -1])
                    radar_points = radar_points[:, :-1]
                    
                    x = radar_points[:, 0] * np.cos(radar_points[:, 1]) * np.cos(radar_points[:, 2])
                    y = radar_points[:, 0] * np.cos(radar_points[:, 1]) * np.sin(radar_points[:, 2])
                    z = radar_points[:, 0] * np.sin(radar_points[:, 1])
                    
                    points = np.stack((x, y, z), axis=1)
                    points_transformed = (radar2lidar @ np.append(points, np.ones((points.shape[0], 1)), 1).T)[:3].T
                    
                    point_cloud.append(points_transformed)
                
                point_cloud = np.concatenate(point_cloud, axis=0)
                velocity = np.concatenate(velocity, axis=0)
                
                # Velocity-based colors
                log_velocity = np.log(1.0 + np.abs(velocity))
                log_velocity_normalized = (log_velocity - log_velocity.min()) / \
                    (log_velocity.max() - log_velocity.min() + 1e-6)
                
                from matplotlib import colormaps as cm
                RANGE = np.linspace(0.0, 1.0, 256)
                RAINBOW = np.array(cm.get_cmap('rainbow')(RANGE))[:, :3]
                
                colors = np.c_[
                    np.interp(log_velocity_normalized, RANGE, RAINBOW[:, 0]),
                    np.interp(log_velocity_normalized, RANGE, RAINBOW[:, 1]),
                    np.interp(log_velocity_normalized, RANGE, RAINBOW[:, 2])
                ]
                
                # Load bounding boxes
                gt_det = np.load(frame_data['GT_DET'], allow_pickle=True)
                
                from tools.visualization_utils import get_global2sensor, transform_bbox
                
                global2lidar = get_global2sensor(frame_data, metadata, 'LIDAR')
                corners, bbox_labels = transform_bbox(gt_det, global2lidar, ctx.ignore_valid_flag)
                
                bboxes = []
                for corner, label in zip(corners, bbox_labels):
                    bboxes.append({'corners': corner, 'label': label})
                
                frames_data.append({
                    'points': point_cloud,
                    'colors': colors,
                    'bboxes': bboxes
                })
            
            visualizer = AdvancedVisualizer(
                title=f'SimBEV Radar - Scene {ctx.scene_number:04d}',
                point_size=5.0
            )
            visualizer.set_frames(frames_data)
            visualizer.run()
            
            break