import mmcv
import torch

import numpy as np

from .pipelines import Compose

from pytorch3d.ops import box3d_overlap

from mmdet.datasets import DATASETS
from torch.utils.data import Dataset

from pyquaternion import Quaternion as Q

from ..core.bbox import LiDARInstance3DBoxes, get_box_type


CAM_NAME = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
            'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

OBJECT_CLASSES = {
    12: 'pedestrian',
    14: 'car',
    15: 'truck',
    16: 'bus',
    18: 'motorcycle',
    19: 'bicycle'
}


@DATASETS.register_module()
class SimBEVDataset(Dataset):
    '''
    This class serves as the API for experiments on the SimBEV dataset.

    Args:
        dataset_root: root directory of the dataset.
        ann_file: annotation file of the dataset.
        object_classes: list of object classes in the dataset.
        map_classes: list of BEV map classes in the dataset.
        pipeline: pipeline used for data processing.
        modality: modality of the input data.
        test_mode: whether the dataset is used for training or testing.
        filter_empty_gt: whether to filter out samples with empty ground
            truth.
        with_velocity: whether to include velocity information in the object
            detection ground truth and predictions.
        use_valid_flag: whether to filter out invalid objects from each
            sample.
        load_interval: interval for loading data samples.
        box_type_3d: type of 3D box used in the dataset, indicating the
            coordinate system of the 3D box. Can be 'LiDAR', 'Depth', or
            'Camera'.
        det_eval_mode: evaluation mode for 3D object detection results, can be
            'iou' or 'distance'.
    '''

    def __init__(
        self,
        dataset_root,
        ann_file,
        object_classes=None,
        map_classes=None,
        pipeline=None,
        modality=None,
        test_mode=False,
        filter_empty_gt=True,
        with_velocity=True,
        use_valid_flag=False,
        load_interval=20,
        box_type_3d='LiDAR',
        det_eval_mode='iou'
    ):
        super().__init__()
        self.dataset_root = dataset_root
        self.ann_file = ann_file
        self.object_classes = object_classes
        self.map_classes = map_classes
        self.modality = modality
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.with_velocity = with_velocity
        self.use_valid_flag = use_valid_flag
        self.load_interval = load_interval

        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)

        self.eval_mode = det_eval_mode
        
        self.epoch = -1

        # Get the list of object classes in the dataset.
        self.CLASSES = self.get_classes(object_classes)

        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}

        # Load annotations from the annotation file.
        self.data_infos = self.load_annotations(self.ann_file)

        # Create the data processing pipeline.
        if pipeline is not None:
            self.pipeline = Compose(pipeline)

        if self.modality is None:
            self.modality = dict(use_camera=True, use_lidar=True)

        if not self.test_mode:
            self._set_group_flag()

    def set_epoch(self, epoch):
        '''
        Set the epoch for transforms that require epoch information along the
        pipeline.

        Args:
            epoch: epoch to set.
        '''
        self.epoch = epoch
        
        if hasattr(self, 'pipeline'):
            for transform in self.pipeline.transforms:
                if hasattr(transform, 'set_epoch'):
                    transform.set_epoch(epoch)
    
    @classmethod
    def get_classes(cls, classes=None):
        '''
        Get the list of object class names in the dataset.

        Args:
            cls: list of dataset classes.
            classes: path to the file containing the list of classes, or the
                list of classes itself.
        
        Returns:
            class_names: list of object class names in the dataset.
        '''
        if classes is None:
            return cls.CLASSES

        if isinstance(classes, str):
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        return class_names
    
    def get_cat_ids(self, index):
        '''
        Get category IDs of objects in the sample.

        Args:
            index: index of the sample in the dataset.

        Returns:
            cat_ids: list of category IDs of objects in the sample.
        '''
        info = self.data_infos[index]

        if self.use_valid_flag:
            mask = info['valid_flag']
            gt_names = set(info['gt_names'][mask])
        else:
            gt_names = set(info['gt_names'])

        cat_ids = []

        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        
        return cat_ids
    
    def load_annotations(self, ann_file):
        '''
        Load annotations from the annotation file.

        Args:
            ann_file: annotation file of the dataset.

        Returns:
            data_infos: list of data samples in the dataset.
        '''
        annotations = mmcv.load(ann_file)

        data_infos = []

        for key in annotations['data']:
            data_infos += annotations['data'][key]['scene_data']
        
        data_infos = data_infos[::self.load_interval]

        self.metadata = annotations['metadata']

        data_infos = self.load_gt_bboxes(data_infos)

        return data_infos
    
    def load_gt_bboxes(self, infos):
        '''
        Load ground truth bounding boxes from file into the list of data
        samples.

        Args:
            infos: list of data samples in the dataset.
        
        Returns:
            infos: list of data samples updated with ground truth bounding
                boxes.
        '''
        for info in infos:
            gt_boxes = []
            gt_names = []
            gt_velocities = []
            
            num_lidar_pts = []
            num_radar_pts = []
            
            valid_flag = []

            # Load ground truth bounding boxes from file.
            gt_det_path = info['GT_DET']

            mmcv.check_file_exist(gt_det_path)

            gt_det = np.load(gt_det_path, allow_pickle=True)

            # Ego to global transformation.
            ego2global = np.eye(4).astype(np.float32)
            
            ego2global[:3, :3] = Q(info['ego2global_rotation']).rotation_matrix
            ego2global[:3, 3] = info['ego2global_translation']

            # Lidar to ego transformation.
            lidar2ego = np.eye(4).astype(np.float32)
            
            lidar2ego[:3, :3] = Q(self.metadata['LIDAR']['sensor2ego_rotation']).rotation_matrix
            lidar2ego[:3, 3] = self.metadata['LIDAR']['sensor2ego_translation']

            global2lidar = np.linalg.inv(ego2global @ lidar2ego)

            # Transform bounding boxes from the global coordinate system to
            # the lidar coordinate system.
            for det_object in gt_det:
                for tag in det_object['semantic_tags']:
                    if tag in OBJECT_CLASSES.keys():
                        global_bbox_corners = np.append(det_object['bounding_box'], np.ones((8, 1)), 1)
                        bbox_corners = (global2lidar @ global_bbox_corners.T)[:3].T

                        # Calculate the center of the bounding box.
                        center = ((bbox_corners[0] + bbox_corners[7]) / 2).tolist()

                        # Calculate the dimensions of the bounding box.
                        center.append(np.linalg.norm(bbox_corners[0] - bbox_corners[2]))
                        center.append(np.linalg.norm(bbox_corners[0] - bbox_corners[4]))
                        center.append(np.linalg.norm(bbox_corners[0] - bbox_corners[1]))

                        # Calculate the yaw angle of the bounding box.
                        diff = bbox_corners[0] - bbox_corners[2]
                        
                        gamma = np.arctan2(diff[1], diff[0])

                        center.append(-gamma)

                        gt_boxes.append(center)
                        gt_names.append(OBJECT_CLASSES[tag])
                        gt_velocities.append(det_object['linear_velocity'][:2])
                        
                        num_lidar_pts.append(det_object['num_lidar_pts'])
                        num_radar_pts.append(det_object['num_radar_pts'])
                        
                        valid_flag.append(det_object['valid_flag'])

            info['gt_boxes'] = np.array(gt_boxes)
            info['gt_names'] = np.array(gt_names)
            info['gt_velocity'] = np.array(gt_velocities)

            info['num_lidar_pts'] = np.array(num_lidar_pts)
            info['num_radar_pts'] = np.array(num_radar_pts)
            
            info['valid_flag'] = np.array(valid_flag)

        return infos
    
    def get_data_info(self, index):
        '''
        Package information from a data sample.

        Args:
            index: index of the sample in the dataset.
        
        Returns:
            data: packaged information from the sample.
        '''
        info = self.data_infos[index]

        data = dict(
            scene = info['scene'],
            frame = info['frame'],
            timestamp = info['timestamp'],
            gt_seg_path = info['GT_SEG'],
            gt_det_path = info['GT_DET'],
            lidar_path = info['LIDAR']
        )

        # Ego to global transformation.
        ego2global = np.eye(4).astype(np.float32)
        
        ego2global[:3, :3] = Q(info['ego2global_rotation']).rotation_matrix
        ego2global[:3, 3] = info['ego2global_translation']
        
        data['ego2global'] = ego2global

        # Lidar to ego transformation.
        lidar2ego = np.eye(4).astype(np.float32)
        
        lidar2ego[:3, :3] = Q(self.metadata['LIDAR']['sensor2ego_rotation']).rotation_matrix
        lidar2ego[:3, 3] = self.metadata['LIDAR']['sensor2ego_translation']
        
        data['lidar2ego'] = lidar2ego

        if self.modality['use_camera']:
            data['image_paths'] = []
            data['camera_intrinsics'] = []
            data['camera2lidar'] = []
            data['lidar2camera'] = []
            data['lidar2image'] = []
            data['camera2ego'] = []

            for camera in CAM_NAME:
                data['image_paths'].append(info['RGB-' + camera])

                # Camera intrinsics.
                camera_intrinsics = np.eye(4).astype(np.float32)

                camera_intrinsics[:3, :3] = self.metadata['camera_intrinsics']
                
                data['camera_intrinsics'].append(camera_intrinsics)
                
                # Lidar to camera transformation.
                camera2lidar = np.eye(4).astype(np.float32)

                camera2lidar[:3, :3] = Q(self.metadata[camera]['sensor2lidar_rotation']).rotation_matrix
                camera2lidar[:3, 3] = self.metadata[camera]['sensor2lidar_translation']

                data['camera2lidar'].append(camera2lidar)

                lidar2camera = np.linalg.inv(camera2lidar)
                
                data['lidar2camera'].append(lidar2camera)

                # Lidar to image transformation.
                lidar2image = camera_intrinsics @ lidar2camera

                data['lidar2image'].append(lidar2image)

                # Camera to ego transformation.
                camera2ego = np.eye(4).astype(np.float32)

                camera2ego[:3, :3] = Q(self.metadata[camera]['sensor2ego_rotation']).rotation_matrix
                camera2ego[:3, 3] = self.metadata[camera]['sensor2ego_translation']

                data['camera2ego'].append(camera2ego)

        data['ann_info'] = self.get_ann_info(index)
        
        return data
    
    def get_ann_info(self, index):
        '''
        Get annotation information for a data sample.

        Args:
            index: index of the sample in the dataset.
        
        Returns:
            anns_results: annotation information from the sample.
        '''
        info = self.data_infos[index]

        if self.use_valid_flag:
            mask = info['valid_flag']
        else:
            mask = info['num_lidar_pts'] > 0
        
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]

        gt_labels_3d = []

        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = info['gt_velocity'][mask]

            nan_mask = np.isnan(gt_velocity[:, 0])
            
            gt_velocity[nan_mask] = [0.0, 0.0]
            
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0)
        ).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
        )

        return anns_results
    
    def pre_pipeline(self, results):
        '''
        Prepare data for the pipeline.

        Args:
            results: data to be prepared for the pipeline.
        '''
        results['img_fields'] = []
        results['bbox3d_fields'] = []
        results['pts_mask_fields'] = []
        results['pts_seg_fields'] = []
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []

        results['box_type_3d'] = self.box_type_3d
        results['box_mode_3d'] = self.box_mode_3d
    
    def prepare_train_data(self, index):
        '''
        Prepare data for training.

        Args:
            index: index of the sample in the dataset.
        
        Returns:
            example: data prepared for training.
        '''
        input_dict = self.get_data_info(index)

        if input_dict is None:
            return None
        
        self.pre_pipeline(input_dict)
        
        example = self.pipeline(input_dict)

        if self.filter_empty_gt and (example is None or ~(example['gt_labels_3d']._data != -1).any()):
            return None

        return example
    
    def prepare_test_data(self, index):
        '''
        Prepare data for testing.

        Args:
            index: index of the sample in the dataset.
        
        Returns:
            example: data prepared for testing.
        '''
        input_dict = self.get_data_info(index)
        
        self.pre_pipeline(input_dict)
        
        example = self.pipeline(input_dict)

        return example
    
    def evaluate_map(self, results):
        '''
        Evaluate BEV map segmentation results.

        Args:
            results: BEV map segmentation results from the model.
        
        Returns:
            metrics: evaluation metrics for BEV map segmentation results.
        '''
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        thresholds = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]).to(device)

        num_classes = len(self.map_classes)
        num_thresholds = len(thresholds)

        tp = torch.zeros(num_classes, num_thresholds).to(device)
        fp = torch.zeros(num_classes, num_thresholds).to(device)
        fn = torch.zeros(num_classes, num_thresholds).to(device)

        for result in results:
            pred = result['masks_bev'].to(device)
            label = result['gt_masks_bev'].to(device)

            pred = pred.detach().reshape(num_classes, -1)
            label = label.detach().bool().reshape(num_classes, -1)

            pred = pred[:, :, None] >= thresholds
            label = label[:, :, None]

            tp += (pred & label).sum(dim=1)
            fp += (pred & ~label).sum(dim=1)
            fn += (~pred & label).sum(dim=1)

        ious = tp / (tp + fp + fn + 1e-6)
        
        metrics = {}
        
        for index, name in enumerate(self.map_classes):
            metrics[f'map/{name}/IoU@max'] = ious[index].max().item()
            
            for threshold, iou in zip(thresholds, ious[index]):
                metrics[f'map/{name}/IoU@{threshold.item():.2f}'] = iou.item()
        
        metrics['map/mean/IoU@max'] = ious.max(dim=1).values.mean().item()

        for index, threshold in enumerate(thresholds):
            metrics[f'map/mean/IoU@{threshold.item():.2f}'] = ious[:, index].mean().item()
        
        # Print IoU table.
        print('\n\n')

        print(f'{"IoU":<12} {0.1:<8}{0.2:<8}{0.3:<8}{0.4:<8}{0.5:<8}{0.6:<8}{0.7:<8}{0.8:<8}{0.9:<8}')

        for index, name in enumerate(self.map_classes):
            print(f'{name:<12}', ''.join([f'{iou:<8.4f}' for iou in ious[index].tolist()]))
        
        print(f'{"mIoU":<12}', ''.join([f'{iou:<8.4f}' for iou in ious.mean(dim=0).tolist()]), '\n')
        
        return metrics
    
    def evaluate(self, results, **kwargs):
        '''
        Evaluate model results.

        Args:
            results: list of results from the model.
        
        Returns:
            metrics: evaluation metrics for the results.
        '''
        metrics = {}

        # Evaluate BEV map segmentation results.
        if 'masks_bev' in results[0]:
            metrics.update(self.evaluate_map(results))

        # Evaluate 3D object detection results.
        if 'boxes_3d' in results[0]:
            simbev_eval = SimBEVDetectionEval(results, self.object_classes, self.eval_mode)

            metrics.update(simbev_eval.evaluate())
        
        return metrics
    
    def _set_group_flag(self):
        '''
        Set the flag for the dataset.
        '''
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def _rand_another(self, index):
        '''
        Get another random data sample from the same group.

        Args:
            index: index of the sample in the dataset.
        
        Returns:
            sample: index of another sample from the same group.
        '''
        pool = np.where(self.flag == self.flag[index])[0]
        
        return np.random.choice(pool)
    
    def __getitem__(self, index):
        if self.test_mode:
            return self.prepare_test_data(index)

        while True:
            data = self.prepare_train_data(index)

            if data is None:
                index = self._rand_another(index)
                continue
            
            return data
    
    def __len__(self):
        return len(self.data_infos)


class SimBEVDetectionEval:
    '''
    Class for evaluating 3D object detection results on the SimBEV dataset.

    Args:
        results: results from the model.
        classes: list of object classes in the dataset.
        mode: evalution mode, can be 'iou' or 'distance'.
    '''
    def __init__(self, results, classes, mode='iou'):
        self.results = results
        self.classes = classes
        self.mode = mode

        iou_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        distance_thresholds = [0.5, 1.0, 2.0, 4.0]

        if self.mode == 'iou':
            self.thresholds = iou_thresholds
        elif self.mode == 'distance':
            self.thresholds = distance_thresholds
        else:
            raise ValueError(f'Unsupported evaluation mode {self.mode}.')

    def evaluate(self):
        '''
        Evaluate 3D object detection results.
        '''
        num_classes = len(self.classes)
        num_thresholds = len(self.thresholds)

        # Dictionary to store Average Precision (AP), Average Translation
        # Error (ATE), Average Orientation Error (AOE), Average Scale Error
        # (ASE), and Average Velocity Error (AVE) for each class and IoU
        # threshold.
        det_metrics = {
            item: torch.zeros((num_classes, num_thresholds)) for item in ['AP', 'ATE', 'AOE', 'ASE', 'AVE']
        }

        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        print('\n')
        
        for k, threshold in enumerate(self.thresholds):
            print(f'Calculating metrics for threshold {threshold}...')

            # Dictionaries to store True Positive (TP) and False Positive (FP)
            # values, scores, ATE, AOE, ASE, AVE, and the total number of
            # ground truth boxes for each class.
            tps = {i: torch.empty((0, )) for i in range(num_classes)}
            fps = {i: torch.empty((0, )) for i in range(num_classes)}

            scores = {i: torch.empty((0, )) for i in range(num_classes)}

            ate = {i: torch.empty((0, )) for i in range(num_classes)}
            aoe = {i: torch.empty((0, )) for i in range(num_classes)}
            ase = {i: torch.empty((0, )) for i in range(num_classes)}
            ave = {i: torch.empty((0, )) for i in range(num_classes)}

            num_gt_boxes = {i: 0 for i in range(num_classes)}

            # Iterate over predictions for each sample.
            for result in self.results:
                boxes_3d = result['boxes_3d']
                scores_3d = result['scores_3d']
                labels_3d = result['labels_3d']
                gt_boxes_3d = result['gt_bboxes_3d']
                gt_labels_3d = result['gt_labels_3d']

                if self.mode == 'iou':
                    if len(boxes_3d.tensor) > 0:
                        boxes_3d_corners = boxes_3d.corners
                    else:
                        boxes_3d_corners = torch.empty((0, 8, 3))

                    if len(gt_boxes_3d.tensor) > 0:
                        gt_boxes_3d_corners = gt_boxes_3d.corners
                    else:
                        gt_boxes_3d_corners = torch.empty((0, 8, 3))
                else:
                    boxes_3d_centers = boxes_3d.gravity_center

                    gt_boxes_3d_centers = gt_boxes_3d.gravity_center

                for cls in range(num_classes):
                    pred_mask = labels_3d == cls
                    
                    gt_mask = gt_labels_3d == cls

                    pred_boxes = boxes_3d[pred_mask]
                    
                    if self.mode == 'iou':
                        pred_box_corners = boxes_3d_corners[pred_mask]
                    else:
                        pred_box_centers = boxes_3d_centers[pred_mask]
                    
                    pred_scores = scores_3d[pred_mask]
                    
                    gt_boxes = gt_boxes_3d[gt_mask]

                    if self.mode == 'iou':
                        gt_box_corners = gt_boxes_3d_corners[gt_mask]
                    else:
                        gt_box_centers = gt_boxes_3d_centers[gt_mask]

                    # Sort predictions by confidence score in descending
                    # order.
                    sorted_indices = torch.argsort(-pred_scores)

                    pred_boxes = pred_boxes[sorted_indices]

                    if self.mode == 'iou':
                        pred_box_corners = pred_box_corners[sorted_indices]
                    else:
                        pred_box_centers = pred_box_centers[sorted_indices]
                    
                    pred_scores = pred_scores[sorted_indices]

                    if self.mode == 'iou':
                        pred_box_corners = pred_box_corners.to(device)
                        gt_box_corners = gt_box_corners.to(device)
                    else:
                        pred_box_centers = pred_box_centers.to(device)
                        gt_box_centers = gt_box_centers.to(device)
                    
                    if self.mode == 'iou':
                        # Calculate Intersection over Union (IoU) between
                        # predicted and ground truth bounding boxes.
                        if len(pred_box_corners) == 0:
                            ious = torch.zeros((0, len(gt_box_corners))).to(device)
                        elif len(gt_box_corners) == 0:
                            ious = torch.zeros((len(pred_box_corners), 0)).to(device)
                        else:
                            _, ious = box3d_overlap(pred_box_corners, gt_box_corners)
                    else:
                        # Calculate Euclidean distance between predicted and
                        # ground truth bounding box centers.
                        dists = torch.cdist(pred_box_centers, gt_box_centers)

                    # Tensor to keep track of ground truth boxes that have
                    # been assigned to a prediction.
                    assigned_gt = torch.zeros(len(gt_boxes), dtype=torch.bool).to(device)

                    tp = torch.zeros(len(pred_boxes))
                    fp = torch.zeros(len(pred_boxes))                  

                    ate_local = []
                    aoe_local = []
                    ase_local = []
                    ave_local = []

                    for i, pred_box in enumerate(pred_boxes):                        
                        matched = False
                        matched_gt_idx = -1
                        
                        if self.mode == 'iou':
                            # Among the ground truth bounding boxes that have not
                            # been matched to a prediction yet, find the one with
                            # the highest IoU value.
                            available_ious = ious[i] * ~assigned_gt

                            if available_ious.shape[0] > 0:
                                iou_max, max_gt_idx = available_ious.max(dim=0)
                                max_gt_idx = max_gt_idx.item()
                            else:
                                iou_max = 0
                                max_gt_idx = -1

                            if iou_max >= threshold:
                                matched = True
                                matched_gt_idx = max_gt_idx
                        else:
                            # Among the ground truth bounding boxes that have not
                            # been matched to a prediction yet, find the one with
                            # the smallest Euclidean distance.
                            available_dists = 10000 - ((10000 - dists[i]) * ~assigned_gt)    

                            if available_dists.shape[0] > 0:
                                dist_min, min_gt_idx = available_dists.min(dim=0)
                                min_gt_idx = min_gt_idx.item()
                            else:
                                dist_min = 10000
                                min_gt_idx = -1
                            
                            if dist_min <= threshold:
                                matched = True
                                matched_gt_idx = min_gt_idx
                        
                        if matched:
                            tp[i] = 1

                            assigned_gt[matched_gt_idx] = True

                            # Calculate ATE, which is the Euclidean distance
                            # between the predicted and ground truth bounding
                            # box centers.
                            ate_local.append(
                                torch.linalg.vector_norm(
                                    pred_boxes[i].tensor[0, :3] - gt_boxes[matched_gt_idx].tensor[0, :3]
                                )
                            )

                            # Calculate AOE, which is the smallest yaw angle
                            # between the predicted and ground truth bounding
                            # boxes.
                            diff_angle = (
                                gt_boxes[matched_gt_idx].tensor[0, 6] - pred_boxes[i].tensor[0, 6] + np.pi
                            ) % (2 * np.pi) - np.pi

                            # Ensure the angle difference is between -pi and
                            # pi.
                            if diff_angle > np.pi:
                                diff_angle = diff_angle - 2 * np.pi

                            aoe_local.append(abs(diff_angle))

                            # Calculate ASE, which is defined as 1 - IOU after
                            # the predicted and ground truth bounding boxes
                            # are translated and rotated to have the same
                            # center and orientation.
                            pred_wlh = pred_boxes[i].tensor[0, 3:6]
                            gt_wlh = gt_boxes[matched_gt_idx].tensor[0, 3:6]

                            min_wlh = torch.minimum(pred_wlh, gt_wlh)

                            pred_vol = torch.prod(pred_wlh)
                            gt_vol = torch.prod(gt_wlh)
                            
                            intersection = torch.prod(min_wlh)

                            union = pred_vol + gt_vol - intersection

                            ase_local.append(1 - intersection / union)

                            # Calculate AVE, which is the L2 norm of the
                            # difference between the predicted and ground
                            # truth bounding box velocities.
                            ave_local.append(
                                torch.linalg.vector_norm(
                                    pred_boxes[i].tensor[0, -2:] - gt_boxes[matched_gt_idx].tensor[0, -2:]
                                )
                            )
                        else:
                            fp[i] = 1
                    
                    tps[cls] = torch.cat((tps[cls], tp))
                    fps[cls] = torch.cat((fps[cls], fp))

                    scores[cls] = torch.cat((scores[cls], pred_scores))

                    ate[cls] = torch.cat((ate[cls], torch.Tensor(ate_local)))
                    aoe[cls] = torch.cat((aoe[cls], torch.Tensor(aoe_local)))
                    ase[cls] = torch.cat((ase[cls], torch.Tensor(ase_local)))
                    ave[cls] = torch.cat((ave[cls], torch.Tensor(ave_local)))

                    num_gt_boxes[cls] += len(gt_boxes)

            for cls in range(num_classes):
                # Sort TP and FP values by confidence score in descending
                # order.
                sorted_indices = torch.argsort(-scores[cls])

                tps[cls] = tps[cls][sorted_indices]
                fps[cls] = fps[cls][sorted_indices]

                tps[cls] = torch.cumsum(tps[cls], dim=0).to(torch.float32)
                fps[cls] = torch.cumsum(fps[cls], dim=0).to(torch.float32)

                recalls = tps[cls] / num_gt_boxes[cls]
                precisions = tps[cls] / (tps[cls] + fps[cls])

                # Add the (0, 1) point to the precision-recall curve.
                recalls = torch.cat((torch.Tensor([0.0]), recalls))
                precisions = torch.cat((torch.Tensor([1.0]), precisions))

                # AP is the area under the precision-recall curve.
                det_metrics['AP'][cls, k] = torch.trapz(precisions, recalls)

                for item, value in zip(['ATE', 'AOE', 'ASE', 'AVE'], [ate, aoe, ase, ave]):
                    det_metrics[item][cls, k] = value[cls].mean()

        metrics = {}

        mean_metrics = {}

        print('\n')

        for item in ['AP', 'ATE', 'AOE', 'ASE', 'AVE']:
            for index, name in enumerate(self.classes):
                metrics[f'det/{name}/{item}@max'] = det_metrics[item][index].max().item()
                metrics[f'det/{name}/{item}@mean'] = det_metrics[item][index].nanmean().item()

                for threshold, value in zip(self.thresholds, det_metrics[item][index]):
                    metrics[f'det/{name}/{item}@{threshold:.2f}'] = value.item()
        
            for index, threshold in enumerate(self.thresholds):
                metrics[f'det/mean/{item}@{threshold:.2f}'] = det_metrics[item][:, index].nanmean().item()
            
            if self.mode == 'iou':
                print(f'{item:<12} {0.1:<8}{0.2:<8}{0.3:<8}{0.4:<8}{0.5:<8}{0.6:<8}{0.7:<8}{0.8:<8}{0.9:<8} {"mean":<8}')
            else:
                print(f'{item:<12} {0.5:<8}{1.0:<8}{2.0:<8}{4.0:<8} {"mean":<8}')

            for index, name in enumerate(self.classes):
                print(
                    f'{name:<12}',
                    ''.join([f'{value:<8.4f}' for value in det_metrics[item][index].tolist()]),
                    f'{det_metrics[item][index].nanmean().item():<8.4f}'
                )
            
            print(
                f'm{item:<11}',
                ''.join([f'{value:<8.4f}' for value in det_metrics[item].nanmean(dim=0).tolist()]),
                '\n'
            )

            if self.mode == 'iou':
                mean_metrics[f'm{item}'] = det_metrics[item][:, 2:].nanmean().item()
            else:
                mean_metrics[f'm{item}'] = det_metrics[item].nanmean().item()

            metrics[f'det/m{item}'] = mean_metrics[f'm{item}']

            print(f'm{item}: ', mean_metrics[f'm{item}'], '\n')

        mATE = max(0.0, 1 - mean_metrics['mATE'])
        mAOE = max(0.0, 1 - mean_metrics['mAOE'])
        mASE = max(0.0, 1 - mean_metrics['mASE'])
        mAVE = max(0.0, 1 - mean_metrics['mAVE'])

        SimBEVDetectionScore = (4 * mean_metrics['mAP'] + mATE + mAOE + mASE + mAVE) / 8

        metrics['det/SDS'] = SimBEVDetectionScore

        print('SDS: ', SimBEVDetectionScore, '\n')
        
        return metrics