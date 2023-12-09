import mmcv
import torch

import numpy as np

from .pipelines import Compose

from mmdet.datasets import DATASETS
from torch.utils.data import Dataset

from pyquaternion import Quaternion as Q


CAM_NAME = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
            'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']


@DATASETS.register_module()
class CarlaDataset(Dataset):
    '''
    This class serves as the API for experiments on a CARLA dataset generated
    using SimBEV.

    Attributes:
        - dataset_root: root directory of the dataset.
        - ann_file: annotation file of the dataset.
        - classes: dataset map classes.
        - modality: modality of the dataset.
        - test_mode: whether the dataset is used for training or testing.
        - load_interval: interval of data samples.
        - CLASSES: dataset map class names.
        - data_infos: dataset annotation information.
        - pipeline: data augmentation pipeline.
        - epoch: current epoch.
        - metadata: dataset metadata (sensor coordinate transformations).
        - flag: flag for data grouping.

    Methods:
        - set_epoch: set current epoch.
        - get_classes: get dataset map class names.
        - get_cat_ids: get class IDs present in a data sample.
        - load_annotations: load dataset annotations.
        - get_data_info: generate meta information for a data sample.
        - prepare_train_data: process a data sample through the augmentation
            pipeline for training.
        - prepare_test_data: process a data sample through the augmentation
            pipeline for testing.
        - evaluate_map: calculate intersection over union (IoU) for each class
            using different thresholds.
        - evaluate: evaluate model performance.
        - _set_group_flag: set flag for data grouping.
        - _rand_another: get another data sample with the same flag.
    '''

    def __init__(
        self,
        dataset_root,
        ann_file,
        classes=None,
        pipeline=None,
        modality=None,
        test_mode=False,
        load_interval=1,
    ):
        super().__init__()
        self.dataset_root = dataset_root
        self.ann_file = ann_file
        self.classes = classes
        self.modality = modality
        self.test_mode = test_mode
        self.load_interval = load_interval

        self.CLASSES = self.get_classes(classes)

        self.data_infos = self.load_annotations(self.ann_file)

        if pipeline is not None:
            self.pipeline = Compose(pipeline)

        if self.modality is None:
            self.modality = dict(use_camera=True, use_lidar=True)

        if not self.test_mode:
            self._set_group_flag()
        
        self.epoch = -1

    def set_epoch(self, epoch):
        self.epoch = epoch
        
        if hasattr(self, 'pipeline'):
            for transform in self.pipeline.transforms:
                if hasattr(transform, 'set_epoch'):
                    transform.set_epoch(epoch)
    
    @classmethod
    def get_classes(cls, classes=None):
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
        info = self.data_infos[index]
        gt_path = info['ground_truth']

        mmcv.check_file_exist(gt_path)
        
        gt_masks = np.load(gt_path)
        categories = gt_masks.any(axis=(1, 2))

        return np.where(categories==True)[0].tolist()
    
    def load_annotations(self, ann_file):
        annotations = mmcv.load(ann_file)

        data_infos = annotations['data']
        data_infos = data_infos[:: self.load_interval]

        self.metadata = annotations['metadata']

        return data_infos
    
    def get_data_info(self, index):
        info = self.data_infos[index]

        data = dict(
            location = self.metadata['location'],
            timestamp = info['timestamp'],
            gt_path = info['ground_truth'],
            lidar_path = info['LIDAR_TOP']
        )

        # Ego to global transformation.
        ego2global = np.eye(4).astype(np.float32)
        
        ego2global[:3, :3] = Q(info['ego2global_rotation']).rotation_matrix
        ego2global[:3, 3] = info['ego2global_translation']
        
        data['ego2global'] = ego2global

        # Lidar to ego transformation.
        lidar2ego = np.eye(4).astype(np.float32)
        
        lidar2ego[:3, :3] = Q(self.metadata['LIDAR_TOP']['sensor2ego_rotation']).rotation_matrix
        lidar2ego[:3, 3] = self.metadata['LIDAR_TOP']['sensor2ego_translation']
        
        data['lidar2ego'] = lidar2ego

        if self.modality['use_camera']:
            data['image_paths'] = []
            data['camera_intrinsics'] = []
            data['lidar2camera'] = []
            data['lidar2image'] = []
            data['camera2ego'] = []
            data['camera2lidar'] = []

            for camera in CAM_NAME:
                data['image_paths'].append(info[camera])

                # Camera intrinsics.
                camera_intrinsics = np.eye(4).astype(np.float32)

                camera_intrinsics[:3, :3] = self.metadata['camera_intrinsics']
                
                data['camera_intrinsics'].append(camera_intrinsics)
                
                # Lidar to camera transformation.
                lidar2camera_r = np.linalg.inv(Q(self.metadata[camera]['sensor2lidar_rotation']).rotation_matrix)
                lidar2camera_t = self.metadata[camera]['sensor2lidar_translation'] @ lidar2camera_r.T
                
                lidar2camera_rt = np.eye(4).astype(np.float32)
                
                lidar2camera_rt[:3, :3] = lidar2camera_r.T
                lidar2camera_rt[3, :3] = -lidar2camera_t
                
                data['lidar2camera'].append(lidar2camera_rt.T)

                # Lidar to image transformation,
                lidar2image = camera_intrinsics @ lidar2camera_rt.T

                data['lidar2image'].append(lidar2image)

                # Camera to ego transformation,
                camera2ego = np.eye(4).astype(np.float32)

                camera2ego[:3, :3] = Q(self.metadata[camera]['sensor2ego_rotation']).rotation_matrix
                camera2ego[:3, 3] = self.metadata[camera]['sensor2ego_translation']

                data['camera2ego'].append(camera2ego)

                # Camera to lidar transformation.
                camera2lidar = np.eye(4).astype(np.float32)

                camera2lidar[:3, :3] = Q(self.metadata[camera]['sensor2lidar_rotation']).rotation_matrix
                camera2lidar[:3, 3] = self.metadata[camera]['sensor2lidar_translation']

                data['camera2lidar'].append(camera2lidar)

        return data
    
    def prepare_train_data(self, index):
        input_dict = self.get_data_info(index)

        if input_dict is None:
            return None
        
        example = self.pipeline(input_dict)

        return example
    
    def prepare_test_data(self, index):
        input_dict = self.get_data_info(index)
        
        example = self.pipeline(input_dict)

        return example
    
    def evaluate_map(self, results):
        thresholds = torch.tensor([0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7])

        num_classes = len(self.classes)
        num_thresholds = len(thresholds)

        tp = torch.zeros(num_classes, num_thresholds)
        fp = torch.zeros(num_classes, num_thresholds)
        fn = torch.zeros(num_classes, num_thresholds)

        for result in results:
            pred = result['masks_bev']
            label = result['gt_masks_bev']

            pred = pred.detach().reshape(num_classes, -1)
            label = label.detach().bool().reshape(num_classes, -1)

            pred = pred[:, :, None] >= thresholds
            label = label[:, :, None]

            tp += (pred & label).sum(dim=1)
            fp += (pred & ~label).sum(dim=1)
            fn += (~pred & label).sum(dim=1)

        ious = tp / (tp + fp + fn + 1e-7)

        metrics = {}
        
        for index, name in enumerate(self.classes):
            metrics[f'map/{name}/iou@max'] = ious[index].max().item()
            
            for threshold, iou in zip(thresholds, ious[index]):
                metrics[f'map/{name}/iou@{threshold.item():.2f}'] = iou.item()
        
        metrics['map/mean/iou@max'] = ious.max(dim=1).values.mean().item()
        
        return metrics
    
    def evaluate(self, results, **kwargs):
        metrics = {}

        metrics.update(self.evaluate_map(results))

        return metrics
    
    def _set_group_flag(self):
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def _rand_another(self, index):
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