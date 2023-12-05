# Copyright (c) Vision & AI Lab. All rights reserved

import mmcv
import torch
import numpy as np
import pyquaternion
import tempfile
from nuscenes.utils.data_classes import Box as NuScenesBox
from os import path as osp

from mmdet.datasets import DATASETS
from ..core import show_result
from ..core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes
from .nuscenes_dataset import NuScenesDataset
from .pipelines import Compose
from nuscenes.nuscenes import NuScenes
import os 
from PIL import Image

@DATASETS.register_module()
class NuScenesMapDataset(NuScenesDataset):
    """
    NuScenesDataset with map modality 
    """
    NameMapping = {
        'movable_object.barrier': 'barrier',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.car': 'car',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.motorcycle': 'motorcycle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'movable_object.trafficcone': 'traffic_cone',
        'vehicle.trailer': 'trailer',
        'vehicle.truck': 'truck'
    }
    DefaultAttribute = {
        'car': 'vehicle.parked',
        'pedestrian': 'pedestrian.moving',
        'trailer': 'vehicle.parked',
        'truck': 'vehicle.parked',
        'bus': 'vehicle.moving',
        'motorcycle': 'cycle.without_rider',
        'construction_vehicle': 'vehicle.parked',
        'bicycle': 'cycle.without_rider',
        'barrier': '',
        'traffic_cone': '',
    }
    AttrMapping = {
        'cycle.with_rider': 0,
        'cycle.without_rider': 1,
        'pedestrian.moving': 2,
        'pedestrian.standing': 3,
        'pedestrian.sitting_lying_down': 4,
        'vehicle.moving': 5,
        'vehicle.parked': 6,
        'vehicle.stopped': 7,
    }
    AttrMapping_rev = [
        'cycle.with_rider',
        'cycle.without_rider',
        'pedestrian.moving',
        'pedestrian.standing',
        'pedestrian.sitting_lying_down',
        'vehicle.moving',
        'vehicle.parked',
        'vehicle.stopped',
    ]
    # https://github.com/nutonomy/nuscenes-devkit/blob/57889ff20678577025326cfc24e57424a829be0a/python-sdk/nuscenes/eval/detection/evaluate.py#L222 # noqa
    ErrNameMapping = {
        'trans_err': 'mATE',
        'scale_err': 'mASE',
        'orient_err': 'mAOE',
        'vel_err': 'mAVE',
        'attr_err': 'mAAE'
    }
    CLASSES = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
               'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
               'barrier')
        
    def _format_bbox(self, results, jsonfile_prefix=None):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}
        mapped_class_names = self.CLASSES

        print('Start to convert detection format...')
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            boxes = output_to_nusc_box(det, self.data_infos[sample_id],
                                       self.speed_mode, self.img_info_prototype,
                                       self.max_interval, self.test_adj,
                                       self.fix_direction,
                                       self.test_adj_ids)
            sample_token = self.data_infos[sample_id]['token']
            boxes = lidar_nusc_box_to_global(self.data_infos[sample_id], boxes,
                                             mapped_class_names,
                                             self.eval_detection_configs,
                                             self.eval_version)
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                if np.sqrt(box.velocity[0]**2 + box.velocity[1]**2) > 0.2:
                    if name in [
                            'car',
                            'construction_vehicle',
                            'bus',
                            'truck',
                            'trailer',
                    ]:
                        attr = 'vehicle.moving'
                    elif name in ['bicycle', 'motorcycle']:
                        attr = 'cycle.with_rider'
                    else:
                        attr = NuScenesDataset.DefaultAttribute[name]
                else:
                    if name in ['pedestrian']:
                        attr = 'pedestrian.standing'
                    elif name in ['bus']:
                        attr = 'vehicle.stopped'
                    else:
                        attr = NuScenesDataset.DefaultAttribute[name]

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                    detection_name=name,
                    detection_score=box.score,
                    attribute_name=attr)
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print('Results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            timestamp=info['timestamp'] / 1e6,
        )
    
        if self.modality['use_camera']:
            if self.img_info_prototype == 'mmcv':
                image_paths = []
                lidar2img_rts = []
                for cam_type, cam_info in info['cams'].items():
                    image_paths.append(cam_info['data_path'])
                    # obtain lidar to image transformation matrix
                    lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                    lidar2cam_t = cam_info[
                        'sensor2lidar_translation'] @ lidar2cam_r.T
                    lidar2cam_rt = np.eye(4)
                    lidar2cam_rt[:3, :3] = lidar2cam_r.T
                    lidar2cam_rt[3, :3] = -lidar2cam_t
                    intrinsic = cam_info['cam_intrinsic']
                    viewpad = np.eye(4)
                    viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                    lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                    lidar2img_rts.append(lidar2img_rt)
                    

                input_dict.update(
                    dict(
                        img_filename=image_paths,
                        lidar2img=lidar2img_rts,
                    ))
            elif self.img_info_prototype == 'bevdet':
                input_dict.update(dict(img_info=info['cams']))
                proj_paths = {} 
                depth_paths = {}
              
                fbev = info['token']
                for cam_type, cam_info in info['cams'].items():
                    
                    # test data
                    fname = cam_info['sample_data_token']
                    proj = os.path.join(self.data_root, 'projmap', cam_type, f'{fname}.jpg')
                    depth = os.path.join(self.data_root, 'projdepth', cam_type, f'{fname}.npy')

                    proj_paths[cam_type] = proj 
                    depth_paths[cam_type] = depth

                bev_path = os.path.join(self.data_root, 'bevmap', f'{fbev}.png')
                input_dict.update(dict(
                    proj = proj_paths,
                    depth = depth_paths, 
                    bev = bev_path
                    ))
            
            elif self.img_info_prototype == 'bevdet_sequential':
                if info ['prev'] is None or info['next'] is None:
                    adjacent= 'prev' if info['next'] is None else 'next'
                else:
                    if self.prev_only or self.next_only:
                        adjacent = 'prev' if self.prev_only else 'next'
                    elif self.test_mode:
                        adjacent = self.test_adj
                    else:
                        adjacent = np.random.choice(['prev', 'next'])
                if type(info[adjacent]) is list:
                    if self.test_mode:
                        if self.test_adj_ids is not None:
                            info_adj=[]
                            select_id = self.test_adj_ids
                            for id_tmp in select_id:
                                id_tmp = min(id_tmp, len(info[adjacent])-1)
                                info_adj.append(info[adjacent][id_tmp])
                        else:
                            select_id = min((self.max_interval+self.min_interval)//2,
                                            len(info[adjacent])-1)
                            info_adj = info[adjacent][select_id]
                    else:
                        if len(info[adjacent])<= self.min_interval:
                            select_id = len(info[adjacent])-1
                        else:
                            select_id = np.random.choice([adj_id for adj_id in range(
                                min(self.min_interval,len(info[adjacent])),
                                min(self.max_interval,len(info[adjacent])))])
                        info_adj = info[adjacent][select_id]
                else:
                    info_adj = info[adjacent]
                input_dict.update(dict(img_info=info['cams'],
                                       curr=info,
                                       adjacent=info_adj,
                                       adjacent_type=adjacent))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
            if self.img_info_prototype == 'bevdet_sequential':
                bbox = input_dict['ann_info']['gt_bboxes_3d'].tensor
                if 'abs' in self.speed_mode:
                    bbox[:, 7:9] = bbox[:, 7:9] + torch.from_numpy(info['velo']).view(1,2).to(bbox)
                if input_dict['adjacent_type'] == 'next' and not self.fix_direction:
                    bbox[:, 7:9] = -bbox[:, 7:9]
                if 'dis' in self.speed_mode:
                    time = abs(input_dict['timestamp'] - 1e-6 * input_dict['adjacent']['timestamp'])
                    bbox[:, 7:9] = bbox[:, 7:9] * time
                input_dict['ann_info']['gt_bboxes_3d'] = LiDARInstance3DBoxes(bbox,
                                                                              box_dim=bbox.shape[-1],
                                                                              origin=(0.5, 0.5, 0.0))
        
        return input_dict

def output_to_nusc_box(detection, info, speed_mode,
                       img_info_prototype, max_interval, test_adj, fix_direction,
                       test_adj_ids):
    """Convert the output to the box class in the nuScenes.

    Args:
        detection (dict): Detection results.

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.

    Returns:
        list[:obj:`NuScenesBox`]: List of standard NuScenesBoxes.
    """
    box3d = detection['boxes_3d']
    scores = detection['scores_3d'].numpy()
    labels = detection['labels_3d'].numpy()

    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()
    # TODO: check whether this is necessary
    # with dir_offset & dir_limit in the head
    box_yaw = -box_yaw - np.pi / 2

    velocity_all = box3d.tensor[:, 7:9]
    if img_info_prototype =='bevdet_sequential':
        if info['prev'] is None or info['next'] is None:
            adjacent = 'prev' if info['next'] is None else 'next'
        else:
            adjacent = test_adj
        if adjacent == 'next' and not fix_direction:
            velocity_all = -velocity_all
        if type(info[adjacent]) is list:
            select_id = min(max_interval // 2, len(info[adjacent]) - 1)
            # select_id = min(2, len(info[adjacent]) - 1)
            info_adj = info[adjacent][select_id]
        else:
            info_adj = info[adjacent]
        if 'dis' in speed_mode and test_adj_ids is None:
            time = abs(1e-6 * info['timestamp'] - 1e-6 * info_adj['timestamp'])
            velocity_all = velocity_all / time

    box_list = []
    for i in range(len(box3d)):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        velocity = (*velocity_all[i,:], 0.0)
        # velo_val = np.linalg.norm(box3d[i, 7:9])
        # velo_ori = box3d[i, 6]
        # velocity = (
        # velo_val * np.cos(velo_ori), velo_val * np.sin(velo_ori), 0.0)
        box = NuScenesBox(
            box_gravity_center[i],
            box_dims[i],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity)
        box_list.append(box)
    return box_list


def lidar_nusc_box_to_global(info,
                             boxes,
                             classes,
                             eval_configs,
                             eval_version='detection_cvpr_2019'):
    """Convert the box from ego to global coordinate.

    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        classes (list[str]): Mapped classes in the evaluation.
        eval_configs (object): Evaluation configuration object.
        eval_version (str): Evaluation version.
            Default: 'detection_cvpr_2019'

    Returns:
        list: List of standard NuScenesBoxes in the global
            coordinate.
    """
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.rotate(pyquaternion.Quaternion(info['lidar2ego_rotation']))
        box.translate(np.array(info['lidar2ego_translation']))
        # filter det in ego.
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        # Move box to global coord system
        box.rotate(pyquaternion.Quaternion(info['ego2global_rotation']))
        box.translate(np.array(info['ego2global_translation']))
        box_list.append(box)
    return box_list
