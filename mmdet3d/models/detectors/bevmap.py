# Copyright (c) Vision & AI Lab. All rights reserved

import torch
from mmcv.runner import force_fp32
import torch.nn.functional as F
from torch import nn

from mmdet.models import DETECTORS
from .centerpoint import CenterPoint
from .bevdet import BEVDepth
from .. import builder

@DETECTORS.register_module()
class BEVDet_Map(CenterPoint):
    def __init__(self, img_view_transformer, img_bev_encoder_backbone, img_bev_encoder_neck, **kwargs):
        super(BEVDet_Map, self).__init__(**kwargs)

        nhidden = 128
        norm_nc = 64
        ks = 3
        pw = ks // 2
        label_nc = 1
        self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def spade(self, x, segmap): 
        '''
        Adopted from SPADE paper
        '''
        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta
        return out

    def extract_img_feat(self, img, img_metas, map, bev):
        """Extract features of images."""
        x = self.image_encoder(img[0])
      
        x = self.img_view_transformer([x] + img[1:], map)
        x = self.spade(x, bev)
        x = self.bev_encoder(x)
        return [x]
    
    def extract_feat(self, points, img, img_metas, map, bev):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas, map, bev)
        pts_feats = None
        return (img_feats, pts_feats)
    
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      proj = None, 
                      depth = None,
                      bev = None):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        # depth = self.filter_depth(depth)
        
        map = torch.cat((proj, torch.unsqueeze(depth, dim = 2)), dim = 2)
        img_feats, pts_feats = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, map = map, bev = bev)
        assert self.with_pts_bbox
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)
        return losses

    def forward_test(self, points=None, img_metas=None, img_inputs=None, proj = None, depth = None, bev = None, **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
       
        # depth = self.filter_depth(depth[0])
      
        map = torch.cat((proj[0], torch.unsqueeze(depth[0], dim = 2)), dim = 2)
        
    
        # map = torch.cat((proj[0], torch.unsqueeze(depth[0], dim = 2)), dim = 2)


        for var, name in [(img_inputs, 'img_inputs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(img_inputs)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(img_inputs), len(img_metas)))

        if not isinstance(img_inputs[0][0],list):
            img_inputs = [img_inputs] if img_inputs is None else img_inputs
            points = [points] if points is None else points
            return self.simple_test(points[0], img_metas[0], img_inputs[0], map, bev[0].unsqueeze(0), **kwargs)
        else:
            return self.aug_test(None, img_metas[0], img_inputs[0], **kwargs)

    def aug_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        combine_type = self.test_cfg.get('combine_type','output')
        if combine_type=='output':
            return self.aug_test_combine_output(points, img_metas, img, rescale)
        elif combine_type=='feature':
            return self.aug_test_combine_feature(points, img_metas, img, rescale)
        else:
            assert False

    def simple_test(self, points, img_metas, img=None, map=None, bev=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats, _ = self.extract_feat(points, img=img, img_metas=img_metas, map = map, bev = bev)
        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list


    def forward_dummy(self, points=None, img_metas=None, img_inputs=None, **kwargs):
        img_feats, _ = self.extract_feat(points, img=img_inputs, img_metas=img_metas)
        from mmdet3d.core.bbox.structures.box_3d_mode import LiDARInstance3DBoxes
        img_metas=[dict(box_type_3d=LiDARInstance3DBoxes)]
        bbox_list = [dict() for _ in range(1)]
        assert self.with_pts_bbox
        bbox_pts = self.simple_test_pts(
            img_feats, img_metas, rescale=False)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list


@DETECTORS.register_module()
class BEVDepth_Map(BEVDepth_Base, BEVDet_Map):
    def extract_img_feat(self, img, img_metas, map, bev):
        """Extract features of images."""
        x = self.image_encoder(img[0])
        x, depth = self.img_view_transformer([x] + img[1:], map)
        x = self.spade(x, bev)
        x = self.bev_encoder(x)
        return [x], depth

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proj = None, 
                      depth = None, 
                      bev = None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        map = torch.cat((proj, torch.unsqueeze(depth, dim = 2)), dim = 2)
        img_feats, pts_feats, _ = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, map = map, bev = bev)
        assert self.with_pts_bbox

        depth_gt = img_inputs[-1]
        # loss_depth = self.get_depth_loss(depth_gt, depth)
        # losses = dict(loss_depth=loss_depth)
        losses = dict()
     
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)
        return losses
