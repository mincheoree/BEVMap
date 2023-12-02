import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt
import descartes
from tqdm import tqdm
import numpy as np
import cv2
import os 
from nuscenes.utils.data_classes import Box
from nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.utils.geometry_utils import view_points
from shapely.geometry import Polygon, MultiPolygon, LineString, Point, box

# from nuscenes.map_expansion import arcline_path_utils
# from nuscenes.map_expansion.bitmap import BitMap
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix
from PIL import Image

def get_nusc_maps(map_folder):
    nusc_maps = {map_name: NuScenesMap(dataroot=map_folder,
                map_name=map_name) for map_name in [
                    "singapore-hollandvillage", 
                    "singapore-queenstown",
                    "boston-seaport",
                    "singapore-onenorth",
                ]}
    return nusc_maps

def depth_transform(cam_depth, resize, resize_dims, crop, flip, rotate):
        """Transform depth based on ida augmentation configuration.

        Args:
            cam_depth (np array): Nx3, 3: x,y,d.
            resize (float): Resize factor.
            resize_dims (list): Final dimension.
            crop (list): x1, y1, x2, y2
            flip (bool): Whether to flip.
            rotate (float): Rotation value.

        Returns:
            np array: [h/down_ratio, w/down_ratio, d]
        """

        import pdb; pdb.set_trace()
        H, W = resize_dims
    
        cam_depth[:, :2] = cam_depth[:, :2] * resize
        cam_depth[:, 0] -= crop[0]
        cam_depth[:, 1] -= crop[1]
        if flip:
            cam_depth[:, 0] = resize_dims[1] - cam_depth[:, 0]

        cam_depth[:, 0] -= W / 2.0
        cam_depth[:, 1] -= H / 2.0

        h = rotate / 180 * np.pi
        rot_matrix = [
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ]
        cam_depth[:, :2] = np.matmul(rot_matrix, cam_depth[:, :2].T).T

        cam_depth[:, 0] += W / 2.0
        cam_depth[:, 1] += H / 2.0

        depth_coords = cam_depth[:, :2].astype(np.int16)

        ## intialize depth map with -1 
        depth_map = np.zeros(resize_dims) - 1
        valid_mask = ((depth_coords[:, 1] < resize_dims[0])
                    & (depth_coords[:, 0] < resize_dims[1])
                    & (depth_coords[:, 1] >= 0)
                    & (depth_coords[:, 0] >= 0))
        depth_map[depth_coords[valid_mask, 1],
                depth_coords[valid_mask, 0]] = cam_depth[valid_mask, 2]

        return depth_map

import matplotlib.pyplot as plt


dataroot = 'data/nuscenes/'
nusc_maps = get_nusc_maps(dataroot)
nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True)
scene2map = {}

for rec in nusc.scene: 
    log = nusc.get('log', rec['log_token'])
    scene2map[rec['name']] = log['location']

layer_names = ['road_segment', 'lane', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area']
cams = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

sample = nusc.sample[1]
sample_token = sample['token']
datatoken = nusc.get('sample', sample_token)['data']['CAM_BACK_LEFT']
cam = Image.open(f'data/nuscenes/projmap/CAM_BACK_LEFT/{datatoken}.jpg')
depth = np.load(f'./data/nuscenes/projdepth/CAM_BACK_LEFT/{datatoken}.npy')

data_config={
    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
    'Ncams': 6,
    'input_size': (256, 704),
    'src_size': (900, 1600),

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test':0.04,
}

import torch

import matplotlib.pyplot as plt
def draw_mapdepth(map, depth):
    plt.clf()
    plt.imshow(map)
    cm = plt.cm.get_cmap('RdYlBu')

    plt.scatter(depth[:, 0], depth[:, 1], vmin = 0, vmax = 51.2, s = 1, c = depth[:, 2], cmap = cm)
    plt.colorbar()
    plt.savefig('viz.png')
    plt.close()

def get_rot(h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

def img_transform_core(img, resize_dims, crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        return img

def img_transform(img, post_rot, post_tran,
                      resize, resize_dims, crop,
                      flip, rotate):
        # adjust image
        img = img_transform_core(img, resize_dims, crop, flip, rotate)

        # post-homography transformation
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b
        return img, post_rot, post_tran

def sample_augmentation(H , W, flip=None, scale=None):
        fH, fW = 128, 352

        resize = float(fW)/float(W)
        resize += np.random.uniform(*data_config['resize'])
        
        resize_dims = (int(W * resize), int(H * resize))
    
        newW, newH = resize_dims
        crop_h = int((1 - np.random.uniform(*data_config['crop_h'])) * newH) - fH
        crop_w = int(np.random.uniform(0, max(0, newW - fW)))
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        flip = data_config['flip'] and np.random.choice([0, 1])
        rotate = np.random.uniform(*data_config['rot'])
    
        return resize, resize_dims, crop, flip, rotate

resize, resize_dims, crop, flip, rotate = sample_augmentation(900, 1600, flip = True, scale = None)
post_rot = torch.eye(2)
post_tran = torch.zeros(2)
camaug, post_rot2, post_tran2 = img_transform(cam, post_rot, post_tran,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate)
import pdb; pdb.set_trace()
depthaug = depth_transform(depth, resize, resize_dims, crop, flip, rotate)
