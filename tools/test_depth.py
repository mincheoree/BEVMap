import torch 
from nuscenes.nuscenes import NuScenes
import mmcv
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv import Config, DictAction
import argparse
from mmdet3d.models import build_model
import torchvision
from PIL import Image
import os 
import numpy as np
import matplotlib.pyplot as plt

normalize_img = torchvision.transforms.Compose((
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])))

def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize Depth')
    parser.add_argument('--config', help='test config file path')
    args = parser.parse_args()
    return args

if __name__ == '__main__': 
    args = parse_args()
    # import pdb; pdb.set_trace()
    cfg = Config.fromfile(args.config)
    nusc = NuScenes(version='v1.0-{}'.format('mini'),
                    dataroot='data/nuscenes',
                    verbose=False)
    sample = nusc.sample[-1]
    cam_type = 'CAM_FRONT_LEFT'
    filename = nusc.get('sample_data', sample['data'][cam_type])['filename']
    img = Image.open('data/nuscenes/' + filename).resize((704, 256))
    img.save('orig.png')
    img = normalize_img(img)
    
    
    map_root = '../mapdataset'
 
    fname = nusc.get('sample_data', sample['data'][cam_type])['ego_pose_token']
    proj = os.path.join(map_root, 'projmap', cam_type, f'{fname}.jpg')
    depth = os.path.join(map_root, 'depth', cam_type, f'{fname}.npy')
    map = normalize_img(Image.open(proj).resize((704, 256)))
    cam_depth = np.load(depth)
    resize_dims = (256, 704)
    resize = float(704)/float(1600)
    cam_depth[:, :2] = cam_depth[:, :2] * resize

    depth_coords = cam_depth[:, :2].astype(np.int16)

    depth_map = np.zeros(resize_dims)
    valid_mask = ((depth_coords[:, 1] < resize_dims[0])
                & (depth_coords[:, 0] < resize_dims[1])
                & (depth_coords[:, 1] >= 0)
                & (depth_coords[:, 0] >= 0))
    depth_map[depth_coords[valid_mask, 1],
            depth_coords[valid_mask, 0]] = cam_depth[valid_mask, 2]
    depth = torch.Tensor(depth_map)/70.0
    
    map = torch.cat((map, depth.unsqueeze(0)), dim = 0)

    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    BASELINE_PATH = 'bevdet-r50.pth'
    PATH='work_dirs/spade2/epoch_24.pth'
    model.load_state_dict(torch.load(PATH)['state_dict'])
    # checkpoint = load_checkpoint(model, 'bevdet-r50.pth', map_location='cpu')
    img_encoder = model.img_backbone
    img_neck = model.img_neck

    feat = img_neck(img_encoder(img.unsqueeze(0)))
    map_encoder = model.img_view_transformer.map_encoder
    map_feat = map_encoder(map.unsqueeze(0))
    
    fmap = torch.cat((feat, map_feat), dim = 1)
    depthnet = model.img_view_transformer.depthnet
    depth = depthnet(fmap)[0][:59].softmax(dim = 0)
    depthmap = torch.argmax(depth, dim = 0)
    plt.axis('off')

   
    im = plt.imshow(depthmap, cmap = 'plasma')    
    plt.tight_layout()
    # plt.colorbar()
    plt.colorbar(im, orientation="horizontal", pad=0.2)
    plt.savefig('depth1.png')
    
    import pdb; pdb.set_trace()

