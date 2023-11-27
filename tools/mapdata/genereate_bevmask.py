import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import cv2
import os 
import pdb
from nuscenes.utils.data_classes import Box
from nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
# from nuscenes.map_expansion import arcline_path_utils
# from nuscenes.map_expansion.bitmap import BitMap
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix
import argparse

def get_rot(h):
    return np.array([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])

def get_binmap(nusc, nusc_maps, rec, layer_names, line_names):
        bx = np.array([-49.75,-49.75])
        dx = np.array([0.5,0.5])
        import pdb; pdb.set_trace()
        sample = nusc.get('sample_data', rec['data']['LIDAR_TOP'])
        egopose = nusc.get('ego_pose',  sample['ego_pose_token'])
        rot = Quaternion(egopose['rotation']).rotation_matrix
        egorot = np.arctan2(rot[1, 0], rot[0, 0])
        egocenter = np.array([egopose['translation'][0], egopose['translation'][1], np.cos(egorot), np.sin(egorot)])
        cs_record = nusc.get("calibrated_sensor", sample["calibrated_sensor_token"])
        rot = Quaternion(cs_record['rotation']).rotation_matrix
        csrot = np.arctan2(rot[1, 0], rot[0, 0])
        cscenter = np.array([cs_record['translation'][0], cs_record['translation'][1], np.cos(csrot), np.sin(csrot)])
        
        map_name = scene2map[nusc.get('scene', rec['scene_token'])['name']]
        # nmap = self.nusc_maps[map_name]
        nmap = nusc_maps[map_name]
        stretch = 50.0
        # layer_names = ['drivable_area']

        box_coords = (egocenter[0] - stretch, 
                      egocenter[1] - stretch, 
                      egocenter[0] + stretch, 
                      egocenter[1] + stretch,)

        polys = {}

        # polygons
        records_in_patch = nmap.get_records_in_patch(box_coords,layer_names=layer_names,  mode='intersect')
        for layer_name in layer_names:
            polys[layer_name] = []
            for token in records_in_patch[layer_name]:
                poly_record = nmap.get(layer_name, token)
        
                if layer_name == 'drivable_area':
                    polygon_tokens = poly_record['polygon_tokens']
                else:
                    polygon_tokens = [poly_record['polygon_token']]

                for polygon_token in polygon_tokens:
                    polygon = nmap.extract_polygon(polygon_token)
                    polys[layer_name].append(np.array(polygon.exterior.xy))
        
        # lines
        for layer_name in line_names:
            polys[layer_name] = []
            for record in getattr(nmap, layer_name):
                token = record['token']

                line = nmap.extract_line(record['line_token'])
                if line.is_empty:  # Skip lines without nodes
                    continue
                xs, ys = line.xy

                polys[layer_name].append(np.array([xs, ys]))

        
        rot = get_rot(np.arctan2(egocenter[3], egocenter[2])).T
        csrot = get_rot(np.arctan2(cscenter[3], cscenter[2])).T

        for layer_name in polys:
            for rowi in range(len(polys[layer_name])):
                # global -> ego
                polys[layer_name][rowi] -= egocenter[:2]
                polys[layer_name][rowi] = np.dot(polys[layer_name][rowi], rot)
                # ego -> lidar (calibrated sensor)
                # polys[layer_name][rowi] -= cscenter[:2]
                # polys[layer_name][rowi] = np.dot(polys[layer_name][rowi], csrot)

        pts_list = []
        masks = []

        for name in layer_names + line_names: 
            mask = np.zeros((200, 200))
            for la in polys[name]: 
                pts = np.round(((la) - bx)/dx).astype(np.int32)
                pts[:, [1, 0]] = pts[:, [0, 1]]
                
                cv2.fillPoly(mask, np.int32([pts]), 1.0)
              
                # cv2.polylines(mask, [pts], False, 1, 2)
            masks.append(mask)
        return np.array(masks)

def get_nusc_maps(map_folder):
    nusc_maps = {map_name: NuScenesMap(dataroot=map_folder,
                map_name=map_name) for map_name in [
                    "singapore-hollandvillage", 
                    "singapore-queenstown",
                    "boston-seaport",
                    "singapore-onenorth",
                ]}
    return nusc_maps

if __name__ == '__main__': 
    nusc_maps = get_nusc_maps('data/mini/')
    nusc = NuScenes(version='v1.0-mini', dataroot='data/mini/', verbose=True)
    scene2map = {}
    for rec in nusc.scene: 
        log = nusc.get('log', rec['log_token'])
        scene2map[rec['name']] = log['location']
    layer_names = ['drivable_area', 'road_segment', 'road_block', 'lane', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area']
    line_names = ['road_divider', 'lane_divider']
    
    for sample in tqdm(nusc.sample): 
    
        map = get_binmap(nusc, nusc_maps, sample, layer_names, line_names)
        index = sample['data']['LIDAR_TOP']
        os.makedirs('data/mini/bevmap' + '/' + str(index))
        for i in range(map.shape[0]):
            single_mask = map[i] * 255
            cv2.imwrite('data/mini/bevmap' + '/' + str(index) + '/' + f'{str(i)}.png', single_mask.astype(np.uint8))
 
