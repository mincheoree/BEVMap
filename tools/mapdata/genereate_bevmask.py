import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import cv2
import os 
from nuscenes.utils.data_classes import Box
from nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix
import argparse


def gen_dx_bx(xbound, ybound, zbound):
    dx = np.array([row[2] for row in [xbound, ybound, zbound]])
    bx = np.array([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = np.array([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])
    return dx, bx, nx

def get_binmap(
        nusc, 
        nusc_maps, 
        rec, 
        layer_names, 
        coord_type, 
        xbound = [-51.2, 51.2, 0.128],
        ybound=[-51.2, 51.2, 0.128],
        zbound=[-10.0, 10.0, 20.0],
    ):

        dx, bx, _ = gen_dx_bx(xbound, ybound, zbound)
        bx = bx[:2]
        dx = dx[:2]
        sample = nusc.get('sample_data', rec['data']['LIDAR_TOP'])
        egopose = nusc.get('ego_pose',  sample['ego_pose_token'])
        center = np.array([egopose["translation"][0], egopose["translation"][1]])
        cs_record = nusc.get("calibrated_sensor", sample["calibrated_sensor_token"])
        map_name = scene2map[nusc.get('scene', rec['scene_token'])['name']]
        cs_center = np.array([cs_record["translation"][0], cs_record["translation"][1], 0])

        nmap = nusc_maps[map_name]
        stretch = 51.2

        box_coords = (center[0] - stretch, 
                      center[1] - stretch, 
                      center[0] + stretch, 
                      center[1] + stretch,)

        masks = []
        # polygons
        records_in_patch = nmap.get_records_in_patch(box_coords,layer_names=layer_names,  mode='intersect')
        for layer_name in layer_names:
            mask_height = int((ybound[1] - ybound[0]) / ybound[2])
            mask_width = int((xbound[1] - xbound[0]) / xbound[2])
            mask = np.zeros((mask_height, mask_width), dtype=np.uint8)
            for token in records_in_patch[layer_name]:
                poly_record = nmap.get(layer_name, token)
        
                if layer_name == 'drivable_area':
                    polygon_tokens = poly_record['polygon_tokens']
                else:
                    polygon_tokens = [poly_record['polygon_token']]

                for polygon_token in polygon_tokens:
                    polygon = nmap.extract_polygon(polygon_token)
                    points = np.array(polygon.exterior.xy)
                    points -= center.reshape((-1, 1))
                    ## add z coordinates 
                    points = np.vstack((points, np.zeros((1, points.shape[1]))))
                    points = np.dot(Quaternion(egopose['rotation']).inverse.rotation_matrix, points) 
                    if coord_type == 'lidar':
                        # ego -> lidar (calibrated sensor)
                        points -= cs_center.reshape((-1, 1))
                        points = np.dot(Quaternion(cs_record['rotation']).inverse.rotation_matrix, points)
                    exteriors = points.T[:, :2]
                    pts = np.round((exteriors - bx)/dx).astype(np.int32)
                    cv2.fillPoly(mask, [pts], 1.0)
                    
                    if len(polygon.interiors) > 0:
                        ptsi = []
                        for pi in polygon.interiors:
                            points = np.array(pi.xy)
                            points -= center.reshape((-1, 1))
                            ## add z coordinates 
                            points = np.vstack((points, np.zeros((1, points.shape[1]))))
                            points = np.dot(Quaternion(egopose['rotation']).inverse.rotation_matrix, points) 
                            if coord_type == 'lidar':
                                # ego -> lidar (calibrated sensor)
                                points -= cs_center.reshape((-1, 1))
                                points = np.dot(Quaternion(cs_record['rotation']).inverse.rotation_matrix, points)
                            interiors = points.T[:, :2]
                            pts = np.round((interiors - bx)/dx).astype(np.int32)    
                            ptsi.append(pts)
                        cv2.fillPoly(mask, ptsi, 0.0)

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

    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--coord", type=str, default='lidar', help="specify coordinate of BEV map"
    )
    parser.add_argument(
        "--category", type = str, default='all'
    )
    args = parser.parse_args()
    category = args.category 
    coord = args.coord
    dataroot = 'data/nuscenes/'
    nusc_maps = get_nusc_maps(dataroot)
    nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True)
    scene2map = {}
    for rec in nusc.scene: 
        log = nusc.get('log', rec['log_token'])
        scene2map[rec['name']] = log['location']
    layer_names = ['drivable_area', 'road_segment', 'road_block', 'lane', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area']
    # line_names = ['road_divider', 'lane_divider']

    if category == 'all':
        for sample in tqdm(nusc.sample): 
        
            map = get_binmap(nusc, nusc_maps, sample, layer_names, coord)
            index = sample['data']['LIDAR_TOP']
            os.makedirs(os.path.join(dataroot, 'bevmap', str(index)))
            for i in range(map.shape[0]):
                single_mask = map[i] * 255
                cv2.imwrite(os.path.join(dataroot, 'bevmap', str(index), f'{str(i)}.png'), single_mask.astype(np.uint8))
    else: 
        # get certain categories
        i = (layer_names).index(category)
        os.makedirs(os.path.join(dataroot, 'bevmap'))
        for sample in tqdm(nusc.sample): 
            map = get_binmap(nusc, nusc_maps, sample, layer_names, coord)
            index = sample['data']['LIDAR_TOP']
            single_mask = map[i] * 255
            cv2.imwrite(os.path.join(dataroot, 'bevmap', f'{str(index)}.png'), single_mask.astype(np.uint8))


        

            