import matplotlib.pyplot as plt
from tqdm import tqdm
from shapely.geometry import Polygon
import descartes
import numpy as np
import cv2
from nuscenes.utils.geometry_utils import view_points
import os 
from nuscenes.utils.data_classes import Box
from nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer 
# from nuscenes.map_expansion import arcline_path_utils
# from nuscenes.map_expansion.bitmap import BitMap
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix
from PIL import Image
from typing import Dict, List, Tuple, Optional, Union
import argparse
import copy

def get_coordinate_from_binarymap(binary_map):
        point_cloud_range = np.array([-51.2, -51.2, 51.2, 51.2], dtype=np.float32)

        py, px = np.where(binary_map)
        pixel_centers = np.stack([px, py]).T

        height, width = binary_map.shape
        pixel_size = (point_cloud_range[2:] - point_cloud_range[:2]) / binary_map.shape

        pixel_centers = (pixel_centers + 0.5) * pixel_size + point_cloud_range[:2]
        return pixel_centers


def get_nusc_maps(map_folder):
    nusc_maps = {map_name: NuScenesMap(dataroot=map_folder,
                map_name=map_name) for map_name in [
                    "singapore-hollandvillage", 
                    "singapore-queenstown",
                    "boston-seaport",
                    "singapore-onenorth",
                ]}
    return nusc_maps


def render_depth_in_image(nusc,
                        nusc_map,
                         sample_token, 
                            camera_channel,
                            near_plane = 1e-8, 
                            render_behind_cam = True, 
                            render_outside_im = True,
                            min_polygon_area = 1000,
                            alpha = 1.0,
                            patch_radius = 50.0, 
                            im_size = (1600, 900), 
                            out_path = None,
                            dataroot = None
                            ):
     
        layer_names = ['road_segment', 'lane', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area']
        # Check layers whether we can render them.

        sample_record = nusc.get('sample', sample_token)
        cam_token = sample_record['data'][camera_channel]
        cam_record = nusc.get('sample_data', cam_token)
        cs_record = nusc.get('calibrated_sensor', cam_record['calibrated_sensor_token'])
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])

        # Retrieve the current map
        poserecord =  nusc.get('ego_pose', cam_record['ego_pose_token'])
        ego_pose = poserecord['translation']
        box_coords = (
        ego_pose[0] - patch_radius,
        ego_pose[1] - patch_radius,
        ego_pose[0] + patch_radius,
        ego_pose[1] + patch_radius,
    )
       
        bevpath = os.path.join(dataroot, f'bevmap/{sample_token}.png')
        bevmap = np.array(Image.open(bevpath))
        points = get_coordinate_from_binarymap(bevmap) 

        # Convert bev map points to pointcloud with -1 height.
        points = np.vstack((points.T, np.zeros((1, points.shape[0])) - 1 ))        

        pointsensor = nusc.get('sample', sample_token)
        point_token = sample_record['data']['LIDAR_TOP']
        pointsensor = nusc.get('sample_data', point_token)

        point_cs = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
        point_pose = nusc.get('ego_pose', pointsensor['ego_pose_token'])
        ## convert lidar points back to egopose 
        points = np.dot(Quaternion(point_cs['rotation']).rotation_matrix, points)
        points += np.array(point_cs['translation']).reshape((-1, 1))
        ## convert egopose back to global 
        points = np.dot(Quaternion(point_pose['rotation']).rotation_matrix, points)
        points += np.array(point_pose['translation']).reshape((-1, 1))
        
        
        # Transform into the ego vehicle frame for the timestamp of the image.
        points = points - np.array(poserecord['translation']).reshape((-1, 1))
        points = np.dot(Quaternion(poserecord['rotation']).rotation_matrix.T, points)

        # Transform into the camera.
        points = points - np.array(cs_record['translation']).reshape((-1, 1))
        points = np.dot(Quaternion(cs_record['rotation']).rotation_matrix.T, points)

        # Remove points that are partially behind the camera.
        depths = points[2, :]
#             print(points.shape)
#             print(depths)
        front = depths > near_plane

        points = points[:, front]
        depths = depths[front]
        # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
        points = view_points(points, cam_intrinsic, normalize=True)
            
        # Skip polygons where all points are outside the image.
        # Leave a margin of 1 pixel for aesthetic reasons.
        inside = np.ones(points.shape[1], dtype=bool)
        inside = np.logical_and(inside, points[0, :] > 1)
        inside = np.logical_and(inside, points[0, :] < im_size[0] - 1)
        inside = np.logical_and(inside, points[1, :] > 1)
        inside = np.logical_and(inside, points[1, :] < im_size[1] - 1)

        points = points[:, inside]
        depths = depths[inside]
        # points = points[:2, :]
        
        
        # ### get polygon
        # points = [(p0, p1) for (p0, p1) in zip(points[0], points[1])]
    
        # proj_range = np.full((900, 1600), -1,
        #                       dtype=np.float32)
        # clip depths to 51.2m 
        depths = np.minimum(depths, 51.2)
        # proj_range[np.floor(points[1]).astype(np.int32), np.floor(points[0]).astype(np.int32)] = depths
    
        depth_points = np.concatenate([points[:2].T, depths[:, None]], axis =1).astype(np.float32)

        np.save(f'{out_path}', depth_points)


if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description="arg parser")
    args = parser.parse_args()
    dataroot = 'data/nuscenes'

    nusc_maps = get_nusc_maps(dataroot)
    nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True)
    scene2map = {}

    for rec in nusc.scene: 
        log = nusc.get('log', rec['log_token'])
        scene2map[rec['name']] = log['location']

    layer_names = ['road_segment', 'road_block', 'lane', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area']
    cams = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    os.makedirs(os.path.join(dataroot, 'projdepth'))
    for cam in cams: 
        os.makedirs(os.path.join(dataroot, 'projdepth', cam))
    for sample in tqdm(nusc.sample): 
        sample_token = sample['token']
        map_name = scene2map[nusc.get('scene', sample['scene_token'])['name']]
        nusc_map = nusc_maps[map_name]
        for cam in cams:
            cam_token = sample['data'][cam]
            out_path = os.path.join(dataroot, 'projdepth', cam, cam_token)
            render_depth_in_image(nusc, nusc_map, sample_token = sample_token, camera_channel=cam, out_path = out_path, dataroot = dataroot)
