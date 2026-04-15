import sys
sys.path.append('../camera-control')
sys.path.append('../../../camera-control')

import os
import cv2
import numpy as np
import open3d as o3d

from camera_control.Module.camera_convertor import CameraConvertor

from infini_depth.Module.detector import Detector


def demo():
    home = os.environ['HOME']
    model_file_path = f'{home}/chLi/Model/InfiniDepth/infinidepth.ckpt'

    colmap_data_folder_path = f'{home}/chLi/Dataset/GS/haizei_zihan/colmap_normalized/'
    depth_folder_path = f'{home}/chLi/Dataset/GS/haizei_zihan/depth/'
    depth_vis_folder_path = f'{home}/chLi/Dataset/GS/haizei_zihan/depth_vis/'
    masked_depth_vis_folder_path = f'{home}/chLi/Dataset/GS/haizei_zihan/masked_depth_vis/'
    pcd_folder_path = f'{home}/chLi/Dataset/GS/haizei_zihan/pcd/'

    camera_list = CameraConvertor.loadColmapDataFolder(colmap_data_folder_path)

    detector = Detector(model_file_path)

    result_cameras = detector.detectCameras(camera_list)

    print(f'[INFO][demo] Got {len(result_cameras)} cameras with depth')

    os.makedirs(depth_folder_path, exist_ok=True)
    os.makedirs(depth_vis_folder_path, exist_ok=True)
    os.makedirs(masked_depth_vis_folder_path, exist_ok=True)
    os.makedirs(pcd_folder_path, exist_ok=True)

    for i, camera in enumerate(result_cameras):
        image_filename = camera.image_id
        format = '.' + image_filename.split('.')[-1]
        image_basename = image_filename.split(format)[0]

        img_shape = tuple(camera.image.shape) if camera.image is not None else None
        depth_shape = tuple(camera.depth.shape) if camera.depth is not None else None
        print(f'\t camera {i}: image shape = {img_shape}, depth shape = {depth_shape}')

        np.save(depth_folder_path + image_basename + '.npy', camera.depth_with_conf.cpu().numpy())
        cv2.imwrite(depth_vis_folder_path + image_filename, camera.toDepthVisCV(use_mask=False))
        cv2.imwrite(masked_depth_vis_folder_path + image_filename, camera.toDepthVisCV(use_mask=True))

        pts = camera.toDepthPoints(use_mask=True)[0].cpu().numpy().reshape(-1, 3)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        o3d.io.write_point_cloud(pcd_folder_path + image_basename + '.ply', pcd)

    return True
