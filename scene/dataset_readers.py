#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple, List
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from recordclass import recordclass, RecordClass



class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    depth: np.array
    image_path: str
    image_name: str
    width: int
    height: int    
class SpatialGenCameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    depth: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    points: np.array
    colors: np.array
    scales: np.array

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    hold_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, step=1, max_cameras=None, load_depth=True):
    cam_infos = []
    holdout_infos = []
    # print(cam_extrinsics)
    if max_cameras is not None:
        keys = list(cam_extrinsics.keys())
        subkeys = []
        N = len(keys)

        j = 0
        holdout = list(range(N))
        for i in range(max_cameras):
            j += step
            subkeys.append(keys[j%N])
            holdout.remove(j%N)
            #subkeys.append(keys[j%N])
            print('Reading Key ', str(j%N))
        holdout = [keys[i] for i in holdout]
    else:
        subkeys = cam_extrinsics
        holdout = []
    # print(subkeys)        
    for idx, key in enumerate(subkeys):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        depth = None
        if load_depth:
            depth_path = os.path.join(os.path.dirname(images_folder), 'depths', image_name+'.npy')
            depth= np.load(depth_path)


        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, depth=depth,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    

    for idx, key in enumerate(holdout):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Holdout camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        depth_path = os.path.join(os.path.dirname(images_folder), 'depths', image_name+'.npy')
        depth= np.load(depth_path)


        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, depth=depth,
                            image_path=image_path, image_name=image_name, width=width, height=height)
        holdout_infos.append(cam_info)
        sys.stdout.write('\n')
    return cam_infos, holdout_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    # colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    scales = np.vstack([vertices['sx'], vertices['sy'], vertices['sz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals, scales=scales)

def storePly(path, xyz, rgb, scales=None):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
            ('sx', 'f4'), ('sy', 'f4'), ('sz', 'f4')]
    
    normals = np.zeros_like(xyz)
    if scales is None:
        scales = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb, scales), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8, step=1, max_cameras=None, load_depth=True):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except Exception as e:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted, holdout_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir), step=step, max_cameras=max_cameras, load_depth=load_depth)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    holdout_infos = sorted(holdout_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           hold_cameras=holdout_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

from typing import Dict, Any
def readSpatialgenCameras(data_dict: Dict[str, Any], load_depth: bool = True) -> List[SpatialGenCameraInfo]:
    # load rgbs, depths, poses
    input_rgbs, input_depths, input_poses, target_rgbs, target_depths, target_poses, intrinsic_np = \
        data_dict["input_rgbs"], data_dict["input_depths"], data_dict["input_poses"], \
        data_dict["target_rgbs"], data_dict["target_depths"], data_dict["target_poses"], \
        data_dict["intrinsic"]
    
    input_points, input_colors = data_dict["input_points"], data_dict["input_colors"]
    target_points, target_colors = data_dict["target_points"], data_dict["target_colors"]
        
    num_cams = len(input_rgbs) + len(target_rgbs)

    cam_infos = []
    holdout_infos = []
    # print(subkeys)    
        
    # take the first num_init_views as RGBD frames
    uid = 0
    for idx, (input_rgb, input_depth, input_pose, input_point, input_color) in enumerate(zip(input_rgbs, input_depths, input_poses, input_points, input_colors)):
        image = Image.fromarray(input_rgb.astype(np.uint8))
        
        c2w_pose = input_pose
        w2c_pose = np.linalg.inv(c2w_pose)

        height = input_rgb.shape[0]
        width = input_rgb.shape[1]

        # c2w rotation
        R = c2w_pose[:3, :3]
        # w2c trans
        T = w2c_pose[:3, 3]

        focal_length_x = intrinsic_np[0, 0]
        FovY = focal2fov(focal_length_x, height)
        FovX = focal2fov(focal_length_x, width)

        image_name = f"input_{idx}"

        depth = input_depth
        init_scales = depth / focal_length_x / np.sqrt(2)
        init_scales = init_scales[:, :, None].repeat(3, axis=-1)
        scales = init_scales.reshape(-1, 3)
        # print(f"scales shape: {scales.shape}, points shape: {input_point.shape}, colors shape: {input_color.shape}")

        uid = idx
        cam_info = SpatialGenCameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, depth=depth,
                              image_path=f'input_rgb_{idx}', image_name=image_name, width=width, height=height,
                              points=input_point, colors=input_color, scales=scales)
        cam_infos.append(cam_info)
    
    for idx, (tar_rgb, tar_depth, tar_pose, tar_point, tar_color) in enumerate(zip(target_rgbs, target_depths, target_poses, target_points, target_colors)):
        image = Image.fromarray(tar_rgb.astype(np.uint8))
        
        c2w_pose = tar_pose
        w2c_pose = np.linalg.inv(c2w_pose)

        height = tar_rgb.shape[0]
        width = tar_rgb.shape[1]

        # c2w rotation
        R = c2w_pose[:3, :3]
        # w2c trans
        T = w2c_pose[:3, 3]

        focal_length_x = intrinsic_np[0, 0]
        FovY = focal2fov(focal_length_x, height)
        FovX = focal2fov(focal_length_x, width)

        image_name = f"target_{idx}"

        depth = tar_depth
        init_scales = depth / focal_length_x / np.sqrt(2)
        init_scales = init_scales[:, :, None].repeat(3, axis=-1)
        scales = init_scales.reshape(-1, 3)
        # print(f"scales shape: {scales.shape}, points shape: {tar_point.shape}, colors shape: {tar_color.shape}") 

        uid += 1
        cam_info = SpatialGenCameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, depth=depth,
                              image_path=f'target_rgb_{uid}', image_name=image_name, width=width, height=height,
                              points=tar_point, colors=tar_color, scales=scales)
        cam_infos.append(cam_info)
    
    return cam_infos

# def readSpatialGenSceneInfo(path, images, eval, llffhold=8, step=1, max_cameras=None, load_depth=True):
def readSpatialGenSceneInfo(path, eval, load_depth=True):
    try:
        scene_info_npzfile = os.path.join(path, "inference_results.npz")
        rooms_infer_res_dict = np.load(scene_info_npzfile, allow_pickle=True)
        room_uid = list(rooms_infer_res_dict.keys())[0]
        room_infer_results = rooms_infer_res_dict[room_uid][()]
        
    except Exception as e:
        print(e)
        sys.exit(-1)
        
    cam_infos_unsorted: List[SpatialGenCameraInfo] = readSpatialgenCameras(data_dict=room_infer_results, load_depth=load_depth)
    cam_infos: List[SpatialGenCameraInfo] = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    # holdout_infos = sorted(holdout_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos)]
        test_cam_infos = [c for idx, c in enumerate(cam_infos)]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    # ply_path = os.path.join(path, "gs_ply.ply")
    # o3d_ply_path = os.path.join(path, "global_scene_ply.ply")
    # if not os.path.exists(ply_path):
    #     print("Converting point cloud from open3d to plydata, will happen only the first time you open the scene.")
    #     import open3d as o3d
    #     o3d_ply = o3d.io.read_point_cloud(o3d_ply_path)
    #     xyz = np.array(o3d_ply.points)
    #     rgb = np.array(o3d_ply.colors)
    #     print(f"xyz shape: {xyz.shape}, rgb shape: {rgb.shape}")
    #     storePly(ply_path, xyz, rgb)
    ply_path = os.path.join(path, "points3D.ply")
    if not os.path.exists(ply_path):
        print("Converting point cloud from SpatialGenCameraInfo to plydata, will happen only the first time you open the scene.")
        xyz_lst, rgb_lst, scale_lst = [], [], []
        for cam_info in train_cam_infos:
            if cam_info.points is not None:
                xyz = cam_info.points
                rgb = cam_info.colors
                scale = cam_info.scales
                xyz_lst.append(xyz)
                rgb_lst.append(rgb)
                scale_lst.append(scale)
        xyz_lst = np.concatenate(xyz_lst, axis=0)
        rgb_lst = np.concatenate(rgb_lst, axis=0)
        scale_lst = np.concatenate(scale_lst, axis=0)
        storePly(ply_path, xyz_lst, rgb_lst, scale_lst)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           hold_cameras=[],
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Spatialgen": readSpatialGenSceneInfo,
}