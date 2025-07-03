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

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
import torch.nn.functional as F
from einops import rearrange
from utils.typing import *
from utils.torch3d_utils import torch3d_rasterize_points

def build_rot_y(th):
        return torch.Tensor([
        [np.cos(th), 0, -np.sin(th)],
        [0, 1, 0],
        [np.sin(th), 0, np.cos(th)]])

def depth_to_pointcloud(depth_map, K) -> np.array:
    H, W = depth_map.shape[0], depth_map.shape[1]
    n_pts = H * W
    depth_map = depth_map.reshape(1, H, W)

    pts_x = np.linspace(0, W, W)
    pts_y = np.linspace(0, H, H)
    
    pts_xx, pts_yy = np.meshgrid(pts_x, pts_y)

    pts = np.stack((pts_xx, pts_yy, np.ones_like(pts_xx)), axis=0)
    pts = np.linalg.inv(K) @ (pts * depth_map).reshape(3, n_pts)
    return pts.transpose()
    
# function edge_filter is fork from VistaDream https://github.com/WHU-USI3DV/VistaDream
def nei_delta(input, pad=2):
    if not type(input) is torch.Tensor:
        input = torch.from_numpy(input.astype(np.float32))
    if len(input.shape) < 3:
        input = input[:, :, None]
    h, w, c = input.shape
    # reshape
    input = input.permute(2, 0, 1)[None]
    input = F.pad(input, pad=(pad, pad, pad, pad), mode="replicate")
    kernel = 2 * pad + 1
    input = F.unfold(input, [kernel, kernel], padding=0)
    input = input.reshape(c, -1, h, w).permute(2, 3, 0, 1).squeeze()  # hw(3)*25
    return torch.amax(input, dim=-1), torch.amin(input, dim=-1), input

def edge_filter(metric_dpt, valid_mask=None, times=0.1):
    if valid_mask is None:
        valid_mask = np.ones_like(metric_dpt, dtype=bool)
    _max = np.percentile(metric_dpt[valid_mask], 95)
    _min = np.percentile(metric_dpt[valid_mask], 5)
    _range = _max - _min
    nei_max, nei_min, _ = nei_delta(metric_dpt)
    delta = (nei_max - nei_min).numpy()
    edge = delta > times * _range
    return edge

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, depth, gt_alpha_mask,
                 image_name, uid, warp_mask, K, src_R, src_T, src_uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R   # c2w rotation
        self.T = T   # w2c translation
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.warp_mask = warp_mask
        self.K = K
        self.src_R = src_R
        self.src_T = src_T
        self.src_uid = src_uid

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.depth = depth
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        
    def update_R_and_T(self, R, T):
        self.R = R
        self.T = T
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, self.trans, self.scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
    
    def update_view_transform(self, view2world: Float[Tensor, "4 4"]):
        world2view = torch.inverse(view2world)
        self.R = view2world[:3, :3].detach().cpu().numpy()
        self.T = world2view[:3, 3].detach().cpu().numpy()
        self.world_view_transform = world2view.transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def generate_warp_gt(self):
        def depth_warping_pt3d(src_img: Float[Tensor, "C H W"], 
                               src_depth: Float[Tensor, "H W"], 
                               K: Float[Tensor, "3 3"], 
                               R_A: np.array, 
                               T_A: np.array, 
                               R_B: np.array, 
                               T_B: np.array):

            # warp the depth maps to target views
            colors = src_img[None, :, :, :]
            h, w = src_img.shape[1], src_img.shape[2]
            
            points: Float[np.array, "Np 3"] = depth_to_pointcloud(src_depth.cpu().numpy(), K.cpu().numpy())
            batch_input_points: Float[Tensor, "B Np 3"] = torch.from_numpy(points).to(src_img)[None, :, :]
            batch_input_colors: Float[Tensor, "B 3 Np"] = rearrange(colors, "B C H W -> B (H W) C", H=h, W=w)
            
            pointcloud = torch.cat([batch_input_points, batch_input_colors], dim=-1)  # B, Np, 6
            pointcloud: Float[Tensor, "BNp 6"] = rearrange(pointcloud, "B Np C -> (B Np) C")
            
            src_w2c_pose = np.eye(4)
            src_w2c_pose[:3, :3] = R_A.T
            src_w2c_pose[:3, 3] = T_A
            tar_w2c_pose = np.eye(4)
            tar_w2c_pose[:3, :3] = R_B.T
            tar_w2c_pose[:3, 3] = T_B
            relative_w2c_poses = tar_w2c_pose @ np.linalg.inv(src_w2c_pose)
            relative_w2c_poses = torch.from_numpy(relative_w2c_poses).to(src_img)
            projected_tar_imgs, projected_tar_depths = torch3d_rasterize_points(
                cv_cam_poses_c2w=None,
                cv_cam_poses_w2c=relative_w2c_poses[None, :, :].float(),
                in_pointcloud=pointcloud.float(),
                intrinsic=K.float().to(src_img),
                image_width=w,
                image_height=h,
                point_radius=0.01,
                device=src_img.device,
            )
            projected_tar_imgs = projected_tar_imgs.clamp(0, 1)
            rendered_masks: Bool[Tensor, "B 1 H W"] = projected_tar_imgs.mean(1, keepdim=True) > 0.01
            out_img = projected_tar_imgs[0].cpu().numpy()
            warp_mask = rendered_masks[0, 0].cpu().numpy()
            warp_depth = projected_tar_depths[0, 0].cpu().numpy()
            return out_img, warp_mask, warp_depth
            
        src_img = self.original_image # Torch tensor
        scaled_depth = self.depth # Torch tensor
        src_R = self.src_R
        src_T = self.src_T
        trg_R = self.R
        trg_T = self.T
        K = self.K
        uid = (self.uid)
        # format .2f
        uid = self.uid
        print(f'Warping {self.src_uid} to {uid: .2f}' )
        warped_img, warped_mask, warped_depth = depth_warping_pt3d(src_img, scaled_depth, K, src_R, src_T, trg_R, trg_T)

        # dialte depth mask
        edge_mask = edge_filter(warped_depth, valid_mask=warped_mask, times=0.1)
        warped_depth[edge_mask] = 0.0
        warped_mask = (warped_depth > 0.01) & warped_mask
        
        warped_mask = warped_mask.astype(np.double)
        warped_img = torch.from_numpy(warped_img).to(torch.float32).to('cuda').detach()
        warped_mask = torch.from_numpy(warped_mask).to(torch.float32).to('cuda').detach()
        warped_depth = torch.from_numpy(warped_depth).to(torch.float32).to('cuda').detach()
        self.original_image = warped_img
        self.warp_mask = warped_mask
        self.depth = warped_depth
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.subplot(3,1,1)
        # plt.imshow(warped_img.clamp(0,1).detach().cpu().numpy().transpose(1,2,0))
        # plt.subplot(3,1,2)
        # plt.imshow(warped_depth.detach().cpu().numpy())
        # plt.subplot(3,1,3)
        # plt.imshow(warped_mask.detach().cpu().numpy())
        # plt.savefig(f"warp_{uid:.2f}.png", bbox_inches='tight', dpi=1000)
        # plt.close()

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
