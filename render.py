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
from typing import List

import torch
from scene import Scene
from scene.cameras import Camera
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state, normalize
from utils.traj_ops import gs_interpolate_trajectory
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import matplotlib.pyplot as plt
import cv2

import imageio    
from PIL import Image

def render_set(model_path: str, name: str, iteration: int, views: List[Camera], gaussians: GaussianModel, pipeline: PipelineParams, background: torch.tensor):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    modes_path = os.path.join(model_path, name, "ours_{}".format(iteration), "modes")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(modes_path, exist_ok=True)

    # for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
    #     view: Camera
    #     results = render(view, gaussians, pipeline, background)
    #     rendering = results["render"]

    #     gt = view.original_image[0:3, :, :]
    #     depth = torch.exp(results["depth"] - results["depth"].min())
    #     alpha_depth = results["alpha_depth"]
  
    #     depth = (depth-depth.min())/((depth.max()-depth.min()+ 1e-5))
    #     depth = torch.clip(depth.clone(), 0, 1).squeeze(0).detach().cpu().numpy()


    #     alpha_depth = results["alpha_depth"]
    #     alpha_depth = (alpha_depth-alpha_depth.min())/((alpha_depth.max()-alpha_depth.min()+ 1e-5))
    #     alpha_depth = torch.clip(alpha_depth.clone(), 0, 1).squeeze(0).detach().cpu().numpy()


    #     modes = normalize(results["modes"])
    #     torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name + ".png"))
    #     torchvision.utils.save_image(gt, os.path.join(gts_path, view.image_name + ".png"))

    #     plt.imsave(os.path.join(depth_path, view.image_name + ".png"), depth, cmap='jet')  

    #     cv2.imwrite(os.path.join(modes_path, view.image_name + ".png"), (modes.detach().cpu().numpy().squeeze() * 65535).astype(np.uint16))

    # render on interpolated trajectory
    rendered_rgbs, rendered_depths = [], []
    interp_views = gs_interpolate_trajectory(views, nframes_interval=20)
    for idx, view in enumerate(tqdm(interp_views, desc="Rendering progress")):
        view: Camera
        results = render(view, gaussians, pipeline, background)
        rendering = results["render"]

        depth = torch.exp(results["depth"] - results["depth"].min())
        alpha_depth = results["alpha_depth"]
  
        depth = (depth-depth.min())/((depth.max()-depth.min()+ 1e-5))
        depth = torch.clip(depth.clone(), 0, 1).squeeze(0).detach().cpu().numpy()


        alpha_depth = results["alpha_depth"]
        alpha_depth = (alpha_depth-alpha_depth.min())/((alpha_depth.max()-alpha_depth.min()+ 1e-5))
        alpha_depth = torch.clip(alpha_depth.clone(), 0, 1).squeeze(0).detach().cpu().numpy()


        modes = normalize(results["modes"])
        torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name + ".png"))
        rendered_rgbs.append(((rendering.clamp(0., 1.).permute(1,2,0).detach().cpu().numpy()) * 255).astype(np.uint8))

        plt.imsave(os.path.join(depth_path, view.image_name + ".png"), depth, cmap='jet') 
        depth_image = np.array(Image.open(os.path.join(depth_path, view.image_name + ".png")))
        rendered_depths.append(depth_image) 

        cv2.imwrite(os.path.join(modes_path, view.image_name + ".png"), (modes.detach().cpu().numpy().squeeze() * 65535).astype(np.uint16))
    
    imageio.mimwrite(f"{render_path}/video_rgb.mp4", rendered_rgbs, fps=20)
    imageio.mimwrite(f"{render_path}/video_dpt.mp4", rendered_depths, fps=20)

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, mode='eval')

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # if not skip_train:
        #      render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        # if not skip_test:
        #      render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

        render_set(model_path=dataset.model_path, name="renders", iteration=scene.loaded_iter, views=scene.getTrainCameras(), gaussians=gaussians, pipeline=pipeline, background=background)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)