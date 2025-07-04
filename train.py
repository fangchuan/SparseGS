import os
import sys
import uuid
from random import randint
from tqdm import tqdm
from argparse import ArgumentParser, Namespace

import torch
import numpy as np
import torch.nn.functional as F
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from guidance.sd_utils import StableDiffusion
from utils.loss_utils import l1_loss, ssim, local_pearson_loss, pearson_depth_loss, mask_l1_loss
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from scene.cameras import Camera
from utils.general_utils import safe_state, normalize
from utils.graphics_utils import depth_double_to_normal
from utils.image_utils import psnr
from utils.graphics_utils import getWorld2View2
from utils.typing import *
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.app_model import AppModel

os.environ['QT_QPA_PLATFORM']='offscreen'
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False



lpips_func = LearnedPerceptualImagePatchSimilarity(normalize=True).to('cuda')

def mask_L1_loss_appearance(image, gt_image, mask, gaussians: GaussianModel, view_idx, return_transformed_image=False):
    appearance_embedding = gaussians.get_apperance_embedding(view_idx)
    # center crop the image
    origH, origW = image.shape[1:]
    downsample_scale = 8
    H = origH // downsample_scale * downsample_scale
    W = origW // downsample_scale * downsample_scale
    left = origW // 2 - W // 2
    top = origH // 2 - H // 2
    crop_image = image[:, top:top+H, left:left+W]
    crop_gt_image = gt_image[:, top:top+H, left:left+W]
    
    # down sample the image
    crop_image_down = torch.nn.functional.interpolate(crop_image[None], size=(H//downsample_scale, W//downsample_scale), mode="bilinear", align_corners=True)[0]
    
    resize_app_emb = appearance_embedding[None, None, :].repeat(H//downsample_scale, W//downsample_scale, 1).permute(2, 0, 1)
    crop_image_down = torch.cat([crop_image_down, resize_app_emb], dim=0)[None]
    mapping_image = gaussians.appearance_network(crop_image_down)
    # print(f"mapping image shape: {mapping_image.shape}, crop_image shape: {crop_image.shape}")
    transformed_image = mapping_image * crop_image
    if not return_transformed_image:
        return mask_l1_loss(transformed_image, crop_gt_image, mask)
    else:
        transformed_image = torch.nn.functional.interpolate(transformed_image, size=(origH, origW), mode="bilinear", align_corners=True)[0]
        return transformed_image
        
# function L1_loss_appearance is fork from GOF https://github.com/autonomousvision/gaussian-opacity-fields/blob/main/train.py
def L1_loss_appearance(image, gt_image, gaussians: GaussianModel, view_idx, return_transformed_image=False):
    appearance_embedding = gaussians.get_apperance_embedding(view_idx)
    # center crop the image
    origH, origW = image.shape[1:]
    downsample_scale = 32
    H = origH // downsample_scale * downsample_scale
    W = origW // downsample_scale * downsample_scale
    left = origW // 2 - W // 2
    top = origH // 2 - H // 2
    crop_image = image[:, top:top+H, left:left+W]
    crop_gt_image = gt_image[:, top:top+H, left:left+W]
    
    # down sample the image
    crop_image_down = torch.nn.functional.interpolate(crop_image[None], size=(H//downsample_scale, W//downsample_scale), mode="bilinear", align_corners=True)[0]
    
    resize_app_emb = appearance_embedding[None, None, :].repeat(H//downsample_scale, W//downsample_scale, 1).permute(2, 0, 1)
    crop_image_down = torch.cat([crop_image_down, resize_app_emb], dim=0)[None]
    mapping_image = gaussians.appearance_network(crop_image_down)
    # print(f"mapping image shape: {mapping_image.shape}, crop_image shape: {crop_image.shape}")
    transformed_image = mapping_image * crop_image
    if not return_transformed_image:
        return l1_loss(transformed_image, crop_gt_image)
    else:
        transformed_image = torch.nn.functional.interpolate(transformed_image, size=(origH, origW), mode="bilinear", align_corners=True)[0]
        return transformed_image

    

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, step, max_cameras, prune_sched):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene =  Scene(dataset, gaussians, step=step, max_cameras=max_cameras)
    gaussians.training_setup(opt)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    if dataset.lambda_diffusion:
        guidance_sd = StableDiffusion(device="cuda")
        guidance_sd.get_text_embeds([""], [""])
        print(f"[INFO] loaded SD!")

    trainCameras: List[Camera] = scene.getTrainCameras().copy()
    if dataset.disable_filter3D:
        gaussians.reset_3D_filter()
    else:
        gaussians.compute_3D_filter(cameras=trainCameras)
    warppedCameras: List[Camera] = scene.getFtCameras().copy()
    gaussians.ft_cameras_setup(opt, num_cams=len(trainCameras) + len(warppedCameras))

    app_model = AppModel()
    app_model.train()
    app_model.cuda()
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        app_model.load_weights(scene.model_path)
        
    warp_cam_stack = None
    
    for iteration in range(first_iter, opt.iterations + 1):        
        '''    
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None
        '''
        
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_idxs = list(np.arange(len(viewpoint_stack)))
        rand = randint(0, len(viewpoint_stack)-1)
        viewpoint_cam: Camera = viewpoint_stack.pop(rand)
        viewpoint_idx = viewpoint_idxs.pop(rand)

        # optimize camera pose
        if opt.pose_opt and viewpoint_cam.uid != 0:
            # print(f"[INFO] Optimizing camera {viewpoint_cam.uid}")
            ori_worldtocam = torch.eye(4)
            ori_worldtocam[:3, :3] = torch.from_numpy(viewpoint_cam.R.T)
            ori_worldtocam[:3, 3] = torch.from_numpy(viewpoint_cam.T)
            ori_camtoworlds = torch.inverse(ori_worldtocam[None, ...]).to("cuda")
            ft_camtoworlds = gaussians.pose_adjust(ori_camtoworlds, torch.tensor([viewpoint_cam.uid], dtype=torch.long, device="cuda"))
            viewpoint_cam.update_view_transform(ft_camtoworlds[0])

        exposure_compensation = True
        if iteration > 1000 and exposure_compensation:
            gaussians.use_app = True
        
        pick_warp_cam = ((randint(1, 10) <= 8) and (dataset.lambda_warp_reg > 0) and iteration > (dataset.warp_reg_start_itr))
        if pick_warp_cam: # A warping cam is picked
            if not warp_cam_stack:
                warp_cam_stack = scene.getFtCameras().copy()
            warp_cam_idx = randint(0, len(warp_cam_stack)-1)
            warp_cam: Camera = warp_cam_stack.pop(warp_cam_idx)
            if opt.pose_opt:
                # print(f"[INFO] Optimizing camera {warp_cam.uid}")
                ori_worldtocam = torch.eye(4)
                ori_worldtocam[:3, :3] = torch.from_numpy(warp_cam.R.T)
                ori_worldtocam[:3, 3] = torch.from_numpy(warp_cam.T)
                ori_camtoworlds = torch.inverse(ori_worldtocam[None, ...]).to("cuda")
                ft_camtoworlds = gaussians.pose_adjust(ori_camtoworlds, torch.tensor([warp_cam.uid], dtype=torch.long, device="cuda"))
                warp_cam.update_view_transform(ft_camtoworlds[0])
            warp_render_pkg = render(warp_cam, gaussians, pipe, background, app_model=app_model)
            warpped_render_image, warp_viewspace_point_tensor, warp_visibility_filter, warp_radii, warp_render_depth = (
                                                                                                                    warp_render_pkg["render"], 
                                                                                                                    warp_render_pkg["viewspace_points"], 
                                                                                                                    warp_render_pkg["visibility_filter"], 
                                                                                                                    warp_render_pkg["radii"],
                                                                                                                    warp_render_pkg["expected_depth"])
            warpped_gt_image = warp_cam.original_image.cuda()
            warpped_mask = warp_cam.warp_mask


        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, kernel_size=0.0, require_coord=True, require_depth=True, app_model=app_model)
        rendered_image: torch.Tensor
        rendered_image, viewspace_point_tensor, visibility_filter, radii, depth = (
                                                                    render_pkg["render"], 
                                                                    render_pkg["viewspace_points"], 
                                                                    render_pkg["visibility_filter"], 
                                                                    render_pkg["radii"],
                                                                    render_pkg["expected_depth"])
        gt_image = viewpoint_cam.original_image.cuda()
        # if iteration == 1:
        #     # save rendered rgb annd depth
        #     import matplotlib.pyplot as plt
        #     plt.figure()
        #     plt.subplot(3,1,1)
        #     plt.imshow(gt_image.clamp(0,1).detach().cpu().numpy().transpose(1,2,0))
        #     plt.subplot(3,1,2)
        #     plt.imshow(rendered_image.detach().cpu().numpy().transpose(1,2,0))
        #     plt.subplot(3,1,3)
        #     plt.imshow(depth.squeeze().detach().cpu().numpy())
        #     plt.savefig(f"itr_{iteration}_{viewpoint_cam.uid:.2f}.png", bbox_inches='tight', dpi=1000)
        #     plt.close()

        # use_decoupled_appearance = True
        # if use_decoupled_appearance:
        #     Ll1_render = L1_loss_appearance(rendered_image, gt_image, gaussians, viewpoint_cam.uid)
        # else:
        #     Ll1_render = l1_loss(rendered_image, gt_image)
        ssim_loss = (1.0 - ssim(rendered_image, gt_image))
        if 'app_image' in render_pkg and ssim_loss < 0.5:
            app_image = render_pkg['app_image']
            Ll1_render = l1_loss(app_image, gt_image)
        else:
            Ll1_render = l1_loss(rendered_image, gt_image)
            
        reg_kick_on = iteration >= 30000
        if reg_kick_on:
            lambda_depth_normal = opt.lambda_depth_normal
            if True:
                rendered_expected_depth: torch.Tensor = render_pkg["expected_depth"]
                rendered_median_depth: torch.Tensor = render_pkg["median_depth"]
                rendered_normal: torch.Tensor = render_pkg["normal"]
                depth_middepth_normal = depth_double_to_normal(viewpoint_cam, rendered_expected_depth, rendered_median_depth)
            else:
                rendered_expected_coord: torch.Tensor = render_pkg["expected_coord"]
                rendered_median_coord: torch.Tensor = render_pkg["median_coord"]
                rendered_normal: torch.Tensor = render_pkg["normal"]
                depth_middepth_normal = point_double_to_normal(viewpoint_cam, rendered_expected_coord, rendered_median_coord)
            depth_ratio = 0.6
            normal_error_map = (1 - (rendered_normal.unsqueeze(0) * depth_middepth_normal).sum(dim=1))
            depth_normal_loss = (1-depth_ratio) * normal_error_map[0].mean() + depth_ratio * normal_error_map[1].mean()
        else:
            lambda_depth_normal = 0
            depth_normal_loss = torch.tensor([0],dtype=torch.float32,device="cuda")
            
        # compute lpips loss and weigth
        if iteration > opt.lpips_start_itr:
            lambda_lpips = 0.25
            lpiploss = lpips_func((rendered_image[None, ...].float()).clamp(0., 1.), gt_image[None, ...].float())
            # print(f"lpips loss: {lpiploss}")
        else:
            lambda_lpips = 0.0
            lpiploss = 0.0
        rgb_loss = (1.0 - opt.lambda_dssim - lambda_lpips) * Ll1_render + opt.lambda_dssim * (1.0 - ssim(rendered_image, gt_image)) + lambda_lpips * lpiploss

        loss = rgb_loss + lambda_depth_normal * depth_normal_loss
       
        diffusion_loss = None
        lp_loss = None
        reg_loss = None
        pearson_loss = None

        if opt.lambda_depth > 0 and (not pick_warp_cam):
            depth_mask = (viewpoint_cam.depth > 0.).float()
            disp = torch.where(depth.squeeze(0) > 0.0, 1.0 / depth.squeeze(0), torch.zeros_like(depth.squeeze(0)))
            disp_gt = torch.where(viewpoint_cam.depth > 0.0, 1.0 / viewpoint_cam.depth, torch.zeros_like(viewpoint_cam.depth))  # [1, M]
            depthloss = (F.l1_loss(disp, disp_gt, reduction="none") * depth_mask * scene.cameras_extent).mean()
            loss += opt.lambda_depth * depthloss
        
        # if dataset.lambda_local_pearson > 0 and (not pick_warp_cam):
        #     lp_loss = local_pearson_loss(depth.squeeze(0), warp_cam.depth, dataset.box_p, dataset.p_corr)
        #     loss += dataset.lambda_local_pearson * lp_loss
        
        if pick_warp_cam:
            ssim_loss = (1.0 - ssim(warpped_render_image, warpped_gt_image))
            if 'app_image' in warp_render_pkg and ssim_loss < 0.5:
                warp_app_image = warp_render_pkg['app_image']
                reg_Ll1 = mask_l1_loss(warp_app_image, warpped_gt_image, warpped_mask)
            else:
                reg_Ll1 = mask_l1_loss(warpped_render_image, warpped_gt_image, warpped_mask)
            # reg_Ll1 = mask_l1_loss(warpped_render_image, warpped_gt_image, warpped_mask)
            
            reg_loss = (1.0 - opt.lambda_dssim) * reg_Ll1 + opt.lambda_dssim * (1.0 - ssim(warpped_render_image, warpped_gt_image))
            loss += dataset.lambda_warp_reg * reg_loss
            if opt.lambda_depth > 0:
                warp_render_depth = warp_render_depth.squeeze(0)
                warp_render_disp = torch.where(warp_render_depth > 0.0, 1.0 / warp_render_depth, torch.zeros_like(warp_render_depth))
                warp_render_disp_gt = torch.where(warp_cam.depth > 0.0, 1.0 / warp_cam.depth, torch.zeros_like(warp_cam.depth))  # [1, M]
                warp_depthloss = (F.l1_loss(warp_render_disp, warp_render_disp_gt, reduction="none") * warpped_mask * scene.cameras_extent).mean()
                loss += opt.lambda_depth * warp_depthloss

        # lambda_opacity_reg = 0.01
        # if lambda_opacity_reg > 0:
        #     opa_reg_loss = gaussians.get_opacity.mean()
        #     print(f"opacity reg loss: {opa_reg_loss.item()}")
        #     loss += lambda_opacity_reg * opa_reg_loss
            
        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            losses = [lp_loss, pearson_loss, diffusion_loss, reg_loss]
            names = ["Local_Depth", "Global Depth", "Diffusion", "Warp Reg"]
            
            # generate warpped images and put it into the train set
            if (iteration in [dataset.warp_reg_start_itr]) and ((dataset.lambda_warp_reg > 0)):
                _warp_cam_stack = scene.getFtCameras()
                for _cam in _warp_cam_stack:
                    _cam.generate_warp_gt()

            if iteration % 10 == 0:
                postfix_dict = {"EMA Loss": f"{ema_loss_for_log:.{7}f}",
                                          "Total Loss": f"{loss.item():.{7}f}"}

                for l,n in zip(losses, names):
                    if l is not None:
                        postfix_dict[n] = f"{l:.{7}f}"
                progress_bar.set_postfix(postfix_dict)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            tr_dict = {names[i]: losses[i] for i in range(len(losses))}
            training_report(tb_writer, iteration, Ll1_render, loss, depth_normal_loss, l1_loss, tr_dict, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter and (not pick_warp_cam):
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, dataset.prune_min_opa, scene.cameras_extent, size_threshold)
                    if dataset.disable_filter3D:
                        gaussians.reset_3D_filter()
                    else:
                        gaussians.compute_3D_filter(cameras=trainCameras)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Densification for warp Cam
            if iteration < opt.densify_until_iter and (pick_warp_cam):
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[warp_visibility_filter] = torch.max(gaussians.max_radii2D[warp_visibility_filter], warp_radii[warp_visibility_filter])
                gaussians.add_densification_stats(warp_viewspace_point_tensor, warp_visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, dataset.prune_min_opa, scene.cameras_extent, size_threshold)
                    if dataset.disable_filter3D:
                        gaussians.reset_3D_filter()
                    else:
                        gaussians.compute_3D_filter(cameras=trainCameras)
                        
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                gaussians.pose_optimizer.step()
                gaussians.pose_optimizer.zero_grad(set_to_none=True)
                app_model.optimizer.step()
                app_model.optimizer.zero_grad(set_to_none=True)
            
                
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    app_model.save_weights(scene.model_path, opt.iterations)
    torch.cuda.empty_cache()
        
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        print("Tensorboard Found!")
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, normal_loss, l1_loss, tr_dict, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        for k,v in tr_dict.items():
            if v is not None:
                tb_writer.add_scalar('train_loss_patches/' + k, v.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/normal_loss', normal_loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        # if tb_writer:
        #     tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        #     tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1, 7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 15000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[7000, 15000, 30000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--max_cameras", type=int, default=None)
    parser.add_argument("--prune_sched", nargs="+", type=int, default=[])
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    dataset = lp.extract(args)
    
    print("Optimizing " + dataset.model_path)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(dataset=dataset, 
             opt=op.extract(args), 
             pipe=pp.extract(args), 
             testing_iterations=args.test_iterations, 
             saving_iterations=args.save_iterations, 
             checkpoint_iterations=args.checkpoint_iterations, 
             checkpoint=args.start_checkpoint, 
             debug_from=args.debug_from, 
             step=args.step, 
             max_cameras=args.max_cameras, 
             prune_sched=args.prune_sched)

    # All done
    print("\nTraining complete.")
