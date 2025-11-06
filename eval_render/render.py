import sys
import os
sys.path.append(os.getcwd())

import torch
from scene import Scene
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from argparse import ArgumentParser
from gaussian_renderer import GaussianModel
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from utils.semantic_utils import colorize
from omegaconf import OmegaConf
import flow_vis_torch
from utils.cmap import color_depth_map
from imageio.v2 import imwrite

def to4x4(R, T):
    RT = np.eye(4,4)
    RT[:3, :3] = R
    RT[:3, 3] = T
    return RT

def apply_colormap(image, cmap="viridis"):
    colormap = cm.get_cmap(cmap)
    colormap = torch.tensor(colormap.colors).to(image.device)  # type: ignore
    image_long = (image * 255).long()
    image_long_min = torch.min(image_long)
    image_long_max = torch.max(image_long)
    assert image_long_min >= 0, f"the min value is {image_long_min}"
    assert image_long_max <= 255, f"the max value is {image_long_max}"
    return colormap[image_long[0, ...]].permute(2, 0, 1)


def apply_depth_colormap(depth, near_plane=None, far_plane=None, cmap="turbo"):
    near_plane = near_plane or float(torch.min(depth))
    far_plane = far_plane or float(torch.max(depth))
    depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
    depth = torch.clip(depth, 0, 1)

    colored_image = apply_colormap(depth, cmap=cmap)
    return colored_image


def save(image_name, results, paths):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(1.408, 0.376)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow((results['error_map'].detach().cpu().numpy().transpose(1,2,0)), cmap='jet')
    plt.savefig(os.path.join(paths['error_map'], image_name + ".png"), dpi=1000)
    plt.close('all')

    torchvision.utils.save_image(results['render'], os.path.join(paths['render'], image_name + ".png"))
    torchvision.utils.save_image(results['gt'], os.path.join(paths['gt'], image_name + ".png"))
    results['semantic'].save(os.path.join(paths['semantic'], image_name + ".png"))
    imwrite(os.path.join(paths['depth'], image_name + ".png"), results['depth'])

    opticalflow = results['flow'].permute(1,2,0)
    opticalflow = opticalflow[..., :2]
    pytorch_optic_rgb = flow_vis_torch.flow_to_color(opticalflow.permute(2, 0, 1))  # (2, h, w)
    torchvision.utils.save_image(pytorch_optic_rgb.float(), os.path.join(paths['flow'], image_name + ".png"), normalize=True)


def render_set(name:str, scene:Scene, background:torch.Tensor):
    paths = {}
    for path_name in ['render', 'semantic', 'flow', 'gt', 'error_map', 'depth']:
        path = os.path.join(scene.model_path, name, "ours_{}".format(scene.loaded_iter), path_name)
        paths[path_name] = path
        makedirs(path, exist_ok=True)

    if name == 'train':
        views = scene.getTrainCameras()
    else:
        views = scene.getTestCameras()
    
    save_tasks = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if scene.data_type == 'kitti':
            gap = 2
        elif scene.data_type == 'kitti360':
            gap = 4
        elif scene.data_type == 'waymo':
            gap = 3
        elif scene.data_type == 'nuscenes' or scene.data_type == 'pandaset':
            gap = 6
            
        if idx - gap < 0:
            prev_view = views[0]
        else:
            prev_view = views[idx-gap]
            
        render_pkg = render(
            view, prev_view, scene.gaussians, scene.dynamic_gaussians, None, background, True
        )
        rendering = render_pkg['render'].detach().cpu()
        semantic = render_pkg['feats'].detach().cpu()
        semantic = torch.argmax(semantic, dim=0)
        semantic_rgb = colorize(semantic.detach().cpu().numpy())
        depth = render_pkg['depth']
        color_depth = color_depth_map(depth[0].detach().cpu().numpy())
        color_depth[semantic == 10] = np.array([255.0, 255.0, 255.0])
        gt = view.original_image[0:3, :, :].detach().cpu()
        error_map = torch.mean((rendering - gt) ** 2, dim=0)[None, ...]
        
        # save
        results = {
            "render": rendering,
            'semantic': semantic_rgb,
            'depth': color_depth,
            'gt': gt,
            'error_map': error_map,
            'flow': render_pkg["opticalflow"]
        }
        save(view.image_name, results, paths)

def render_sets(args):
    cfg = OmegaConf.load(os.path.join(args.model_path, 'cfg.yaml'))
    cfg.model_path = args.model_path
    with torch.no_grad():
        gaussians = GaussianModel(cfg.model.sh_degree, affine=cfg.affine)
        scene = Scene(cfg, gaussians, load_iteration=args.iteration, shuffle=False, data_type=cfg.data_type)

        bg_color = [1,1,1] if cfg.model.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not args.skip_train:
            render_set("train", scene, background)

        if not args.skip_test:
            render_set("test", scene, background)


if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--iteration", type=int, default=30_000)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    args = parser.parse_args()
    print("Rendering " + args.model_path)

    render_sets(args)