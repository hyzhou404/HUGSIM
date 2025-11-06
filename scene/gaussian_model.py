import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH, SH2RGB
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
import open3d as o3d
import tinycudann as tcnn
from math import sqrt
from scene.ground_model import GroundModel
from io import BytesIO

                
class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = torch.logit

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, feat_mutable=True, affine=False, ground_args=None):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._feats3D = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.feat_mutable = feat_mutable
        self.setup_functions()

        self.pos_enc = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={"otype": "Frequency", "n_frequencies": 2},
        )
        self.dir_enc = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 3,
            },
        )

        self.affine = affine
        if affine:
            self.appearance_model = tcnn.Network(
                n_input_dims=self.pos_enc.n_output_dims + self.dir_enc.n_output_dims,
                n_output_dims=12,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 32,
                    "n_hidden_layers": 2,
                }
            )
        else:
            self.appearance_model = None

        if ground_args:
            self.ground_model = GroundModel(sh_degree, model_args=ground_args, finetune=True)
        else:
            self.ground_model = None

    def capture(self):
        if self.ground_model is not None:
            ground_model_params = self.ground_model.capture()
        else:
            ground_model_params = None
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._feats3D,
            self._scaling,
            self._rotation,
            self._opacity,
            self.spatial_lr_scale,
            self.appearance_model.state_dict(),
            ground_model_params,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._feats3D,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.spatial_lr_scale,
        appearance_state_dict,
        ground_model_params,
        ) = model_args
        self.appearance_model.load_state_dict(appearance_state_dict, strict=False)
        if training_args is not None:
            self.training_setup(training_args)
        if ground_model_params is not None:
            self.ground_model = GroundModel(self.max_sh_degree, model_args=ground_model_params)
        
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
        
    @property
    def get_full_scaling(self):
        assert self.ground_model is not None
        return torch.cat([self.scaling_activation(self._scaling), self.ground_model.get_scaling])
        
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_full_rotation(self):
        assert self.ground_model is not None
        return torch.cat([self.rotation_activation(self._rotation), self.ground_model.get_rotation])
    
    @property
    def get_xyz(self):
        return self._xyz
        
    @property
    def get_full_xyz(self):
        assert self.ground_model is not None
        return torch.cat([self._xyz, self.ground_model.get_xyz])

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_full_features(self):
        assert self.ground_model is not None
        sh = torch.cat((self._features_dc, self._features_rest), dim=1)
        return torch.cat([sh, self.ground_model.get_features])
    
    @property
    def get_3D_features(self):
        return torch.softmax(self._feats3D, dim=-1)
        
    @property
    def get_full_3D_features(self):
        assert self.ground_model is not None
        return torch.cat([torch.softmax(self._feats3D, dim=-1), self.ground_model.get_3D_features])

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
        
    @property
    def get_full_opacity(self):
        assert self.ground_model is not None
        return torch.cat([self.opacity_activation(self._opacity), self.ground_model.get_opacity])
    
    # def get_covariance(self, scaling_modifier = 1):
    #     return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        # self.spatial_lr_scale = 1
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        if self.feat_mutable:
            feats3D = torch.rand(fused_color.shape[0], 20).float().cuda()
            self._feats3D = nn.Parameter(feats3D.requires_grad_(True))
        else:
            feats3D = torch.zeros(fused_color.shape[0], 20).float().cuda()
            feats3D[:, 13] = 1
            self._feats3D = feats3D

        print("Number of points at initialization : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        # self.spatial_lr_scale /= 3

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init*self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr*self.spatial_lr_scale, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
        ]

        if self.affine:
            l.append({'params': [*self.appearance_model.parameters()], 'lr': 1e-3, "name": "appearance_model"})
        
        if self.feat_mutable:
            l.append({'params': [self._feats3D], 'lr': 1e-2, "name": "feats3D"})

        if self.ground_model is not None:
            self.ground_optimizer = self.ground_model.optimizer
        else:
            self.ground_optimizer = None

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        for i in range(self._feats3D.shape[1]):
            l.append('semantic_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path=None):
        mkdir_p(os.path.dirname(path))

        if self.ground_model is not None:
            xyz = self.get_full_xyz.detach().cpu().numpy()
            normals = np.zeros_like(xyz)
            f_dc = torch.cat([self._features_dc, self.ground_model._features_dc]).detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            f_rest = torch.cat([self._features_rest, self.ground_model._features_rest]).detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            feats3D = torch.cat([self._feats3D, self.ground_model._feats3D]).detach().cpu().numpy()
            opacities = torch.cat([self._opacity, self.ground_model._opacity]).detach().cpu().numpy()
            scale = self.scaling_inverse_activation(self.get_full_scaling).detach().cpu().numpy()
            rotation = torch.cat([self._rotation, self.ground_model._rotation]).detach().cpu().numpy()
        else:
            xyz = self.get_xyz.detach().cpu().numpy()
            normals = np.zeros_like(xyz)
            f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            feats3D = self._feats3D.detach().cpu().numpy()
            opacities = self._opacity.detach().cpu().numpy()
            scale = self.scaling_inverse_activation(self.get_scaling).detach().cpu().numpy()
            rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, feats3D, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        plydata = PlyData([el])
        if path is not None:
            plydata.write(path)
        return plydata

    def save_splat(self, ply_path, splat_path):
        plydata = self.save_ply(ply_path)
        vert = plydata["vertex"]
        sorted_indices = np.argsort(
            -np.exp(vert["scale_0"] + vert["scale_1"] + vert["scale_2"])
            / (1 + np.exp(-vert["opacity"]))
        )
        buffer = BytesIO()
        for idx in sorted_indices:
            v = plydata["vertex"][idx]
            position = np.array([v["x"], v["y"], v["z"]], dtype=np.float32)
            scales = np.exp(
                np.array(
                    [v["scale_0"], v["scale_1"], v["scale_2"]],
                    dtype=np.float32,
                )
            )
            rot = np.array(
                [v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]],
                dtype=np.float32,
            )
            SH_C0 = 0.28209479177387814
            color = np.array(
                [
                    0.5 + SH_C0 * v["f_dc_0"],
                    0.5 + SH_C0 * v["f_dc_1"],
                    0.5 + SH_C0 * v["f_dc_2"],
                    1 / (1 + np.exp(-v["opacity"])),
                ]
            )
            buffer.write(position.tobytes())
            buffer.write(scales.tobytes())
            buffer.write((color * 255).clip(0, 255).astype(np.uint8).tobytes())
            buffer.write(
                ((rot / np.linalg.norm(rot)) * 128 + 128)
                .clip(0, 255)
                .astype(np.uint8)
                .tobytes()
            )
        with open(splat_path, "wb") as f:
            f.write(buffer.getvalue())

    def save_semantic_pcd(self, path):
        color_dict = {
            0: np.array([128, 64, 128]),  # Road
            1: np.array([244, 35, 232]),  # Sidewalk
            2: np.array([70, 70, 70]),  # Building
            3: np.array([102, 102, 156]),  # Wall
            4: np.array([190, 153, 153]),  # Fence
            5: np.array([153, 153, 153]),  # Pole
            6: np.array([250, 170, 30]),  # Traffic Light
            7: np.array([220, 220, 0]),  # Traffic Sign
            8: np.array([107, 142, 35]),  # Vegetation
            9: np.array([152, 251, 152]),  # Terrain
            10: np.array([0, 0, 0]),  # Black (trainId 10)
            11: np.array([70, 130, 180]),  # Sky
            12: np.array([220, 20, 60]),  # Person
            13: np.array([255, 0, 0]),  # Rider
            14: np.array([0, 0, 142]),  # Car
            15: np.array([0, 0, 70]),  # Truck
            16: np.array([0, 60, 100]),  # Bus
            17: np.array([0, 80, 100]),  # Train
            18: np.array([0, 0, 230]),  # Motorcycle
            19: np.array([119, 11, 32])  # Bicycle
        }
        semantic_idx = torch.argmax(self.get_full_3D_features, dim=-1, keepdim=True)
        opacities = self.get_full_opacity[:, 0]
        mask = ((semantic_idx != 10)[:, 0]) & ((semantic_idx != 11)[:, 0]) & (opacities > 0.2)

        semantic_idx = semantic_idx[mask]
        semantic_rgb = torch.zeros_like(semantic_idx).repeat(1, 3)
        for idx in range(20):
            rgb = torch.from_numpy(color_dict[idx]).to(semantic_rgb.device)[None, :]
            semantic_rgb[(semantic_idx == idx)[:, 0], :] = rgb
        semantic_rgb = semantic_rgb.float() / 255.0
        pcd_xyz = self.get_full_xyz[mask]
        smt_pcd = o3d.geometry.PointCloud()
        smt_pcd.points = o3d.utility.Vector3dVector(pcd_xyz.detach().cpu().numpy())
        smt_pcd.colors = o3d.utility.Vector3dVector(semantic_rgb.detach().cpu().numpy())
        o3d.io.write_point_cloud(path, smt_pcd)

    def save_vis_ply(self, path):
        mkdir_p(os.path.dirname(path))
        xyz = self.get_xyz.detach().cpu().numpy()
        if self.ground_model:
            xyz = np.concatenate([xyz, self.ground_model.get_xyz.detach().cpu().numpy()])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        colors = SH2RGB(self._features_dc[:, 0, :].detach().cpu().numpy()).clip(0, 1)
        if self.ground_model:
            ground_colors = SH2RGB(self.ground_model._features_dc[:, 0, :].detach().cpu().numpy()).clip(0, 1)
            colors = np.concatenate([colors, ground_colors])
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(path, pcd)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state
                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group['name'] == 'appearance_model':
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        if self.feat_mutable:
            self._feats3D = optimizable_tensors["feats3D"]
        else:
            self._feats3D = self._feats3D[1, :].repeat((self._xyz.shape[0], 1))
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group['name'] not in tensors_dict:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_feats3D, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "feats3D": new_feats3D,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        if self.feat_mutable:
            self._feats3D = optimizable_tensors["feats3D"]
        else:
            self._feats3D = self._feats3D[1, :].repeat((self._xyz.shape[0], 1))
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_feats3D = self._feats3D[selected_pts_mask].repeat(N,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_feats3D, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_feats3D = self._feats3D[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_feats3D, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, cam_pos=None):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            if cam_pos is not None:
                # points_cam_dist = torch.abs(self.get_xyz[:, None, :] - cam_pos[None, ...])
                # points_cam_nearest_idx = torch.argmin(torch.norm(points_cam_dist, dim=-1), dim=1)
                # points_cam_dist = points_cam_dist[torch.arange(points_cam_dist.shape[0]), points_cam_nearest_idx, :]
                # near_mask1 = (points_cam_dist[:, 1] < 5) & (points_cam_dist[:, 0] < 10) & (points_cam_dist[:, 2] < 10)
                # big_points_ws1 = near_mask1 & (self.get_scaling.max(dim=1).values > 1.0)
                # near_mask2 = (points_cam_dist[:, 1] < 10) & (points_cam_dist[:, 0] < 20) & (points_cam_dist[:, 2] < 20)
                # big_points_ws2 = near_mask2 & (self.get_scaling.max(dim=1).values > 3.0)
                # big_points_ws = (self.get_scaling.max(dim=1).values > 10.0) | big_points_ws1 | big_points_ws2 
                big_points_ws = self.get_scaling.max(dim=1).values > 10
                prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
            else:
                big_points_ws = self.get_scaling.max(dim=1).values > 5
                prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()
    
    def add_densification_stats_grad(self, tensor_grad, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(tensor_grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
        