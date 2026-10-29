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
from utils.general_utils import PILtoTorch
import cv2

def _u8_property(name):
    """A photo/mask tensor stored as uint8 and read back as float. Lossless: PILtoTorch makes them as k / 255 from 8-bit
    images, and k / 255 is recomputed exactly on read. 4x less memory with --data_device cpu (e.g. 2,518 photos at
    1600x1066 with masks: ~70 GB as float32, ~17 GB as uint8)."""
    key = "_u8_" + name
    def get(self):
        t = self.__dict__.get(key)
        return None if t is None else t.float() / 255.0
    def put(self, v):
        object.__setattr__(self, key, None if v is None else (v * 255.0).round().to(torch.uint8))
    return property(get, put)

class Camera(nn.Module):
    original_image = _u8_property("original_image")
    alpha_mask = _u8_property("alpha_mask")

    def __init__(self, resolution, colmap_id, R, T, FoVx, FoVy, depth_params, image, invdepthmap,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 train_test_exp = False, is_test_dataset = False, is_test_view = False
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        resized_image_rgb = PILtoTorch(image, resolution)
        gt_image = resized_image_rgb[:3, ...]
        if resized_image_rgb.shape[0] == 4:
            alpha_mask = resized_image_rgb[3:4, ...].clone()
        else: 
            alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...])

        if train_test_exp and is_test_view:     # edited before storing: the stored mask is read back as a copy
            if is_test_dataset:
                alpha_mask[..., :alpha_mask.shape[-1] // 2] = 0
            else:
                alpha_mask[..., alpha_mask.shape[-1] // 2:] = 0
        self.alpha_mask = alpha_mask.to(self.data_device)

        self.original_image = gt_image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        self.invdepthmap = None
        self.depth_reliable = False
        if invdepthmap is not None:
            self.invdepthmap = cv2.resize(invdepthmap, resolution)
            self.invdepthmap[self.invdepthmap < 0] = 0
            # no depth gradient where the photo is masked (alpha 0) or the prior has no depth (0)
            self.depth_mask = self.alpha_mask * torch.from_numpy((self.invdepthmap > 0)[None]).to(self.alpha_mask)
            self.depth_reliable = True

            if depth_params is not None:
                if depth_params["scale"] < 0.2 * depth_params["med_scale"] or depth_params["scale"] > 5 * depth_params["med_scale"]:
                    self.depth_reliable = False
                    self.depth_mask *= 0
                
                if depth_params["scale"] > 0:
                    self.invdepthmap = self.invdepthmap * depth_params["scale"] + depth_params["offset"]

            if self.invdepthmap.ndim != 2:
                self.invdepthmap = self.invdepthmap[..., 0]
            self.invdepthmap = torch.from_numpy(self.invdepthmap[None]).to(self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        
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

