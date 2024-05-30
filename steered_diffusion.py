# Copyright (C) 2023-2024 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (C) 2022-2023 Omri Avrahami

# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: MIT

# Code adapted  from https://github.com/omriav/blended-diffusion -- MIT License

import os
from pathlib import Path

import lpips
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import tqdm
import yaml
from numpy import random
from PIL import Image, ImageOps
from torchvision.transforms import functional as TF
from tqdm import tqdm

from guided_diffusion.guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
from losses.ssim import SSIM

torch.autograd.set_detect_anomaly(True)

from losses.Full_loss import Full_loss


class ImageEditor:
    def __init__(self, args) -> None:
        self.args = args
        self.data = args["data"]
        self.params = args["params"]
        self.network = args["diffusion_network"]
        self.checkpoints = args["checkpoints"]
        self.Full_loss = Full_loss(args)
        count = 0
        out_path = os.path.join(self.params["results_dir"])
        if os.path.exists(out_path) == False:
            os.makedirs(out_path)
        self.data["output_path"] = out_path

        if self.args["seed"] is not None:
            torch.manual_seed(self.args["seed"])
            np.random.seed(self.args["seed"])
            random.seed(self.args["seed"])

        self.model_config = model_and_diffusion_defaults()

        self.model_config.update(self.network)
        gpu_id = self.args["gpu_id"]
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        self.model, self.diffusion = create_model_and_diffusion(**self.model_config)
        self.model.load_state_dict(torch.load(self.checkpoints["ffhq"]))
        self.model.requires_grad_(False).eval().to(self.device)
        for name, param in self.model.named_parameters():
            if "qkv" in name or "norm" in name or "proj" in name:
                param.requires_grad_()
        if self.model_config["use_fp16"]:
            self.model.convert_to_fp16()

        self.image_size = (self.params["image_size"], self.params["image_size"])

    def edit_image(self):

        if self.params["use_ddim"]:
            self.init_ddim = self.diffusion.ddim_reverse_sample_loop(
                self.model,
                self.init_image_transfer,
            )["sample"]

        batch = self.params["batch_size"]
        img = self.data["init_image"]
        init_image = Image.open(img).convert("RGB")
        init_image = init_image.resize(self.image_size, Image.BICUBIC)
        if self.params["cond"] == "inpaint":
            mask_image = Image.open(self.data["init_mask"]).convert("L")
            mask_image = mask_image.resize(self.image_size, Image.BICUBIC)

        init_image = TF.to_tensor(init_image).to(self.device).unsqueeze(0).mul(2).sub(1)
        self.init_image = init_image

        if self.params["cond"] == "inpaint":
            mask_image = TF.to_tensor(mask_image).to(self.device).unsqueeze(0)
            mask_image = mask_image.repeat(1, 3, 1, 1)
        else:
            mask_image = None
        image_name = img.split("/")[-1].strip(".jpg")
        model_kwargs = {
            "cond": self.params["cond"],
            "mask_image": mask_image,
            "init_image": self.init_image,
            "num_iters": 1,
            "factor": self.params["scale_factor"],
        }

        shape = (
            batch,
            3,
            self.model_config["image_size"],
            self.model_config["image_size"],
        )

        model_kwargs["dest_fold"] = os.path.join("./results", model_kwargs["cond"])

        samples = self.diffusion.conditional_sample_loop_progressive(
            model=self.model,
            shape=shape,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            noise=None,
            cond_fn=self.Full_loss,
            progress=True,
        )

        for count, sample in enumerate(samples):
            pred_image = sample["sample"]
            pred_image = pred_image.add(1).div(2).clamp(0, 1)
            degraded_image = sample["degraded"].add(1).div(2).clamp(0, 1)
            dest_fold = os.path.join(self.params["results_dir"], model_kwargs["cond"], image_name)
            if os.path.exists(dest_fold) == False:
                os.makedirs(dest_fold)
            for j in range(pred_image.shape[0]):
                degraded_pred = torch.cat([degraded_image[j], pred_image[j]], dim=2)
                pred_image_pil = TF.to_pil_image(degraded_pred)
                pred_path = os.path.join(dest_fold, str(j) + ".jpg")
                pred_image_pil.save(pred_path)
