# Copyright (C) 2023-2024 Mitsubishi Electric Research Laboratories (MERL)
# SPDX-License-Identifier: AGPL-3.0-or-later

import clip
import torch
import torch.nn as nn
import torchvision.transforms.functional as F1
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms


def d_clip_loss(x, y, use_cosine=True):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)

    if use_cosine:
        distance = 1 - (x @ y.t()).squeeze()
    else:
        distance = (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

    return distance


def find_cossim(x, y):
    sim_fun = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
    return sim_fun(x, y)


class Total_farlloss(nn.Module):
    def __init__(self, args):
        super(Total_farlloss, self).__init__()
        self.args = args
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = clip.load("ViT-B/16", device=device, jit=False)[0].eval().requires_grad_(False)
        self.model = self.model.to(device)
        farl_state = torch.load(
            self.args["checkpoint"]
        )  # you can download from https://github.com/FacePerceiver/FaRL#pre-trained-backbones
        self.model.load_state_dict(farl_state["state_dict"], strict=False)
        self.clip_normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
        )
        self.clip_size = self.model.visual.input_resolution

        self.model.eval()

    def forward_image(self, x_in):
        x_in = x_in.add(1).div(2)

        clip_loss = torch.tensor(0)
        clip_in = self.clip_normalize(x_in)
        clip_in = F1.resize(clip_in, [self.clip_size, self.clip_size])
        image_embeds = self.model.encode_image(clip_in).float()
        return image_embeds

    def forward(self, pred_img, t, gt=None, diffusion=None):
        loss = 0
        b = pred_img.shape[0]
        clip_loss = 0
        id_loss = 0
        if self.args["farlclip"]["use"]:
            if t <= self.args["farlclip"]["max_t"] and t >= self.args["farlclip"]["min_t"]:
                pred_image_embed = self.forward_image(pred_img)
                text_embed = self.model.encode_text(clip.tokenize(self.args["farlclip"]["prompt"]).to("cuda:0")).float()
                clip_loss_full = d_clip_loss(pred_image_embed, text_embed)
                clip_loss = clip_loss + clip_loss_full.mean()
                loss = loss + clip_loss * self.args["farlclip"]["lambda"]
        if self.args["farledit"]["use"]:
            if t <= self.args["farledit"]["max_t"] and t >= self.args["farledit"]["min_t"]:
                noise = torch.randn_like(gt)
                gt_noisy = gt
                pred_image_embed = self.forward_image(pred_img)
                gt_image_embed = self.forward_image(gt_noisy)

                text_embed = self.model.encode_text(clip.tokenize(self.args["farledit"]["prompt"]).to("cuda:0")).float()
                clip_loss_full = d_clip_loss(pred_image_embed, text_embed)
                clip_loss = clip_loss + clip_loss_full.mean()
                loss = loss + clip_loss * self.args["farledit"]["lambda"]
        if self.args["farlidentity"]["use"]:

            if t <= self.args["farlidentity"]["max_t"] and t >= self.args["farlidentity"]["min_t"]:
                pred_image_embed = self.forward_image(pred_img)
                gt_image_embed = self.forward_image(gt)
                id_loss_full = d_clip_loss(pred_image_embed, gt_image_embed)

                id_loss = id_loss + id_loss_full.mean()

                loss = loss + self.args["farlidentity"]["lambda"] * id_loss
        return loss
