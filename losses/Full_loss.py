# Copyright (C) 2023-2024 Mitsubishi Electric Research Laboratories (MERL)
# SPDX-License-Identifier: AGPL-3.0-or-later


import lpips
import numpy as np
import torch
import torchvision.transforms.functional as F1
import yaml

from networks.face_parsing_losses.parse_losses import Total_Faceparseloss
from networks.FARL_losses.farl_losses import Total_farlloss
from networks.vgg_face.perceptual import Total_VGGloss as VGGfaceNetwork


class Full_loss(torch.nn.Module):
    def __init__(self, args):
        super(Full_loss, self).__init__()

        dtype = torch.cuda.FloatTensor
        self.args = args
        self.vggface_loss = VGGfaceNetwork(args["networks"]["VGGface"])
        self.parsingloss = Total_Faceparseloss(args["networks"]["Semantics"])
        self.farl_loss = Total_farlloss(args["networks"]["FARL"])

    def forward(self, pred_img, identity_img, t, diffusion=None):
        loss = 0
        loss = loss + self.vggface_loss(pred_img, identity_img, t)
        loss = loss + self.farl_loss(pred_img, t, identity_img, diffusion)
        loss = loss + self.parsingloss(pred_img, identity_img, t)

        return loss
