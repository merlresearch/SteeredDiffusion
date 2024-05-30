# Copyright (C) 2023-2024 Mitsubishi Electric Research Laboratories (MERL)
# SPDX-License-Identifier: AGPL-3.0-or-later

import sys

import torch

sys.path.append("..")
device = "cuda"
import os

import cv2
import facer
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.nn.functional import cross_entropy
from torch.nn.modules.loss import _WeightedLoss
from torchvision.transforms import functional as TF


class LogNLLLoss(_WeightedLoss):
    __constants__ = ["weight", "reduction", "ignore_index"]

    def __init__(self, weight=None, size_average=None, reduce=None, reduction=None, ignore_index=-100):
        super(LogNLLLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, y_input, y_target):
        # y_input = torch.log(y_input + EPSILON)
        return cross_entropy(y_input, y_target, weight=self.weight, ignore_index=self.ignore_index)


class Total_Faceparseloss(nn.Module):
    def __init__(self, args):
        super(Total_Faceparseloss, self).__init__()
        device = "cuda"
        dtype = torch.cuda.FloatTensor
        self.args = args
        self.device = device
        self.save = False
        self.face_detector = facer.face_detector("retinaface/mobilenet", device=device)
        self.face_parser = facer.face_parser("farl/lapa/448", device=device)
        # self.face_detector.eval()
        # self.face_parser.eval()

        parse_loss_criterion = args["criterion"]

        if parse_loss_criterion == "LogNLLLoss":
            self.loss = LogNLLLoss()

        elif parse_loss_criterion == "nn.BCEWithLogitsLoss":
            self.loss = nn.BCEWithLogitsLoss()

    def detect_faces(self, det_input):
        det_input = (det_input + 1) / 2
        det_input = det_input.clip(0, 1)
        det_input = det_input * 255.0
        det_clone = torch.clone(det_input)
        det_faces = self.face_detector(det_clone)
        return det_faces

    def parse_faces(self, det_input, det_faces, setgt=False):
        det_input = (det_input + 1) / 2
        det_input = det_input.clip(0, 1)
        det_input = det_input * 255.0
        if setgt:
            with torch.inference_mode():
                parsefaces = self.face_parser(det_input, det_faces)
        else:
            parsefaces = self.face_parser(det_input, det_faces)

        parse_logits = parsefaces["seg"]["logits"]
        parsed_outputs = parse_logits.softmax(dim=1)  # nfaces x nclasses x h x w

        return parsed_outputs

    def forward(self, pred_img, gt_segment, t):
        gt_segment = torch.clone(gt_segment).detach()
        a, b, c, d = pred_img.shape

        loss = 0

        if self.args["face_segment_parse"]["use"]:
            det_faces = self.detect_faces(gt_segment)
            if t[0] <= self.args["face_segment_parse"]["max_t"] and t[0] >= self.args["face_segment_parse"]["min_t"]:
                inp_segment = pred_img
                gt_segment = torch.clone(gt_segment)
                parsed_input = self.parse_faces(inp_segment, det_faces)
                parsed_gt = self.parse_faces(gt_segment, det_faces, setgt=True)
                req_input = parsed_input
                req_gt = parsed_gt
                loss_entropy = self.args["face_segment_parse"]["lambda"] * self.loss(req_input, req_gt)
                loss = loss + loss_entropy

        return loss


if __name__ == "__main__":

    init_image_pil_transfer = Image.open("./18.jpg").convert("RGB")
    init_image_pil_transfer = init_image_pil_transfer.resize((256, 256), Image.BICUBIC)  # type: ignore
    init_image_transfer = TF.to_tensor(init_image_pil_transfer).cuda().unsqueeze(0).mul(2).sub(1)
    faceparser = parsefaceloss_faces()
    # img[img>0]=1
    faceparser(init_image_transfer, init_image_transfer)
