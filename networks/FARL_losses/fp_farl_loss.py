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

label_map = {
    "background": 0,
    "skin": 1,
    "left_eyebrow": 2,
    "right_eyebrow": 3,
    "left_eye": 4,
    "right_eye": 5,
    "nose": 6,
    "upper_lip": 7,
    "inner_mouth": 8,
    "lower_lip": 9,
    "hair": 10,
}


class LogNLLLoss(_WeightedLoss):
    __constants__ = ["weight", "reduction", "ignore_index"]

    def __init__(self, weight=None, size_average=None, reduce=None, reduction=None, ignore_index=-100):
        super(LogNLLLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, y_input, y_target):
        return cross_entropy(y_input, y_target, weight=self.weight, ignore_index=self.ignore_index)


class parsefacesegment_faces(torch.nn.Module):
    def __init__(self, criterion="nn.BCEWithLogitsLoss", label_idx="hair", save=False):
        super(parsefacesegment_faces, self).__init__()

        self.label_idx = label_map[label_idx]
        self.save = False
        self.face_detector = facer.face_detector("retinaface/mobilenet", device=device)
        self.face_parser = facer.face_parser("farl/lapa/448", device=device)
        self.face_detector.eval()
        self.face_parser.eval()

        if criterion == "LogNLLLoss":
            self.loss = LogNLLLoss()
        elif criterion == "nn.BCEWithLogitsLoss":
            self.loss = nn.BCEWithLogitsLoss()

    def forward(self, pred_img, gt, save_fold="./parsed/", thres=0.9):
        idx = self.label_idx
        pred_img = (pred_img + 1) / 2
        pred_img = pred_img
        pred_img = pred_img * 255.0
        gt = (gt + 1) / 2
        gt = gt.clip(0, 1)
        gt = gt * 255.0
        gt_clone = torch.clone(gt)
        gtfaces1 = self.face_detector(gt_clone)

        with torch.inference_mode():
            gtfaces = self.face_parser(gt, gtfaces1)

        gtseg_logits = gtfaces["seg"]["logits"]
        gtout = gtseg_logits.softmax(dim=1)  # nfaces x nclasses x h x w

        faces = self.face_parser(pred_img, gtfaces1)

        seg_logits = faces["seg"]["logits"]
        out = seg_logits.softmax(dim=1)  # nfaces x nclasses x h x w

        a, b, c, d = out.shape
        if self.save:
            for i in range(out.shape[1]):
                parsed = out[0, i, :, :]
                parsed = parsed.detach().cpu().numpy()
                parsed = np.uint8(parsed * 255.0)
                fold = save_fold
                if os.path.exists(fold) == False:
                    os.mkdir(fold)
                cv2.imwrite(fold + str(i) + ".png", parsed)

        parsing = out[:, :, :, :]
        gtparsing = gtout[:, :, :, :]
        loss = self.loss(gtparsing, parsing)
        return loss, parsing, gtparsing


class parsefaceloss_faces(torch.nn.Module):
    def __init__(self, criterion="nn.BCEWithLogitsLoss", label_idx="hair", save=True):
        super(parsefaceloss_faces, self).__init__()

        self.label_idx = label_map[label_idx]
        self.save = save
        self.face_detector = facer.face_detector("retinaface/mobilenet", device=device)
        self.face_parser = facer.face_parser("farl/lapa/448", device=device)
        self.face_detector.eval()
        self.face_parser.eval()

        if criterion == "LogNLLLoss":
            self.loss = LogNLLLoss()
        elif criterion == "nn.BCEWithLogitsLoss":
            self.loss = nn.BCEWithLogitsLoss()

    def forward(self, pred_img, gt, save_fold="./parsed/", thres=0.9):
        idx = self.label_idx
        pred_img = (pred_img + 1) / 2
        pred_img = pred_img
        pred_img = pred_img * 255.0
        gt = (gt + 1) / 2
        gt = gt.clip(0, 1)
        gt = gt * 255.0
        gt_clone = torch.clone(gt)
        gtfaces1 = self.face_detector(gt_clone)

        with torch.inference_mode():
            gtfaces = self.face_parser(gt, gtfaces1)

        gtseg_logits = gtfaces["seg"]["logits"]
        gtout = gtseg_logits.softmax(dim=1)  # nfaces x nclasses x h x w

        faces = self.face_parser(pred_img, gtfaces1)

        seg_logits = faces["seg"]["logits"]
        out = seg_logits.softmax(dim=1)  # nfaces x nclasses x h x w

        a, b, c, d = out.shape
        if self.save:
            for i in range(out.shape[1]):
                parsed = out[0, i, :, :]
                parsed = parsed.detach().cpu().numpy()
                parsed = np.uint8(parsed * 255.0)
                fold = save_fold
                if os.path.exists(fold) == False:
                    os.mkdir(fold)
                cv2.imwrite(fold + str(i) + ".png", parsed)

            gt_mask = gtout[0, idx].detach().cpu().numpy()
            gt_mask = np.uint8(gt_mask.clip(0, 1) * 255.0)
            cv2.imwrite(fold + "gt_mask.png", gt_mask)
            gt_img = gt[0].permute(1, 2, 0).detach().cpu().numpy()  # .clip(0,255.0)
            cv2.imwrite(fold + "gt_img.png", np.uint8(gt_img[:, :, ::-1]))
            gtparsing = gtout[:, idx, :, :]
            gtparsing = gtparsing.unsqueeze(1)
            masked_gt = gt * gtparsing
            masked_pred = pred_img * gtparsing
            gt_img = masked_gt[0].permute(1, 2, 0).detach().cpu().numpy().clip(0, 255.0)
            cv2.imwrite(fold + "gt_masked.png", gt_img[:, :, ::-1])
        parsing = out[:, idx, :, :]
        gtparsing = gtout[:, idx, :, :]
        loss = self.loss(gtparsing, parsing)
        parsing = parsing.unsqueeze(1)
        gtparsing = gtparsing.unsqueeze(1)
        return loss, parsing, gtparsing


class parsefaceloss(torch.nn.Module):
    def __init__(self, criterion="nn.BCEWithLogitsLoss"):
        super(parsefaceloss, self).__init__()
        self.face_detector = facer.face_detector("retinaface/mobilenet", device=device)
        self.face_parser = facer.face_parser("farl/lapa/448", device=device)

        if criterion == "LogNLLLoss":
            self.loss = LogNLLLoss()
        elif criterion == "nn.BCEWithLogitsLoss":
            self.loss = nn.BCEWithLogitsLoss()

    def forward(self, pred_img, gt, save=True, save_fold="./parsed/", thres=0.9):
        pred_img = (pred_img + 1) / 2
        pred_img = pred_img  # .clip(0,1)
        img = pred_img * 255.0
        with torch.inference_mode():
            faces = self.face_detector(img)

        with torch.inference_mode():
            faces = self.face_parser(img, faces)

        seg_logits = faces["seg"]["logits"]
        out = seg_logits.softmax(dim=1)  # nfaces x nclasses x h x w
        print(out.shape)
        a, b, c, d = out.shape
        if save:
            for i in range(out.shape[1]):
                parsed = out[0, i, :, :]
                parsed = parsed.detach().cpu().numpy()
                parsed = np.uint8(parsed * 255.0)
                fold = save_fold
                if os.path.exists(fold) == False:
                    os.mkdir(fold)
                cv2.imwrite(fold + str(i) + ".png", parsed)
                gt_mask = gt[0, 0].detach().cpu().numpy()
                gt_mask = np.uint8(gt_mask.clip(0, 1) * 255.0)
                cv2.imwrite(fold + "gt_mask.png", gt_mask)

        parsing = out[:, 10, :, :].view(a, 1, c, d)
        gt = gt.repeat(a, 1, 1, 1)
        loss = self.loss(parsing, gt)
        return loss


if __name__ == "__main__":

    init_image_pil_transfer = Image.open("./18.jpg").convert("RGB")
    init_image_pil_transfer = init_image_pil_transfer.resize((256, 256), Image.BICUBIC)  # type: ignore
    init_image_transfer = TF.to_tensor(init_image_pil_transfer).cuda().unsqueeze(0).mul(2).sub(1)
    faceparser = parsefaceloss_faces()
    faceparser(init_image_transfer, init_image_transfer)
