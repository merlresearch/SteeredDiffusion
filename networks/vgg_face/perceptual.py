# Copyright (C) 2023-2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import torch
import torch.nn as nn
import torch.nn.functional as F


def gray_resize_for_identity(out, size=128):
    # print(out.shape)
    out_gray = 0.2989 * out[:, 0, :, :] + 0.5870 * out[:, 1, :, :] + 0.1140 * out[:, 2, :, :]
    out_gray = out_gray.unsqueeze(1).repeat(1, 3, 1, 1)
    # out_gray = F.interpolate(out_gray, (size, size), mode='bilinear', align_corners=False)
    return out_gray


class Vgg16(torch.nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, X):
        X1 = gray_resize_for_identity(X)
        h = F.relu(self.conv1_1(X1))
        h = F.relu(self.conv1_2(h))
        relu1_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        relu2_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        relu3_3 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        relu4_3 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        relu5_3 = h

        return [relu1_2, relu2_2, relu3_3]


def load_vgg(checkpoint):
    vgg = Vgg16()
    state_dict_g = torch.load(checkpoint)
    new_state_dict_g = {}
    new_state_dict_g["conv1_1.weight"] = state_dict_g["0.weight"]
    new_state_dict_g["conv1_1.bias"] = state_dict_g["0.bias"]
    new_state_dict_g["conv1_2.weight"] = state_dict_g["2.weight"]
    new_state_dict_g["conv1_2.bias"] = state_dict_g["2.bias"]
    new_state_dict_g["conv2_1.weight"] = state_dict_g["5.weight"]
    new_state_dict_g["conv2_1.bias"] = state_dict_g["5.bias"]
    new_state_dict_g["conv2_2.weight"] = state_dict_g["7.weight"]
    new_state_dict_g["conv2_2.bias"] = state_dict_g["7.bias"]
    new_state_dict_g["conv3_1.weight"] = state_dict_g["10.weight"]
    new_state_dict_g["conv3_1.bias"] = state_dict_g["10.bias"]
    new_state_dict_g["conv3_2.weight"] = state_dict_g["12.weight"]
    new_state_dict_g["conv3_2.bias"] = state_dict_g["12.bias"]
    new_state_dict_g["conv3_3.weight"] = state_dict_g["14.weight"]
    new_state_dict_g["conv3_3.bias"] = state_dict_g["14.bias"]
    new_state_dict_g["conv4_1.weight"] = state_dict_g["17.weight"]
    new_state_dict_g["conv4_1.bias"] = state_dict_g["17.bias"]
    new_state_dict_g["conv4_2.weight"] = state_dict_g["19.weight"]
    new_state_dict_g["conv4_2.bias"] = state_dict_g["19.bias"]
    new_state_dict_g["conv4_3.weight"] = state_dict_g["21.weight"]
    new_state_dict_g["conv4_3.bias"] = state_dict_g["21.bias"]
    new_state_dict_g["conv5_1.weight"] = state_dict_g["24.weight"]
    new_state_dict_g["conv5_1.bias"] = state_dict_g["24.bias"]
    new_state_dict_g["conv5_2.weight"] = state_dict_g["26.weight"]
    new_state_dict_g["conv5_2.bias"] = state_dict_g["26.bias"]
    new_state_dict_g["conv5_3.weight"] = state_dict_g["28.weight"]
    new_state_dict_g["conv5_3.bias"] = state_dict_g["28.bias"]
    vgg.load_state_dict(new_state_dict_g)
    return vgg


class Total_VGGloss(nn.Module):
    def __init__(self, args):
        super(Total_VGGloss, self).__init__()
        self.args = args
        self.vgg_model = load_vgg(args["checkpoint"])
        self.vgg_model.cuda()
        self.vgg_model.eval()
        self.args = args

    def forward_network(self, pred_img, gt):
        loss = []
        pred_img_features = self.vgg_model(pred_img)
        gt_features = self.vgg_model(gt)
        for pred_img_feature, gt_feature in zip(pred_img_features, gt_features):
            loss.append(F.mse_loss(pred_img_feature, gt_feature))

        return loss  # sum(loss)/len(loss)

    def forward(self, pred_img, gt, t):
        loss = 0

        use_VGG = self.args["multiscale"]["use"] or self.args["singlescale"]["use"]
        if self.args["multiscale"]["use"]:
            min_range = self.args["multiscale"]["min_t"]
            max_range = self.args["multiscale"]["max_t"]
        if self.args["singlescale"]["use"]:
            min_range = self.args["singlescale"]["min_t"]
            max_range = self.args["singlescale"]["max_t"]

        if use_VGG:
            if t <= max_range and t >= min_range:
                loss_val = self.forward_network(pred_img, gt)
        else:
            return loss

        if t <= max_range and t >= min_range:
            if self.args["multiscale"]["use"]:
                loss_multi = sum(loss_val) / len(loss_val)
                loss = loss_multi * self.args["multiscale"]["lambda"]

            if self.args["singlescale"]["use"]:
                loss = loss + loss_val[-1] * self.args["singlescale"]["lambda"]

        return loss
