# Copyright (C) 2023-2024 Mitsubishi Electric Research Laboratories (MERL)

# SPDX-License-Identifier: AGPL-3.0-or-later


from parser import get_arguments

import yaml

from steered_diffusion import ImageEditor

if __name__ == "__main__":
    args_config = get_arguments()
    config = args_config.config

    args_yaml_file = open(config)
    args = yaml.load(args_yaml_file, Loader=yaml.FullLoader)
    args["data"]["init_image"] = args_config.img_path
    args["data"]["init_mask"] = args_config.mask_path
    args["data"]["data_fold"] = args_config.data_fold
    args["params"]["cond"] = args_config.condition
    if args_config.condition == "Semantics":
        args["networks"]["Semantics"]["face_segment_parse"]["use"] = True
    elif args_config.condition == "Identity":
        args["networks"]["FARL"]["farlidentity"]["use"] = True
    elif args_config.condition == "editing":
        args["networks"]["FARL"]["farledit"]["use"] = True
        args["networks"]["VGGface"]["multiscale"]["use"] = True
        args["networks"]["FARL"]["farledit"]["prompt"] = args_config.editing_text

    image_editor = ImageEditor(args)
    image_editor.edit_image()
