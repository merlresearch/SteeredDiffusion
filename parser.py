# Copyright (C) 2023-2024 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (C) 2022-2023 Omri Avrahami

# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: MIT

# Code adapted  from https://github.com/omriav/blended-diffusion -- MIT License

import argparse


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-config", "--config", type=str, help="Config file with generations", default="configs/diffusion_config.yml"
    )
    parser.add_argument(
        "-img_path", "--img_path", type=str, help="Path of example image", default="./input_example/faces/4.jpg"
    )
    parser.add_argument(
        "-mask_path", "--mask_path", type=str, help="Path of example mask", default="./input_example/masks/4.png"
    )
    parser.add_argument("-data_fold", "--data_fold", type=str, help="Path of data fold", default="./data")
    parser.add_argument("-condition", "--condition", type=str, help="Required condition", default="grayscale")
    parser.add_argument(
        "-editing_text",
        "--editing_text",
        type=str,
        help="Required text for editing",
        default="A woman with blonde hair",
    )
    args = parser.parse_args()
    return args
