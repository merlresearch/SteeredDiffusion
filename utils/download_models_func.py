# Copyright (C) 2023-2024 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (C) 2022-2023 Nithin Gopalakrishnan Nair
# Copyright (C) 2021-2022 OpenAi


# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

# Code adapted from https://github.com/openai/glide-text2im/blob/main/glide_text2im/download.py -- MIT License
# Code adapted from https://github.com/Nithin-GK/UniteandConquer/blob/main/download_models.py -- Apache-2.0 license

import os
from functools import lru_cache
from typing import Dict, Optional

import requests
from filelock import FileLock
from tqdm.auto import tqdm


@lru_cache()
def default_cache_dir():
    return os.path.join(os.path.abspath(os.getcwd()), "checkpoints")


MODEL_PATHS = {
    "model_face": "https://www.dropbox.com/scl/fi/jcv8a178943o10ml02f3r/ffhq_10m.pt?rlkey=o3nl8gpbg24l49uk1z3xmdpv4&dl=1",
    "arcface": "https://www.dropbox.com/scl/fi/8yf5tw71xbdf6a7nyzg0a/arcface18.pth?rlkey=9qa4e4y1digdvmnt7huzxyjie&dl=1",
    "farl_clip": "https://www.dropbox.com/scl/fi/6xwjn5amuu2zyjpbaxz5q/FaRL-Base-Patch16-LAIONFace20M-ep64.pth?rlkey=jszbu9zbmq5euyj97xjdp4bnk&dl=1",
    "farl_parse": "https://www.dropbox.com/scl/fi/fa3mmuom0sagg7b6x61gb/face_parse.pth?rlkey=4c45rtoydue5xyb5bkg36iam8&dl=1",
    "vggface": "https://www.dropbox.com/scl/fi/se50l3z1iaafccxiksf1r/VGG_FACE.pth?rlkey=8pc9m8na7cxlfv2wdme7fecqn&dl=1",
    "imagenet_diffusion": "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt",
}


LOCAL_PATHS = {
    "model_face": "./checkpoints/ffhq_10m.pt",
    "farl_clip": "./checkpoints/FaRL-Base-Patch16-LAIONFace20M-ep64.pth",
    "farl_parse": "./checkpoints/face_parse.pth",
    "arcface": "./checkpoints/arcface18.pth",
    "vggface": "./checkpoints/VGG_FACE.pth",
    "imagenet_diffusion": "./checkpoints/diffusion256x256.pt",
}

if os.path.exists("./checkpoints") == False:
    os.mkdir("./checkpoints")
# taken from this StackOverflow answer: https://stackoverflow.com/a/39225039
import requests


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={"id": id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={"id": id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def fetch_file_cached(
    url: str, key: str, progress: bool = True, cache_dir: Optional[str] = None, chunk_size: int = 4096
) -> str:
    """
    Download the file at the given URL into a local file and return the path.
    If cache_dir is specified, it will be used to download the files.
    Otherwise, default_cache_dir() is used.
    """
    if cache_dir is None:
        cache_dir = default_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    local_path = LOCAL_PATHS[key]
    print(local_path)
    if os.path.exists(local_path):
        return LOCAL_PATHS[key]
    response = requests.get(url, stream=True)
    size = int(response.headers.get("content-length", "0"))
    with FileLock(local_path + ".lock"):
        if progress:
            pbar = tqdm(total=size, unit="iB", unit_scale=True)
        tmp_path = local_path + ".tmp"
        with open(tmp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size):
                if progress:
                    pbar.update(len(chunk))
                f.write(chunk)
        os.rename(tmp_path, local_path)
        if progress:
            pbar.close()
        return local_path


def download_files():
    for _ in MODEL_PATHS:
        model = fetch_file_cached(MODEL_PATHS[_], _)


if __name__ == "__main__":
    download_files()
