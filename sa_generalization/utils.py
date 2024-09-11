"""
Copyright 2024 Max-Planck-Gesellschaft
Code author: Markus Krimmel
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen
This source code is licensed under the license found in the
LICENSE.mpi.md file in the root directory of this source tree.
"""

import requests
from tqdm import tqdm


# content_length is a hint if the header doesn't specify it
def download_with_pbar(url, target_path, content_length=None):
    chunkSize = 1024
    r = requests.get(url, stream=True)
    # Inspired by https://stackoverflow.com/a/42071418/5196836
    with open(target_path, "wb") as f:
        if "Content-Length" in r.headers:
            content_length = int(r.headers["Content-Length"])
        pbar = tqdm(total=content_length, unit_scale=True, unit="B")
        for chunk in r.iter_content(chunk_size=chunkSize):
            if chunk:
                pbar.update(len(chunk))
                f.write(chunk)
    pbar.close()
