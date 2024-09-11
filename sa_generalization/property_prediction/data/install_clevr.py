"""
Copyright 2024 Max-Planck-Gesellschaft
Code author: Markus Krimmel
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen
This source code is licensed under the license found in the
LICENSE.mpi.md file in the root directory of this source tree.
"""

######## Boiler plate code ##########
import os
import pickle
from collections import OrderedDict

import h5py
import numpy as np
import torch
from multi_object_datasets.clevr_with_masks import *
from torchvision import transforms
from tqdm import tqdm

from sa_generalization.property_prediction import DATA_DIR
from sa_generalization.utils import download_with_pbar

MAX_OBJECTS = 10


def to_hdf5(data_dir):
    tf_path = os.path.join(data_dir, "clevr_with_masks_train.tfrecords")

    if not os.path.exists(tf_path):
        url = "https://storage.googleapis.com/multi-object-datasets/clevr_with_masks/clevr_with_masks_train.tfrecords"
        download_with_pbar(url, tf_path)

    dset = dataset(tf_path)
    dset_len = 100_000
    subset_lengths = OrderedDict([("train", 70_000), ("val", 15_000), ("test", 15_000)])

    current_start_index = 0
    start_indices = OrderedDict()
    for subset_name, subset_length in subset_lengths.items():
        start_indices[subset_name] = current_start_index
        current_start_index += subset_length

    transformed_data_path = data_dir.joinpath("transformed_data.hdf5")
    if os.path.exists(transformed_data_path):
        print("Dataset already processed...")
        return

    g = h5py.File(transformed_data_path, "w")
    relevant_features = {
        "image": {
            "dtype": "uint8",
            "shape": (128, 128, 3),
            "transform": transforms.Compose(
                [
                    lambda x: torch.Tensor(x["image"].numpy()).permute(2, 0, 1),
                    transforms.CenterCrop(192),
                    transforms.Resize(
                        (128, 128), interpolation=transforms.InterpolationMode.NEAREST
                    ),
                    lambda x: x.permute(1, 2, 0),
                    lambda x: x.numpy().astype("uint8"),
                ]
            ),
            "assert": lambda x: True,
        },
        "mask": {
            "dtype": "bool",
            "shape": (128, 128, MAX_NUM_ENTITIES),
            "transform": transforms.Compose(
                [
                    lambda x: torch.squeeze(torch.Tensor(x["mask"].numpy())),
                    transforms.CenterCrop(192),
                    transforms.Resize(
                        (128, 128), interpolation=transforms.InterpolationMode.NEAREST
                    ),
                    lambda x: x.permute(1, 2, 0),
                    lambda x: x.numpy().astype("bool"),
                ]
            ),
            "assert": lambda x: True,
        },
        "n_objects": {
            "dtype": "uint8",
            "shape": (),
            "transform": lambda x: np.count_nonzero(x["visibility"].numpy()[1:]),
            "assert": lambda x: np.count_nonzero(x["visibility"].numpy()[1:])
            == np.count_nonzero(x["size"].numpy()[1:]),
        },
        "color": {
            "dtype": "uint8",
            "shape": (MAX_NUM_ENTITIES,),
            "transform": lambda x: x["color"],
            "assert": lambda x: True,
        },
        "shape": {
            "dtype": "uint8",
            "shape": (MAX_NUM_ENTITIES,),
            "transform": lambda x: x["shape"],
            "assert": lambda x: True,
        },
        "size": {
            "dtype": "uint8",
            "shape": (MAX_NUM_ENTITIES,),
            "transform": lambda x: x["size"],
            "assert": lambda x: True,
        },
        "material": {
            "dtype": "uint8",
            "shape": (MAX_NUM_ENTITIES,),
            "transform": lambda x: x["material"],
            "assert": lambda x: True,
        },
        "visibility": {
            "dtype": "uint8",
            "shape": (MAX_NUM_ENTITIES,),
            "transform": lambda x: x["visibility"],
            "assert": lambda x: True,
        },
        "position": {
            "dtype": "float32",
            "shape": (MAX_NUM_ENTITIES, 3),
            "transform": lambda x: np.stack([x["x"], x["y"], x["z"]], axis=-1),
            "assert": lambda x: True,
        },
    }
    for subset_name, subset_length in subset_lengths.items():
        group = g.create_group(subset_name)
        for feature, description in relevant_features.items():
            group.create_dataset(
                feature,
                (subset_length, *description["shape"]),
                dtype=description["dtype"],
            )

    print("Processing CLEVR...")
    images_for_max_objects = {
        subset_name: {i: [] for i in range(1, MAX_OBJECTS + 1)}
        for subset_name in subset_lengths.keys()
    }
    for i, item in tqdm(enumerate(dset), total=dset_len):
        current_subset = [
            subset_name
            for subset_name, start_index in start_indices.items()
            if start_index <= i
        ][-1]
        current_features = {}
        for feature, description in relevant_features.items():
            current_features[feature] = description["transform"](item)
            g[current_subset][feature][
                i - start_indices[current_subset]
            ] = current_features[feature]
            assert description["assert"](item)

        for n in range(current_features["n_objects"], MAX_OBJECTS + 1):
            images_for_max_objects[current_subset][n].append(
                {"global_index": i - start_indices[current_subset]}
            )

    g.close()

    with open(data_dir.joinpath("images_for_max_objects.pkl"), "wb") as f:
        pickle.dump(images_for_max_objects, f)


if __name__ == "__main__":
    to_hdf5(DATA_DIR)
