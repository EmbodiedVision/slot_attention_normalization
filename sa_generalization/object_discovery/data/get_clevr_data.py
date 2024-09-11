"""
Copyright 2024 Max-Planck-Gesellschaft
Code author: Markus Krimmel
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen
This source code is licensed under the license found in the
LICENSE.mpi.md file in the root directory of this source tree.
"""

import json
import os
import pickle
import tempfile
import zipfile
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from sa_generalization.object_discovery import DATA_DIR
from sa_generalization.utils import download_with_pbar

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

MAX_OBJECTS = 10


def install_original_clevr():
    clevr_path = DATA_DIR.joinpath("CLEVR_v1.0")
    if not os.path.exists(clevr_path):
        url = "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip"
        with tempfile.TemporaryDirectory() as temp_dir:
            print("Downloading Data...")
            zip_path = Path(temp_dir).joinpath("download.zip")
            download_with_pbar(url, zip_path)
            print("Extracting Data...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(DATA_DIR)

    """
    images_for_max_objects_{subset} takes an integer k in [1, max_objects] and gives a list
    of all images in {subset} that contain k or fewer objects
    """

    transformed_data_path = clevr_path.joinpath("transformed_data.hdf5")
    if os.path.exists(transformed_data_path):
        print("Dataset already processed...")
        return

    g = h5py.File(transformed_data_path, "w")
    transform = transforms.Compose(
        [transforms.CenterCrop(192), transforms.Resize((128, 128))]
    )

    subset_names = ["train", "val"]
    images_for_max_objects = {
        subset_name: {i: [] for i in range(1, MAX_OBJECTS + 1)}
        for subset_name in subset_names
    }

    for subset in subset_names:
        print(f"Creating image registry for {subset}...")
        with open(clevr_path.joinpath(f"scenes/CLEVR_{subset}_scenes.json"), "r") as f:
            metadata = json.load(f)

        group = g.create_group(subset)
        image_data = group.create_dataset(
            "image", (len(metadata["scenes"]), 128, 128, 3), dtype="uint8"
        )

        for global_idx, scene in enumerate(tqdm(metadata["scenes"])):
            num_objects = len(scene["objects"])
            assert num_objects <= MAX_OBJECTS
            for n in range(num_objects, MAX_OBJECTS + 1):
                images_for_max_objects[subset][n].append(
                    {"filename": scene["image_filename"], "global_index": global_idx}
                )

            image = Image.open(
                clevr_path.joinpath(f"images/{subset}").joinpath(
                    scene["image_filename"]
                )
            )
            image = image.convert("RGB")
            image = transform(image)
            image_data[global_idx] = np.array(image)

    with open(clevr_path.joinpath("images_for_max_objects.pkl"), "wb") as f:
        pickle.dump(images_for_max_objects, f)

    g.close()


def install_iodine_clevr():
    iodine_dir = DATA_DIR.joinpath("IODINE")
    os.makedirs(iodine_dir, exist_ok=True)
    tf_path = iodine_dir.joinpath("clevr_with_masks_train.tfrecords")

    if not os.path.exists(tf_path):
        url = "https://storage.googleapis.com/multi-object-datasets/clevr_with_masks/clevr_with_masks_train.tfrecords"
        download_with_pbar(url, tf_path)

    COMPRESSION_TYPE = tf.io.TFRecordOptions.get_compression_type_string("GZIP")
    IMAGE_SIZE = [240, 320]
    MAX_NUM_ENTITIES = 11
    BYTE_FEATURES = ["mask", "image", "color", "material", "shape", "size"]

    features = {
        "image": tf.io.FixedLenFeature(IMAGE_SIZE + [3], tf.string),
        "mask": tf.io.FixedLenFeature([MAX_NUM_ENTITIES] + IMAGE_SIZE + [1], tf.string),
        "x": tf.io.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
        "y": tf.io.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
        "z": tf.io.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
        "pixel_coords": tf.io.FixedLenFeature([MAX_NUM_ENTITIES, 3], tf.float32),
        "rotation": tf.io.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
        "size": tf.io.FixedLenFeature([MAX_NUM_ENTITIES], tf.string),
        "material": tf.io.FixedLenFeature([MAX_NUM_ENTITIES], tf.string),
        "shape": tf.io.FixedLenFeature([MAX_NUM_ENTITIES], tf.string),
        "color": tf.io.FixedLenFeature([MAX_NUM_ENTITIES], tf.string),
        "visibility": tf.io.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
    }

    def _decode(example_proto):
        # Parse the input `tf.Example` proto using the feature description dict above.
        single_example = tf.io.parse_single_example(example_proto, features)
        for k in BYTE_FEATURES:
            single_example[k] = tf.squeeze(
                tf.io.decode_raw(single_example[k], tf.uint8), axis=-1
            )
        return single_example

    raw_dataset = tf.data.TFRecordDataset(tf_path, compression_type=COMPRESSION_TYPE)
    dset = raw_dataset.map(_decode)
    dset_len = 100_000
    subset_lengths = OrderedDict([("train", 70_000), ("val", 15_000), ("test", 15_000)])

    current_start_index = 0
    start_indices = OrderedDict()
    for subset_name, subset_length in subset_lengths.items():
        start_indices[subset_name] = current_start_index
        current_start_index += subset_length

    transformed_data_path = iodine_dir.joinpath("transformed_data.hdf5")
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
        "image_wide": {
            "dtype": "uint8",
            "shape": (128, 128, 3),
            "transform": transforms.Compose(
                [
                    lambda x: torch.Tensor(x["image"].numpy()).permute(2, 0, 1),
                    transforms.CenterCrop(240),
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
        "mask_wide": {
            "dtype": "bool",
            "shape": (128, 128, MAX_NUM_ENTITIES),
            "transform": transforms.Compose(
                [
                    lambda x: torch.squeeze(torch.Tensor(x["mask"].numpy())),
                    transforms.CenterCrop(240),
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
    }
    for subset_name, subset_length in subset_lengths.items():
        group = g.create_group(subset_name)
        for feature, description in relevant_features.items():
            group.create_dataset(
                feature,
                (subset_length, *description["shape"]),
                dtype=description["dtype"],
            )

    print("Processing IODINE...")
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

    with open(iodine_dir.joinpath("images_for_max_objects.pkl"), "wb") as f:
        pickle.dump(images_for_max_objects, f)


if __name__ == "__main__":
    # print("Getting original data...")
    # install_original_clevr()
    # print()
    print("Getting IODINE data...")
    install_iodine_clevr()
