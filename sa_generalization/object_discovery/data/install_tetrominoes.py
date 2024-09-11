"""
Copyright 2024 Max-Planck-Gesellschaft
Code author: Markus Krimmel
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen
This source code is licensed under the license found in the
LICENSE.mpi.md file in the root directory of this source tree.
"""

import os
import pickle
from collections import OrderedDict

import h5py
import numpy as np
import tensorflow as tf
from torchvision import transforms
from tqdm import tqdm

from sa_generalization.object_discovery import DATA_DIR
from sa_generalization.object_discovery.data.get_clevr_data import download_with_pbar

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

MAX_OBJECTS = 3


def install_tetrominoes():
    tetrominoes_dir = DATA_DIR.joinpath("TETROMINOES")
    os.makedirs(tetrominoes_dir, exist_ok=True)
    tf_path = tetrominoes_dir.joinpath("tetrominoes_train.tfrecords")

    if not os.path.exists(tf_path):
        url = "https://storage.googleapis.com/multi-object-datasets/tetrominoes/tetrominoes_train.tfrecords"
        download_with_pbar(url, tf_path)

    COMPRESSION_TYPE = tf.io.TFRecordOptions.get_compression_type_string("GZIP")
    IMAGE_SIZE = [35, 35]
    MAX_NUM_ENTITIES = 4
    BYTE_FEATURES = ["mask", "image"]

    features = {
        "image": tf.io.FixedLenFeature(IMAGE_SIZE + [3], tf.string),
        "mask": tf.io.FixedLenFeature([MAX_NUM_ENTITIES] + IMAGE_SIZE + [1], tf.string),
        "x": tf.io.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
        "y": tf.io.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
        "shape": tf.io.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
        "color": tf.io.FixedLenFeature([MAX_NUM_ENTITIES, 3], tf.float32),
        "visibility": tf.io.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
    }

    def _decode(example_proto):
        single_example = tf.io.parse_single_example(example_proto, features)
        for k in BYTE_FEATURES:
            single_example[k] = tf.squeeze(
                tf.io.decode_raw(single_example[k], tf.uint8), axis=-1
            )
        return single_example

    raw_dataset = tf.data.TFRecordDataset(tf_path, compression_type=COMPRESSION_TYPE)
    dset = raw_dataset.map(_decode)

    dset_len = 1_000_000
    subset_lengths = OrderedDict(
        [("train", 700_000), ("val", 150_000), ("test", 150_000)]
    )

    current_start_index = 0
    start_indices = OrderedDict()
    for subset_name, subset_length in subset_lengths.items():
        start_indices[subset_name] = current_start_index
        current_start_index += subset_length

    transformed_data_path = tetrominoes_dir.joinpath("transformed_data.hdf5")
    if os.path.exists(transformed_data_path):
        print("Dataset already processed...")
        return

    g = h5py.File(transformed_data_path, "w")
    relevant_features = {
        "image": {
            "dtype": "uint8",
            "shape": (35, 35, 3),
            "transform": lambda x: x["image"].numpy().astype("uint8"),
            "assert": lambda x: True,
        },
        "mask": {
            "dtype": "bool",
            "shape": (35, 35, MAX_NUM_ENTITIES),
            "transform": transforms.Compose(
                [
                    lambda x: np.squeeze(x["mask"].numpy()),
                    lambda x: np.transpose(x, (1, 2, 0)),
                    lambda x: x.astype("bool"),
                ]
            ),
            "assert": lambda x: True,
        },
        "n_objects": {
            "dtype": "uint8",
            "shape": (),
            "transform": lambda x: np.count_nonzero(x["visibility"].numpy()[1:]),
            "assert": lambda x: np.count_nonzero(x["visibility"].numpy()[1:]) == 3,
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

    print("Processing TETROMINOES...")
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

    with open(tetrominoes_dir.joinpath("images_for_max_objects.pkl"), "wb") as f:
        pickle.dump(images_for_max_objects, f)


def test_tetrominoes():
    import matplotlib.pyplot as plt

    with h5py.File(DATA_DIR.joinpath("TETROMINOES", "transformed_data.hdf5"), "r") as g:
        fig, ax = plt.subplots(nrows=1, ncols=5)
        ax[0].imshow(g["train"]["image"][-1])
        for i in range(4):
            ax[1 + i].imshow(g["train"]["mask"][-1, ..., i])
    plt.savefig("tetrominoes.png")


if __name__ == "__main__":
    install_tetrominoes()
    test_tetrominoes()
