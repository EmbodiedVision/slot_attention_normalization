"""
Copyright 2024 Max-Planck-Gesellschaft
Code author: Markus Krimmel
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen
This source code is licensed under the license found in the
LICENSE.mpi.md file in the root directory of this source tree.
"""

import os
import subprocess
import tempfile
from itertools import chain

import h5py
import numpy as np
import tensorflow_datasets as tfds
import timm
import torch
import torchvision
from tqdm import tqdm

from sa_generalization.dinosaur import DATA_DIR


def _get_from_hierarchical_dict(dictionary, hierarchical_key):
    split_key = hierarchical_key.split(".")
    if len(split_key) == 1:
        return dictionary[hierarchical_key]
    return _get_from_hierarchical_dict(dictionary, ".".join(split_key[:-1]))[
        split_key[-1]
    ]


def to_hdf5(
    dset_name,
    data_to_save=(
        "video",
        "segmentations",
        "depth",
        "forward_flow",
        "backward_flow",
        "metadata.num_instances",
        "instances.image_positions",
    ),
):
    hdf5_path = DATA_DIR.joinpath(DATA_DIR, f"{dset_name}.hdf5")
    if os.path.exists(hdf5_path):
        print("Dataset already exists. May append...")

    f = h5py.File(hdf5_path, "a")
    ds, ds_info = tfds.load(
        dset_name, data_dir="gs://kubric-public/tfds", with_info=True
    )

    name_to_max_objects = {"movi_c": 10, "movi_d": 23}
    max_objects = name_to_max_objects[dset_name]

    for split in ["train", "test", "validation"]:
        ds_split = ds[split]
        length = len(ds_split)
        try:
            group = f.create_group(split)
        except ValueError:
            print(f"Group {split} already exists. May append...")
            group = f[split]

        to_save_for_group = []
        for item in data_to_save:
            if item.startswith("instances."):
                tfds_shape = _get_from_hierarchical_dict(
                    ds_split.element_spec, item
                ).shape.as_list()
                assert (
                    len(tfds_shape) == 3
                ), "Currently only supports frame-wise instance data..."
                assert (
                    tfds_shape.count(None) == 1
                ), f"Currently only supports instance data with one ragged axis (object axis), got {tfds_shape} for {item}"
                tfds_shape = [l if l is not None else max_objects for l in tfds_shape]
                tfds_shape = [
                    tfds_shape[1],
                    tfds_shape[0],
                    *tfds_shape[2:],
                ]  # Swap object and frame axes
                shape = (length, *tfds_shape)
            else:
                shape = (
                    length,
                    *tuple(
                        _get_from_hierarchical_dict(ds_split.element_spec, item).shape
                    ),
                )

            try:
                group.create_dataset(
                    item,
                    shape=shape,
                    dtype=_get_from_hierarchical_dict(
                        ds_split.element_spec, item
                    ).dtype.as_numpy_dtype,
                )
                print(f"Created dataset for {item} of shape {shape}")
                to_save_for_group.append(item)
            except ValueError:
                print(f"Dataset for {item} already exists. Not updating this")

        if len(to_save_for_group) == 0:
            continue
        np_iterator = iter(tfds.as_numpy(ds_split))
        for i, sequence in enumerate(tqdm(np_iterator, total=length)):
            for item in to_save_for_group:
                if item.startswith("instances."):
                    group[item][
                        i, :, : sequence["metadata"]["num_instances"]
                    ] = np.transpose(
                        _get_from_hierarchical_dict(sequence, item), (1, 0, 2)
                    )
                else:
                    group[item][i] = _get_from_hierarchical_dict(sequence, item)

    f.close()
    print("Done")


def precompute_features(dset_name, encoder_name, new_size=224):
    hdf5_path = DATA_DIR.joinpath(DATA_DIR, f"{dset_name}.hdf5")
    assert os.path.exists(hdf5_path)
    f = h5py.File(hdf5_path, "a")
    encoder = timm.create_model(encoder_name, pretrained=True)
    encoder.eval()
    data_config = timm.data.resolve_data_config(args=[], model=encoder)

    encoder = encoder.to("cuda")

    encoder_to_feature_info = {"vit_base_patch8_224_dino": (785, 768)}

    n_features, feature_dim = encoder_to_feature_info[encoder_name]
    resizer = torchvision.transforms.Resize(
        size=new_size, interpolation=torchvision.transforms.InterpolationMode.BICUBIC
    )

    for split in ["train", "test", "validation"]:
        video_dset = f[split]["video"]
        try:
            feature_dset = f[split].create_dataset(
                encoder_name,
                shape=(*video_dset.shape[:2], n_features, feature_dim),
                dtype=np.float16,
            )
        except ValueError:  # dset already exists
            feature_dset = f[split][encoder_name]

        with torch.no_grad():
            for i, sequence in tqdm(enumerate(video_dset), total=len(video_dset)):
                normed = (
                    (np.asarray(sequence) / 255) - np.asarray(data_config["mean"])
                ) / data_config["std"]
                normed = resizer(
                    torch.as_tensor(
                        np.transpose(normed, (0, 3, 1, 2)), dtype=torch.float
                    ).to("cuda")
                )
                features = encoder.forward_features(normed)
                feature_dset[i] = features.cpu().numpy().astype(np.float16)

    f.close()


class FailedCopy(Exception):
    pass


def copy_dataset_to_tmp(dset_name, dset_dir=None, timeout_h=3):
    dset_dir = DATA_DIR if dset_dir is None else dset_dir
    source_hdf5_path = os.path.join(dset_dir, f"{dset_name}.hdf5")
    tmp_dir = tempfile.TemporaryDirectory()
    code = subprocess.call(
        ["rsync", "-ah", "--progress", str(source_hdf5_path), tmp_dir.name],
        timeout=timeout_h * 3600,
    )
    if code != 0 and code != 28:
        raise FailedCopy()
    elif code == 28:
        raise Exception("No disk space left to copy to...")
    return tmp_dir


class MOViFrameDataSet(torch.utils.data.Dataset):
    def __init__(
        self,
        dset_name,
        dset_dir=None,
        split="train",
        feature_names=("video", 0),
        load_to_memory=True,
        copy_to_temp=False,
        copy_timeout_h=3,
        sub_sequence_length=1,
        n_whole_sequences=np.inf,
        seed=42,
        resize=224,
        max_objects=None,
    ):
        """
        :param dset_name: Name of dataset, e.g. "movi_c", "movi_e"
        :param dset_dir: Directory that contains hdf5 file. By default DATA_DIR
        :param split: "train", "validation", or "test"
        :param feature_names: Tuple of features that should be extracted, e.g. "video", or features of vit encoder
        :param load_to_memory: Whether to load the relevant data to memory
        :param copy_to_temp: Whether to copy the dataset to /tmp before accessing
        :param copy_timeout_h: Timeout in hours for copying operation, if applicable
        :param sequence_length: Length of sequences that should be extracted
        :param n_whole_sequences: Total number of full-length sequences that we look at
        :param seed:
        """
        n_whole_sequences = np.inf if n_whole_sequences == "inf" else n_whole_sequences
        dset_dir = DATA_DIR if dset_dir is None else dset_dir
        self.max_objects = max_objects

        if copy_to_temp:
            assert not load_to_memory

        self.split = split
        self.feature_names = feature_names
        if not copy_to_temp:
            hdf5_path = os.path.join(dset_dir, f"{dset_name}.hdf5")
            self.tmp_dir = None
        else:
            self.tmp_dir = copy_dataset_to_tmp(
                dset_name=dset_name, dset_dir=dset_dir, timeout_h=copy_timeout_h
            )
            hdf5_path = os.path.join(self.tmp_dir.name, f"{dset_name}.hdf5")

        self.f = h5py.File(hdf5_path, "r")
        self.whole_sequence_length = self.f[split][feature_names[0]].shape[
            1
        ]  # The length of video sequences
        self.n_whole_sequences = min(
            self.f[split][feature_names[0]].shape[0], n_whole_sequences
        )

        if load_to_memory:
            self.dset = {
                feature_name: self.f[split][feature_name][: self.n_whole_sequences]
                for feature_name in feature_names
                if feature_name != 0
            }
            self.f.close()
            self.f = None
        else:
            self.dset = self.f[split]

        self.rng = np.random.RandomState(seed)
        self.subsequence_length = sub_sequence_length

        self.resizer = (
            (lambda x: x)
            if resize is None
            else torchvision.transforms.Resize(
                size=resize,
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
            )
        )

    @property
    def subsequence_length(self):
        return self._subseqeunce_length

    @subsequence_length.setter
    def subsequence_length(self, subsequence_length):
        self._subseqeunce_length = subsequence_length
        self.subsequences_per_sequence = (
            self.whole_sequence_length // subsequence_length
        )
        # self.n_subsequences = self.n_whole_sequences * self.subsequences_per_sequence
        # self.permutation = self.rng.permutation(self.n_subsequences)
        self.permute()

    def permute(self):
        if self.max_objects is None:
            self.n_subsequences = (
                self.n_whole_sequences * self.subsequences_per_sequence
            )
            self.permutation = self.rng.permutation(self.n_subsequences)
        else:
            allowed_whole_sequences = [
                i
                for i in range(self.n_whole_sequences)
                if self.dset["metadata.num_instances"][i] <= self.max_objects
            ]
            self.n_subsequences = (
                len(allowed_whole_sequences) * self.subsequences_per_sequence
            )
            allowed_subsequences = list(
                chain(
                    *[
                        [
                            seq_idx * self.subsequences_per_sequence + k
                            for k in range(self.subsequences_per_sequence)
                        ]
                        for seq_idx in allowed_whole_sequences
                    ]
                )
            )
            self.permutation = self.rng.permutation(allowed_subsequences)

    def __getitem__(self, item):
        idx = self.permutation[item]
        result = []
        for feature in self.feature_names:
            whole_sequence_idx = idx // self.subsequences_per_sequence
            start_frame_idx = (
                idx % self.subsequences_per_sequence
            ) * self.subsequence_length
            if self.subsequence_length == 1:
                subsequence_slice = start_frame_idx
                transpose_axis = (2, 0, 1)
            else:
                subsequence_slice = slice(
                    start_frame_idx, start_frame_idx + self.subsequence_length
                )
                transpose_axis = (0, 3, 1, 2)
            if feature in ["video", "forward_flow", "backward_flow"]:
                transposed = torch.as_tensor(
                    np.transpose(
                        self.dset[feature][whole_sequence_idx, subsequence_slice],
                        transpose_axis,
                    )
                )
                resized = self.resizer(transposed)
                result.append(resized)
            elif feature == "depth":
                resized = self.resizer(
                    torch.as_tensor(
                        self.dset[feature][whole_sequence_idx, subsequence_slice]
                    )
                )
                result.append(resized)
            elif feature == 0:
                result.append(0)
            elif feature == "metadata.num_instances":
                result.append(
                    torch.as_tensor(
                        self.dset[feature][whole_sequence_idx].astype(np.int32)
                    )
                )
            else:
                result.append(
                    torch.as_tensor(
                        self.dset[feature][whole_sequence_idx, subsequence_slice],
                        dtype=torch.float32,
                    )
                )
        return tuple(result)

    def __len__(self):
        return self.n_subsequences

    def __del__(self):
        if self.f is not None:
            self.f.close()
        if self.tmp_dir is not None:
            self.tmp_dir.cleanup()

    def close(self):
        if self.f is not None:
            self.f.close()
            self.f = None
        if self.tmp_dir is not None:
            self.tmp_dir.cleanup()
            self.tmp_dir = None


def _plot_debug_sequences():
    import matplotlib.pyplot as plt

    dset = MOViFrameDataSet(
        "movi_c",
        feature_names=("video", "instances.image_positions"),
        load_to_memory=False,
        sub_sequence_length=7,
        seed=1,
    )
    first_sequence, first_pos = dset[len(dset) - 1]
    second_sequence, second_pos = dset[len(dset) - 2]
    fig, ax = plt.subplots(
        ncols=len(first_sequence), nrows=2, figsize=(2 * len(first_sequence), 4)
    )
    for i, (frame, pos) in enumerate(zip(first_sequence, first_pos)):
        ax[0, i].imshow(np.transpose(np.asarray(frame), (1, 2, 0)), extent=(0, 1, 0, 1))
        ax[0, i].scatter(np.asarray(pos[:, 0]), 1 - np.asarray(pos[:, 1]))
        ax[0, i].axis("off")
    for i, (frame, pos) in enumerate(zip(second_sequence, second_pos)):
        ax[1, i].imshow(np.transpose(np.asarray(frame), (1, 2, 0)), extent=(0, 1, 0, 1))
        ax[1, i].scatter(np.asarray(pos[:, 0]), 1 - np.asarray(pos[:, 1]))
        ax[1, i].axis("off")
    fig.tight_layout()
    plt.show()
    print(dset[0][1])
    dset.close()


if __name__ == "__main__":
    # to_hdf5("movi_e")
    # to_hdf5("movi_d")
    print("Creating dataset...")
    to_hdf5("movi_c")
    #to_hdf5("movi_d")
    print("Pre-computing features...")
    precompute_features("movi_c", encoder_name="vit_base_patch8_224_dino")
    # _plot_debug_sequences()
