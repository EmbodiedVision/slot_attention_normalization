"""
Copyright 2024 Max-Planck-Gesellschaft
Code author: Markus Krimmel
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen
This source code is licensed under the license found in the
LICENSE.mpi.md file in the root directory of this source tree.
"""

import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def _categorical_to_onehot(k, mini, maxi):
    result = np.zeros(maxi - mini + 1)
    if mini <= k <= maxi:
        result[k - mini] = 1
    return result


class PropertyDataset(Dataset):
    _RANGES = {
        "shape": [1, 3],
        "size": [1, 2],
        "material": [1, 2],
        "color": [1, 8],
        "position": [np.array([-3, -3, 0.35]), np.array([3, 3, 0.7])],
    }
    _CATEGORICAL_PROPERTIES = ["shape", "size", "material", "color"]

    def __init__(
        self,
        data_dir,
        subset="train",
        features=("image", "property", "mask"),
        transform=None,
        preload=False,
    ):
        super(PropertyDataset, self).__init__()
        self._subset = subset
        self._features = features
        transform_list = [
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=0.5, std=0.5),
        ]
        if transform != None:
            transform_list.append(transform)
        self._transform = transforms.Compose(transform_list)

        self._data_file = h5py.File(os.path.join(data_dir, "transformed_data.hdf5"))
        self._group = self._data_file[self._subset]

        """
        self._CATEGORICAL_PROPERTIES = ["shape", "size", "material", "color"]

        self._RANGES = {prop: [float("inf"), -float("inf")] for prop in self._CATEGORICAL_PROPERTIES}
        self._RANGES["position"] = [np.array([100, 100, 100]),
                                    np.array([-100, -100, -100])]

        for i in range(200):
            n_objects = self._group["n_objects"][i]
            self._RANGES["position"] = [np.minimum(self._RANGES["position"][0], self._group["position"][i, 1:1+n_objects].min(axis=0)),
                                        np.maximum(self._RANGES["position"][1], self._group["position"][i, 1:1+n_objects].max(axis=0))]
            for prop in self._CATEGORICAL_PROPERTIES:
                self._RANGES[prop] = [min(self._RANGES[prop][0], self._group[prop][i, 1:1+n_objects].min(axis=0)),
                                      max(self._RANGES[prop][1], self._group[prop][i, 1:1+n_objects].max(axis=0))]

        """
        self._property_size = (
            sum(
                [
                    self._RANGES[prop][1] - self._RANGES[prop][0] + 1
                    for prop in self._CATEGORICAL_PROPERTIES
                ]
            )
            + 3
            + 1
        )  # +2 for position and +1 for visibility

        if preload:
            group_dict = {}
            for feature in features:
                if feature in ["image", "mask"]:
                    group_dict[feature] = self._group[feature][:]
                elif feature == "property":
                    for key in [
                        "position",
                        "visibility",
                    ] + self._CATEGORICAL_PROPERTIES:
                        group_dict[key] = self._group[key][:]
                else:
                    raise NotImplementedError
            self._group = group_dict
            self._data_file.close()
            self._data_file = None

    def __len__(self):
        return len(self._group["image"])

    def _compute_property(self, idx):
        result = np.zeros(
            (10, self._property_size), dtype=np.float32
        )  # List of one-hot encoded propertiess
        for obj_idx in range(10):
            start_idx = 0
            for prop_idx, prop in enumerate(self._CATEGORICAL_PROPERTIES):
                one_hot_prop = _categorical_to_onehot(
                    self._group[prop][idx, obj_idx + 1],
                    self._RANGES[prop][0],
                    self._RANGES[prop][1],
                )
                result[
                    obj_idx, start_idx : start_idx + len(one_hot_prop)
                ] = one_hot_prop
                start_idx = start_idx + len(one_hot_prop)
            result[obj_idx, start_idx : start_idx + 3] = (
                self._group["position"][idx, obj_idx + 1] - self._RANGES["position"][0]
            ) / (self._RANGES["position"][1] - self._RANGES["position"][0])
            result[obj_idx, start_idx + 3] = self._group["visibility"][idx, obj_idx + 1]
            assert start_idx + 3 == self._property_size - 1
        return result

    @classmethod
    def decode_property(cls, single_property):
        categorical_predictions = []
        start_idx = 0
        for prop_idx, prop in enumerate(cls._CATEGORICAL_PROPERTIES):
            prop_len = cls._RANGES[prop][1] - cls._RANGES[prop][0] + 1
            prop_vals = single_property[start_idx : start_idx + prop_len]
            categorical_predictions.append(np.argmax(prop_vals) + cls._RANGES[prop][0])
            start_idx += prop_len
        pos = single_property[start_idx : start_idx + 3]
        start_idx += 3
        assert start_idx == len(single_property) - 1
        real_obj = single_property[start_idx]
        return pos, *categorical_predictions, real_obj

    @property
    def property_size(self):
        return self._property_size

    def __getitem__(self, idx):
        result = []
        for feature in self._features:
            if feature == "image":
                result.append(
                    self._transform(
                        torch.from_numpy(self._group["image"][idx].transpose(2, 0, 1))
                    )
                )
            elif feature == "mask":
                result.append(
                    torch.from_numpy(self._group["mask"][idx].transpose(2, 0, 1))
                )
            elif feature == "property":
                result.append(torch.from_numpy(self._compute_property(idx)))
            else:
                raise NotImplementedError
        return tuple(result)

    def close(self):
        if self._data_file is not None:
            self._data_file.close()
            self._data_file = None


if __name__ == "__main__":
    from slot_attention_sets import DATA_DIR
    from slot_attention_sets.metrics import average_precision_clevr
    from torch.utils.data import DataLoader

    ds = PropertyDataset(DATA_DIR, subset="val")
    print(ds._RANGES)
    # loader = DataLoader(ds)
    # _, batch, _ = next(iter(loader))
    # batch = batch.numpy()
    # print(batch.shape)
    # print(average_precision_clevr(batch, batch, -1))
