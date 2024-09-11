"""
Copyright 2024 Max-Planck-Gesellschaft
Code author: Markus Krimmel
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen
This source code is licensed under the license found in the
LICENSE.mpi.md file in the root directory of this source tree.
"""

import datetime
import os
import pickle
import subprocess
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from sa_generalization.object_discovery import DATA_DIR


class EndlessDataLoader:
    def __init__(self, data_loader):
        self._data_loader = data_loader
        self._data_loader_iter = iter(data_loader)
        self._buf = []

    def __iter__(self):
        return self

    def __next__(self):
        try:
            result = next(self._data_loader_iter)
        except StopIteration:
            self._data_loader_iter = iter(self._data_loader)
            result = next(self._data_loader_iter)
        return result

    @property
    def buffer_length(self):
        return len(self._buf)

    def pop(self):
        return self._buf.pop()

    def push_to_buf(self, batch):
        self._buf.insert(0, batch)


class FailedCopy(Exception):
    pass


def copy_folder_to_tmp(path, file_types=None, timeout_h=3):
    tmp_dir = tempfile.TemporaryDirectory()
    arguments = []
    if file_types is not None:
        arguments = ["--prune-empty-dirs"]
        arguments += ["--include", f"{os.path.basename(path)}/"]
        for file_type in file_types:
            arguments.append("--include")
            arguments.append(f"*.{file_type}")
        arguments += ["--exclude", "*"]
    all_args = ["rsync", "-ah", "--progress", *arguments, str(path), tmp_dir.name]
    print(subprocess.list2cmdline(all_args))
    code = subprocess.call(all_args, timeout=timeout_h * 3600)
    if code != 0 and code != 28:
        raise FailedCopy()
    elif code == 28:
        raise Exception("No disk space left to copy to...")
    return tmp_dir


class SADataset(Dataset):
    def __init__(
        self,
        dset="iodine",
        subset="train",
        with_masks=False,
        with_n_objects=False,
        max_objects=10,
        min_objects=0,
        num_images="inf",
        seed=None,
        preload=False,
        transform=None,
        driver=None,
        copy_to_tmp=False,
        parent_dir=None,
        timeout_h=3,
    ):
        """

        :param str dset: The dataset to use. Either "iodine" or "original_clevr".
        :param str subset:  The subset to use. Either "train" or "val" if dset="original" or "train", "val" or "test" if "iodine"
        :param Union[int] max_objects:  The maximum number of objects in the scene
        :param Union[None, torch.nn.Module] transform:   A transformation to apply to the images
        :param Union[int, str] num_images:  Number of images to consider in this dataset
        :param Union[int, None] seed:   Seed for selecting images
        :param bool preload:    Whether or not to load all images into memory
        :param Union[None, torch.nn.Module]: Additional transformation to apply
        """

        read_sequentially = (max_objects == "inf") and (min_objects <= 0)

        if with_masks and dset not in [
            "iodine",
            "bouncing_balls",
            "multi_mnist",
            "mgso128",
            "mgso256",
            "tetrominoes",
        ]:
            raise ValueError("Masks are not supported for this dataset.")

        self._dset = dset
        self._subset = subset
        self._with_masks = with_masks
        self._with_n_objects = with_n_objects
        self._max_objects = max_objects
        self._min_objects = min_objects
        self._rng = np.random.RandomState(seed)
        self._preload = preload
        transform_list = [
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=0.5, std=0.5),
        ]
        if transform != None:
            transform_list.append(transform)

        self._transform = transforms.Compose(transform_list)

        dset_to_folder = {
            "iodine": "IODINE",
            "original": "CLEVR_v1.0",
            "bouncing_balls": "BOUNCING_BALLS",
            "multi_mnist": "MULTI_MNIST",
            "mgso128": "MGSO/MGSO128",
            "mgso256": "MGSO/MGSO256",
            "tetrominoes": "TETROMINOES",
        }
        parent_dir = Path(parent_dir) if parent_dir is not None else DATA_DIR
        self.data_dir = parent_dir.joinpath(dset_to_folder[self._dset])
        if copy_to_tmp:
            self.tmp_dir = copy_folder_to_tmp(
                self.data_dir, file_types=["hdf5", "pkl"], timeout_h=timeout_h
            )
            self.data_dir = Path(
                os.path.join(self.tmp_dir.name, dset_to_folder[self._dset])
            )
        else:
            self.tmp_dir = None

        with open(self.data_dir.joinpath("images_for_max_objects.pkl"), "rb") as f:
            indices_max_objs = pickle.load(f)[subset]
            indices_available = [
                description["global_index"]
                for description in indices_max_objs[self._max_objects]
            ]

            indices_excluded = set()
            for i in range(1, min_objects):
                if i in indices_max_objs:
                    indices_excluded.update(
                        [
                            description["global_index"]
                            for description in indices_max_objs[i]
                        ]
                    )

            indices_available = list(
                filter(lambda idx: idx not in indices_excluded, indices_available)
            )

            if num_images != "inf":
                self._image_indices = self._rng.choice(
                    indices_available, num_images, replace=False
                )
            else:
                self._image_indices = indices_available

        transformed_data_path = self.data_dir.joinpath("transformed_data.hdf5")
        t0 = datetime.datetime.now()
        self._data_file = h5py.File(transformed_data_path, "r", driver=driver)
        print(
            f"Getting the HDF5 file with {driver} driver took {int((datetime.datetime.now() - t0).total_seconds()) // 60} minutes"
        )
        self._group = self._data_file[self._subset]
        t0 = datetime.datetime.now()
        if self._preload:
            images = []
            masks = []
            n_objects = []
            print("Loading Dataset...")
            if read_sequentially:
                num_images_int = (
                    len(self._group["image"]) if num_images == "inf" else num_images
                )
                self._torch_images = torch.from_numpy(
                    self._group["image"][:num_images_int].transpose(0, 3, 1, 2)
                )
                if self._with_masks:
                    self._torch_masks = torch.from_numpy(
                        self._group["mask"][:num_images_int].transpose(0, 3, 1, 2)
                    )
                    assert (
                        self._torch_masks.dtype == torch.bool
                    ), self._torch_masks.dtype

                if self._with_n_objects:
                    self._torch_n_objects = torch.from_numpy(
                        self._group["n_objects"][:num_images_int]
                    )
                    assert not torch.is_floating_point(
                        self._torch_n_objects
                    ), self._torch_n_objects.dtype
            else:
                for index in tqdm(self._image_indices):
                    images.append(
                        torch.from_numpy(self._group["image"][index].transpose(2, 0, 1))
                    )
                    if self._with_masks:
                        masks.append(
                            torch.from_numpy(
                                self._group["mask"][index].transpose(2, 0, 1)
                            )
                        )
                    if self._with_n_objects:
                        n_objects.append(self._group["n_objects"][index])

                self._torch_images = torch.stack(images)
                if self._with_masks:
                    self._torch_masks = torch.stack(masks)
                    assert (
                        self._torch_masks.dtype == torch.bool
                    ), self._torch_masks.dtype
                if self._with_n_objects:
                    self._torch_n_objects = torch.Tensor(n_objects).int()

            assert self._torch_images.dtype == torch.uint8, self._torch_images.dtype
            self._data_file.close()
            print(
                f"Loading data took {int((datetime.datetime.now() - t0).total_seconds()) // 60} minutes"
            )

    def __len__(self):
        return len(self._image_indices)

    def __getitem__(self, idx):
        result = []

        if self._preload:
            result.append(self._transform(self._torch_images[idx]))
            if self._with_masks:
                result.append(self._torch_masks[idx])
            if self._with_n_objects:
                result.append(self._torch_n_objects[idx])
        else:
            result.append(
                self._transform(
                    torch.from_numpy(
                        self._group["image"][self._image_indices[idx]].transpose(
                            2, 0, 1
                        )
                    )
                )
            )
            if self._with_masks:
                result.append(
                    torch.from_numpy(
                        self._group["mask"][self._image_indices[idx]].transpose(2, 0, 1)
                    )
                )
            if self._with_n_objects:
                result.append(self._group["n_objects"][self._image_indices[idx]])

        return tuple(result) if len(result) > 1 else result[0]

    def __del__(self):
        self._data_file.close()
        if self.tmp_dir is not None:
            self.tmp_dir.cleanup()


def test_dataset(n, dset, with_masks, max_objects=10, min_objects=0, subset="train"):
    dataset = SADataset(
        dset=dset,
        subset=subset,
        with_masks=with_masks,
        max_objects=max_objects,
        min_objects=min_objects,
        preload=True,
        seed=42,
        num_images=50,
    )

    import matplotlib.pyplot as plt

    nrows = 1 if not with_masks else 2
    fig, ax = plt.subplots(nrows, n)
    # If we show masks, ax has to be indexed by integers, else by tuples. Convenience functon:
    convert_idx = lambda a, b: b if not with_masks else (a, b)
    for i in range(n):
        data = dataset[i]
        if with_masks:
            image, mask = data
            ax[convert_idx(1, i)].imshow(mask[0].numpy())
            ax[convert_idx(1, i)].axis("off")
        else:
            image = data
        ax[convert_idx(0, i)].imshow((image.numpy().transpose([1, 2, 0]) + 1) / 2)
        ax[convert_idx(0, i)].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_dataset(4, "iodine", True, max_objects=10, min_objects=6)
    # test_dataset(4, "iodine", True, max_objects=3)
    # test_dataset(4, "original", False, max_objects=3)
    # test_dataset(4, "iodine", True, max_objects=3, subset="test")
    # test_dataset(4, "original", False, max_objects=3, subset="val")
