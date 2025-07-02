
import numpy as np # linear algebra
import struct
import torch

import os.path as osp

from .utils.dataset import simpleDataset


class MNISTDataset(simpleDataset):
    def __init__(self, root: str = "data/mnist", train: bool = True, **kwargs):
        split = "train" if train else "t10k"
        imgs = osp.join(root, f"{split}-images-idx3-ubyte/{split}-images-idx3-ubyte")
        lbls = osp.join(root, f"{split}-labels-idx1-ubyte/{split}-labels-idx1-ubyte")
        images, labels = self._read(imgs, lbls)
        x = torch.from_numpy(images).unsqueeze(1).float() / 255
        y = torch.tensor(labels, dtype=torch.long)
        super().__init__(x=x, y=y, training=train, **kwargs)

    @staticmethod
    def _read(img_path: str, lbl_path: str):
        with open(lbl_path, "rb") as f:
            magic, n = struct.unpack(">II", f.read(8))
            if magic != 2049:
                raise ValueError(f"label magic {magic}")
            labels = np.frombuffer(f.read(), dtype=np.uint8).copy()

        with open(img_path, "rb") as f:
            magic, n, r, c = struct.unpack(">IIII", f.read(16))
            if magic != 2051:
                raise ValueError(f"image magic {magic}")
            images = np.frombuffer(f.read(), dtype=np.uint8).copy().reshape(n, r, c)

        return images, labels
    