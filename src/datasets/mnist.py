
import numpy as np # linear algebra
import struct
import torch

import os.path as osp

from .utils.dataset import simpleDataset


class MNISTDataset(simpleDataset):
    def __init__(self, root: str = "data/mnist", train: bool = True):
        split = "train" if train else "t10k"
        imgs = osp.join(root, f"{split}-images-idx3-ubyte/{split}-images-idx3-ubyte")
        lbls = osp.join(root, f"{split}-labels-idx1-ubyte/{split}-labels-idx1-ubyte")
        images, labels = self._read(imgs, lbls)
        self.x = torch.from_numpy(images).unsqueeze(1).float() / 255
        self.y = torch.tensor(labels, dtype=torch.long)

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
    

# matplotlib inline
import random
import matplotlib.pyplot as plt


def show_images(images, title_texts):
    cols = 5
    rows = (len(images) + cols - 1) // cols
    plt.figure(figsize=(10, 6))  # Smaller canvas
    index = 1    
    for img, title in zip(images, title_texts):        
        plt.subplot(rows, cols, index)        
        plt.imshow(img, cmap=plt.cm.gray)
        if title:
            plt.title(title, fontsize=8)  # Smaller text
        plt.axis('off')  # Optional: hide axes
        index += 1
    plt.tight_layout()
    plt.show()


# Load training and test datasets
train_set = MNISTDataset(root="data/mnist", train=True)
test_set = MNISTDataset(root="data/mnist", train=False)

print(len(train_set))

# Show some random training and test images
images_2_show = []
titles_2_show = []

for _ in range(10):
    r = random.randint(0, len(train_set.x) - 1)
    images_2_show.append(train_set.x[r][0].numpy())
    titles_2_show.append(f"training image [{r}] = {train_set.y[r].item()}")

for _ in range(5):
    r = random.randint(0, len(test_set.x) - 1)
    images_2_show.append(test_set.x[r][0].numpy())
    titles_2_show.append(f"test image [{r}] = {test_set.y[r].item()}")

show_images(images_2_show, titles_2_show)
