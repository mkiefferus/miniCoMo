import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random

from .easy_dataset import EasyDataset


class BaseDataset(Dataset, EasyDataset):
    """Stores tensors x (features) and y (labels)."""
    def __init__(self, 
                 x: torch.Tensor, 
                 y: torch.Tensor, 
                 training: bool = True, 
                 num_samples: int = None):
        
        self.x = x[:num_samples] if num_samples is not None else x
        self.y = y[:num_samples] if num_samples is not None else y
        self.training = training

    def _process(self, x: torch.Tensor) -> torch.Tensor:
        
        # ===== Augment data =====

        if self.training:
            # Random rotation
            angle = random.choice([0, 90, 180, 270])
            x = TF.rotate(x, angle)

        # ===== Split Image =====

        h, w = x.shape[1:3]
        left = x[:, :, :w//2]
        right = x[:, :, w//2:]

        gt_left = left.clone()
        gt_right = right.clone()

        # Randomly swap left and right
        if self.training:
            if random.random() < 0.5:
                left, right = right, left

        return left, right, gt_left, gt_right

    def __len__(self) -> int:
        return self.x.size(0)

    def __getitem__(self, idx: int):
        img = self.x[idx]
        left, right, gt_left, gt_right = self._process(img)

        return dict(left=left, 
                    right=right, 
                    gt_left=gt_left, 
                    gt_right=gt_right, 
                    label=self.y[idx])