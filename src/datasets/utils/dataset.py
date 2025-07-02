import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random



class simpleDataset(Dataset):
    """Stores tensors x (features) and y (labels)."""
    def __init__(self, 
                 x: torch.Tensor, 
                 y: torch.Tensor, 
                 training: bool = True, 
                 num_samples: int = None):
        
        self.x = x[:num_samples] if num_samples is not None else x
        self.y = y[:num_samples] if num_samples is not None else y
        self.training = training

    def _augment(self, x: torch.Tensor) -> torch.Tensor:

        # Random rotation
        angle = random.choice([0, 90, 180, 270])
        x = TF.rotate(x, angle)

        return x

    def __len__(self) -> int:
        return self.x.size(0)

    def __getitem__(self, idx: int):
        img = self.x[idx]
        # if self.training: img = self._augment(img)

        left = img[:, :, :14]
        right = img[:, :, 14:]

        return dict(left=left, right=right, gt_img=img, label=self.y[idx])