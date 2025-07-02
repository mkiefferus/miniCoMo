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

    def _process(self, x: torch.Tensor) -> torch.Tensor:
        
        # ===== Augment data =====

        if self.training:
            # Random rotation
            angle = random.choice([0, 90, 180, 270])
            x = TF.rotate(x, angle)
            
            # # Random horizontal flip
            # if random.random() < 0.5:
            #     x = TF.hflip(x)

            # # Random vertical flip
            # if random.random() < 0.5:
            #     x = TF.vflip(x)

        # ===== Split Image =====

        h, w = x.shape[1:3]
        left = x[:, :, :w//2]
        right = x[:, :, w//2:]


        # Randomly swap left and right
        if self.training and random.random() < 0.5:
            left, right = right, left

        return left, right, x

    def __len__(self) -> int:
        return self.x.size(0)

    def __getitem__(self, idx: int):
        img = self.x[idx]
        left, right, img = self._process(img)

        return dict(left=left, right=right, gt_img=img, label=self.y[idx])