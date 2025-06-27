import torch
from torch.utils.data import Dataset


class simpleDataset(Dataset):
    """Stores tensors x (features) and y (labels)."""
    x: torch.Tensor
    y: torch.Tensor

    def __len__(self) -> int:
        return self.x.size(0)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]