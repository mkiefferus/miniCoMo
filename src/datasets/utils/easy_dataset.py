import numpy as np

class EasyDataset:
    """a dataset that you can easily resize and combine.
    Examples:
    ---------
        2 * dataset ==> duplicate each element 2x

        10 @ dataset ==> set the size to 10 (random sampling, duplicates if necessary)

        dataset1 + dataset2 ==> concatenate datasets
    """
    def __add__(self, other):
        return CatDataset([self, other])

    def __rmul__(self, factor):
        return MulDataset(factor, self)

    def __rmatmul__(self, size):
        return ResizedDataset(size, self)

    def set_epoch(self, epoch): pass


class MulDataset(EasyDataset):
    def __init__(self, n, dataset):
        self.n, self.dataset = n, dataset

    def __len__(self): return self.n * len(self.dataset)

    def __getitem__(self, idx): return self.dataset[idx // self.n]


class ResizedDataset(EasyDataset):
    def __init__(self, size, dataset):
        self.size, self.dataset = size, dataset
        self._map = None

    def __len__(self): return self.size

    def set_epoch(self, epoch):
        rng = np.random.default_rng(epoch + 777)
        idxs = rng.permutation(len(self.dataset))
        repeats = (self.size + len(idxs) - 1) // len(idxs)
        self._map = np.tile(idxs, repeats)[:self.size]

    def __getitem__(self, idx):
        assert self._map is not None, "Call set_epoch first"
        return self.dataset[self._map[idx]]


class CatDataset(EasyDataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.cum_sizes = np.cumsum([len(d) for d in datasets])

    def __len__(self): return self.cum_sizes[-1]

    def __getitem__(self, idx):
        for i, cs in enumerate(self.cum_sizes):
            if idx < cs:
                prev = self.cum_sizes[i-1] if i > 0 else 0
                return self.datasets[i][idx - prev]