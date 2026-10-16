import os
import pickle
from types import ModuleType

import torch
from torch.utils.data import Dataset, random_split
from torch_geometric.loader import DataLoader

from building_diffusion.data.graph import VolumeDesignGraph

# Attributes that get a batch vector (e.g. data.voxel_feature_batch) when graphs are batched
FOLLOW_BATCH = ['program_target_ratio', 'program_class_feature', 'voxel_feature']

# Processed graphs were pickled while VolumeDesignGraph lived at these module paths.
_LEGACY_MODULES = {"VolumeDesignGraph", "Data.VolumeDesignGraph"}


class _LegacyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module in _LEGACY_MODULES and name == "VolumeDesignGraph":
            return VolumeDesignGraph
        return super().find_class(module, name)


# Stand-in for the pickle module, so torch.load resolves legacy class paths without sys.path hacks
_legacy_pickle = ModuleType("legacy_pickle")
_legacy_pickle.Unpickler = _LegacyUnpickler
_legacy_pickle.load = lambda f, **kwargs: _LegacyUnpickler(f, **kwargs).load()


class LargeFilenameDataset(Dataset):
    """Processed graphs loaded from disk one file at a time (fname_list in data_dir)."""
    def __init__(self, data_dir, fname_list):
        self.data_dir = data_dir
        self.fname_list = fname_list

    def __len__(self):
        return len(self.fname_list)

    def __getitem__(self, i):
        return torch.load(os.path.join(self.data_dir, self.fname_list[i]), weights_only=False, pickle_module=_legacy_pickle)


def make_splits(fname_list, train_size, test_size, batch_size, val_fraction=0.2, seed=42):
    """
    Split the sorted list of processed files into train / validation / test file lists:
    sizes are rounded down to a multiple of batch_size, the first train_size files are split
    80/20 into train and validation with a seeded random_split, and the next test_size files are the test set.
    """
    train_size = train_size // batch_size * batch_size
    test_size = test_size // batch_size * batch_size

    val_subset_size = int(val_fraction * train_size)
    train_subset_size = train_size - val_subset_size
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(range(train_size), [train_subset_size, val_subset_size], generator=generator)

    train_files = [fname_list[i] for i in train_subset.indices]
    val_files = [fname_list[i] for i in val_subset.indices]
    test_files = fname_list[train_size:train_size + test_size]
    return train_files, val_files, test_files


def make_loader(data_dir, fname_list, batch_size, shuffle, num_workers):
    """PyG DataLoader over the processed graphs fname_list in data_dir, with batch vectors for FOLLOW_BATCH."""
    return DataLoader(LargeFilenameDataset(data_dir, fname_list), follow_batch=FOLLOW_BATCH,
                      batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=False)


def make_split_loaders(data_cfg, batch_size, test_batch_size):
    """Train / validation / test loaders of the processed dataset (DataConfig), split with make_splits."""
    fname_list = sorted(os.listdir(data_cfg.processed_dir))
    train_files, val_files, test_files = make_splits(fname_list, data_cfg.train_size, data_cfg.test_size, batch_size)
    print("Total %d data: %d train / %d val / %d test" % (len(fname_list), len(train_files), len(val_files), len(test_files)))

    train_loader = make_loader(data_cfg.processed_dir, train_files, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = make_loader(data_cfg.processed_dir, val_files, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = make_loader(data_cfg.processed_dir, test_files, batch_size=test_batch_size, shuffle=False, num_workers=8)
    return train_loader, val_loader, test_loader
