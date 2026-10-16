import os
import random
from contextlib import contextmanager

import numpy as np
import torch
import torch_geometric


def set_seed(seed=42, deterministic=False):
    """
    Seed every random number generator used by the project: Python, NumPy, PyTorch (CPU and all GPUs)
    and PyTorch Geometric. Call it at the start of a script, before the data loaders and the models are created.

    deterministic: also make PyTorch use deterministic algorithms, so that two runs with the same seed give
    the same results on GPU too. Without it, GPU operations such as the scatter / sum of the graph layers
    are not deterministic, and runs drift apart slightly. On GPU, the deterministic scatter is slower and needs
    much more memory (the diffusion model with batch size 32 does not fit in 16 GB). On CPU, runs with the
    same seed are identical without it.
    """
    random.seed(seed)                   # Python random
    np.random.seed(seed)                # NumPy
    torch.manual_seed(seed)             # PyTorch CPU
    torch.cuda.manual_seed(seed)        # PyTorch GPU
    torch.cuda.manual_seed_all(seed)    # all GPUs
    torch_geometric.seed_everything(seed)  # PyTorch Geometric (seeds Python, NumPy and PyTorch again)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if deterministic:
        # Required by cuBLAS for deterministic matrix products; set before the first CUDA operation
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.use_deterministic_algorithms(True)


@contextmanager
def allow_nondeterministic():
    """
    Temporarily allow PyTorch operations that have no deterministic implementation (set_seed(deterministic=True)
    makes them raise an error). Use it only for operations whose result is repeatable in practice.
    """
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(False)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
