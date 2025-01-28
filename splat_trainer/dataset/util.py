import torch

def partition_stride(idx: torch.Tensor, val_stride: int):
    assert val_stride != 0
    val_idx = idx[::val_stride]
    mask = (idx % val_stride != 0)
    train_idx = idx[mask]
    
    return train_idx, val_idx


def split_train_val(n: int, val_stride: int):
    idx = torch.arange(n, dtype=torch.long)
    if val_stride > 0:
        return partition_stride(idx, val_stride)
    else:
        return idx, torch.empty(0, dtype=torch.long)
