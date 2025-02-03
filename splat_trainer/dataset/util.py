import torch

def partition_stride(idx: torch.Tensor, val_stride: int):
    assert val_stride != 0
    val_idx = idx[::val_stride]
    mask = (idx % val_stride != 0)
    train_idx = idx[mask]
    
    return train_idx, val_idx


def split_every(n: int, test_every: int=8, padding: int=0):
    """ Split the frames into training and test sets, 
    where each test set is separated by test_every frames.
    
    Args:
        n: Total number of frames
        test_every: Number of frames between test samples
        padding: Number of frames to exclude from both ends
    """
    idx = torch.arange(padding, n - padding, dtype=torch.long)
    
    if test_every > 0:
        return partition_stride(idx, test_every)
    else:
        return idx, torch.empty(0, dtype=torch.long)

    

    
def expand_index(idx: torch.Tensor, num_cameras: int) -> torch.Tensor:
    """ Expand chosen index to image_indexes from camera_indexes """
    # Create camera indices [0,1,2,...,num_cameras-1] repeated for each input index
    camera_indices = torch.arange(num_cameras).repeat(len(idx))
    # Create frame indices [idx[0],idx[0],...,idx[1],idx[1],...] repeated num_cameras times each
    frame_indices = idx.repeat_interleave(num_cameras)
    # Combine them to get final indices
    return camera_indices + frame_indices * num_cameras




def concat_lists(xs):
  return [x for x in xs for x in x]



