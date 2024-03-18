from .dataset import Dataset, CameraView
from .colmap.dataset import COLMAPDataset

def ScanDataset(*args, **kwargs):
    from .scan.dataset import ScanDataset
    return ScanDataset(*args, **kwargs)


__all__ = [
    "CameraView",
    "Dataset",
    
    "ScanDataset",  
    "COLMAPDataset"
    ]