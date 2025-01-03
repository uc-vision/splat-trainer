from .dataset import Dataset, ImageView
from .colmap.dataset import COLMAPDataset

def ScanDataset(*args, **kwargs):
    from .scan.dataset import ScanDataset
    return ScanDataset(*args, **kwargs)


__all__ = [
    "ImageView",
    "Dataset",
    
    "ScanDataset",  
    "COLMAPDataset"
    ]