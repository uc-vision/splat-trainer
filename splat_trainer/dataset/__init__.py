from .dataset import Dataset, CameraView


def ScanDataset(*args, **kwargs):
    from .scan.dataset import ScanDataset
    return ScanDataset(*args, **kwargs)


__all__ = ["ScanDataset", "Dataset", "CameraView"]