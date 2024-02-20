from os import path
from omegaconf import OmegaConf 


def add_resolvers():
    OmegaConf.register_new_resolver("dirname", path.dirname)
    OmegaConf.register_new_resolver("basename", path.basename)
    OmegaConf.register_new_resolver("without_ext", lambda x: path.splitext(x)[0])
    OmegaConf.register_new_resolver("ext", lambda x: path.splitext(x)[1])

    OmegaConf.register_new_resolver("scan_name", lambda scan_file: path.join(
        path.dirname(scan_file), f"gaussian_{path.splitext(path.basename(scan_file))[0]}"))

def pretty(cfg):
    return OmegaConf.to_yaml(cfg, resolve=True)