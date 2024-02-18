import os
from omegaconf import OmegaConf 


def add_resolvers():
    OmegaConf.register_new_resolver("dirname", os.path.dirname)
    OmegaConf.register_new_resolver("basename", os.path.basename)
    OmegaConf.register_new_resolver("without_ext", lambda x: os.path.splitext(x)[0])

def pretty(cfg):
    return OmegaConf.to_yaml(cfg, resolve=True)