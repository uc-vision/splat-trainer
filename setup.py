from pathlib import Path
from setuptools import find_packages, setup

setup(
    name='splat_trainer',
    version='0.1',
    packages=find_packages(),
    install_requires = [
        'taichi-splatting',
        'tqdm',
        'plyfile',
        'pypcd4',
        'hydra-core',
        'omegaconf',
        'roma>=1.4',
        'tabulate',
    ],

    entry_points={
      'console_scripts': 
      [
        'train-scan=splat_trainer.scripts.train_scan:main',
      ],
    },

)
