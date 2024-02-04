from pathlib import Path
from setuptools import find_packages, setup

scripts = [f'{script.stem} = splat_trainer.scripts.{script.stem}:main'
  for script in Path('splat_trainer/scripts').glob('*.py') if script.stem != '__init__']


setup(
    name='splat_trainer',
    version='0.1',
    packages=find_packages(),
    install_requires = [
        'taichi-splatting',
        'tqdm'
    ],

    entry_points={
      'console_scripts': scripts
    },

    include_package_data=True,
    package_data={'': ['*.ui', '*.yaml']},


)
