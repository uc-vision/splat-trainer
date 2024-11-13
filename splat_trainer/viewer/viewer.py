from abc import ABCMeta, abstractmethod

import typing
if typing.TYPE_CHECKING:
  from splat_trainer.trainer import Trainer

class ViewerConfig(metaclass=ABCMeta):
  @abstractmethod
  def create_viewer(self, trainer: 'Trainer') -> 'Viewer':
    raise NotImplementedError()
  
class Viewer(metaclass=ABCMeta):
  @abstractmethod
  def update(self):
    raise NotImplementedError()
  
  @abstractmethod
  def wait_for_exit(self):
    raise NotImplementedError()

  @abstractmethod
  def spin(self):
    raise NotImplementedError()


class NilViewerConfig(ViewerConfig):
  def create_viewer(self, trainer: 'Trainer'):
    return NilViewer()

class NilViewer(Viewer):
  def update(self):
    pass

  def wait_for_exit(self):
    pass

  def spin(self):
    pass