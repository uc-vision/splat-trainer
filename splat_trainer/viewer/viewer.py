from abc import ABCMeta, abstractmethod

import typing
if typing.TYPE_CHECKING:
  from splat_trainer.trainer import Trainer

class ViewerConfig(metaclass=ABCMeta):
  @abstractmethod
  def create_viewer(self, trainer: 'Trainer', enable_training: bool = False) -> 'Viewer':
    raise NotImplementedError()
  
class Viewer(metaclass=ABCMeta):
  @abstractmethod
  def update(self):
    raise NotImplementedError()
  
  @abstractmethod
  def wait_for_exit(self):
    """ Run while there are still clients connected """
    raise NotImplementedError()

  @abstractmethod
  def spin(self):
    """ Run and update the viewer indefinitely """
    raise NotImplementedError()


class NilViewerConfig(ViewerConfig):
  def create_viewer(self, trainer: 'Trainer', enable_training: bool = False):
    return NilViewer()

class NilViewer(Viewer):
  def update(self):
    pass

  def wait_for_exit(self):
    pass

  def spin(self):
    pass