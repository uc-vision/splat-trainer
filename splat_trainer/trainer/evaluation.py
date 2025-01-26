# Python standard library
from dataclasses import dataclass, replace
from functools import cached_property

# Third party packages
import torch
from fused_ssim import fused_ssim
from splat_trainer.util.colors import fit_colors, compute_psnr


from taichi_splatting import Rendering



@dataclass(frozen=True)
class Evaluation:
  filename:str
  rendering:Rendering
  source_image:torch.Tensor

  @property
  def image_id(self):
    return self.filename.replace('/', '_')
  
  @property
  def image(self):
    return self.rendering.image
    
  @cached_property
  def psnr(self):
    return compute_psnr(self.image, self.source_image).item()
  
  @cached_property
  def l1(self):
    return torch.nn.functional.l1_loss(self.image, self.source_image).item()
  
  @cached_property
  def ssim(self):
    ref = self.source_image.unsqueeze(0).permute(0, 3, 1, 2).to(memory_format=torch.channels_last)
    pred = self.image.unsqueeze(0).permute(0, 3, 1, 2).to(memory_format=torch.channels_last)

    return fused_ssim(pred, ref, padding="valid").item()

  @cached_property
  def metrics(self):
    return dict(psnr=self.psnr, l1=self.l1, ssim=self.ssim)
  
  def color_corrected(self) -> 'Evaluation':
    return replace(self, image=fit_colors(self.image, self.source_image))
    

