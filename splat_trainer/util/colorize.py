import torch
import cv2
import numpy as np

def get_cv_colormap(cmap=cv2.COLORMAP_TURBO):
  colormap = np.arange(256, dtype=np.uint8).reshape(1, -1)
  colormap = cv2.applyColorMap(colormap, cmap)
  return torch.from_numpy(colormap.astype(np.uint8)).squeeze(0) 

 
def colorize_depth(color_map, depth, near=0.2):
  depth = depth.clone()
  depth[depth == 0] = torch.inf
  
  inv_depth =  (near / depth).clamp(0, 1)
  inv_depth = (255 * inv_depth).to(torch.int)

  return (color_map[inv_depth])


def colorize(color_map, t):

  t =  (t.clamp(0, 1) * 255).to(torch.int) 
  return (color_map[t])