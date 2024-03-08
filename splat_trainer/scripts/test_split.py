from taichi_splatting.tests.random_data import random_3d_gaussians, random_camera
from taichi_splatting import render_gaussians
import torch

import cv2


def main():

  camera = random_camera()
  gassians = random_3d_gaussians(10, camera)


  rendering = render_gaussians(gassians, camera_params=camera)

  image = (rendering.image * 255).to(torch.uint8).cpu().numpy()
  cv2.imshow("rendering", image)