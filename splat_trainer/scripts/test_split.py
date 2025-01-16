from taichi_splatting.tests.random_data import random_3d_gaussians, random_camera

from taichi_splatting import render_gaussians, TaichiQueue
from taichi_splatting import Gaussians3D
from splat_trainer.gaussians import split

import torch

import cv2
import taichi as ti


def show_image(image):
  image = (image * 255).to(torch.uint8).cpu().numpy()
  cv2.imshow("rendering", image)
  while cv2.waitKey(1) == -1:
    pass


def main():
  TaichiQueue.init(arch=ti.cuda, debug=True)

  while True:
    camera = random_camera(image_size=(640, 480))
    gaussians = random_3d_gaussians(5, camera, alpha_range=(0.5, 1.0), scale_factor=0.2)

    device = torch.device("cuda:0")
    camera, gaussians = camera.to(device), gaussians.to(device)

    image = render_gaussians(gaussians, camera_params=camera).image
    show_image(image)

    split_gaussians = split.split_gaussians_uniform(gaussians.to_tensordict(), n=2)

    image = render_gaussians(Gaussians3D.from_tensordict(split_gaussians), camera_params=camera).image
    show_image(image)

if __name__ == "__main__":
  main()