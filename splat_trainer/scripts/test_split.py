from taichi_splatting.tests.random_data import random_3d_gaussians, random_camera

from taichi_splatting import render_gaussians
from splat_trainer.gaussians import split

import torch

import cv2
import taichi as ti


def show_image(image):
  image = (image * 255).to(torch.uint8).cpu().numpy()
  cv2.imshow("rendering", image)
  return cv2.waitKey(0)


def main():
  ti.init(arch=ti.cuda, debug=True)

  while True:
    camera = random_camera(image_size=(640, 480))
    gaussians = random_3d_gaussians(5, camera, alpha_range=(1.0, 1.0), scale_factor=0.2)

    device = torch.device("cuda:0")
    camera, gaussians = camera.to(device), gaussians.to(device)

    image = render_gaussians(gaussians, camera_params=camera).image
    show_image(image)

    split_gaussians = split.split_gaussians(gaussians, n=2)

    image = render_gaussians(split_gaussians, camera_params=camera).image
    show_image(image)

if __name__ == "__main__":
  main()