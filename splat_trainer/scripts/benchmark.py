from time import time
from typing import List
import torch
from tqdm import tqdm

from splat_trainer.scripts.checkpoint import arguments, with_trainer
from splat_trainer.trainer import Trainer

from taichi_splatting.perspective import CameraParams

def benchmark(name:str, f, cameras:List[CameraParams]):
  torch.cuda.synchronize()
  start = time()
  for camera_params in tqdm(cameras, desc=name):
    f(camera_params)
  torch.cuda.synchronize()
  end = time()
  elapsed = end - start

  print(f"Elapsed {name}: {elapsed:.2f}s")



def main():
  parser = arguments()
  parser.add_argument("--repeats", type=int, default=1, help="Number of times to repeat each benchmark")
  args = parser.parse_args()


  def f(trainer:Trainer):
    trainer.warmup()

    cameras = [trainer.camera_params(i) for i in range(trainer.camera_table.num_images)] * args.repeats

    def render_forward(camera_params:CameraParams):
      rendering =trainer.render(camera_params)
      loss, losses = trainer.losses(rendering, torch.zeros_like(rendering.image, device=trainer.device))
      return loss


    def render_backward(camera_params:CameraParams):
      loss = render_forward(camera_params)
      loss.backward()

    benchmark("render_forward", render_forward, cameras)

    with torch.enable_grad():
      benchmark("render_backward", render_backward, cameras)



  with_trainer(f, args)
