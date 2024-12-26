from time import time
from typing import List
import torch
from tqdm import tqdm

from splat_trainer.scripts.checkpoint import arguments, with_trainer
from splat_trainer.trainer import Trainer

from taichi_splatting.perspective import CameraParams

def benchmark(name:str, iter, total:int):
  torch.cuda.synchronize()
  start = time()
  for x in tqdm(iter, desc=name, total=total):
    pass

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

    def render_forward():
      for camera_params in cameras:
        rendering = trainer.render(camera_params)
        loss, losses = trainer.losses(rendering, torch.zeros_like(rendering.image, device=trainer.device))
        yield loss

    def render_backward():
      for loss in render_forward():
        loss.backward()
        yield  


    def training_step():
      for _ in range(args.repeats):
        for filename, camera_params, image_idx, image in trainer.iter_train():
          trainer.training_step(filename, camera_params, image_idx, image)
          yield

    benchmark("render_forward", render_forward(), len(cameras))

    trainer.iter_train()



    with torch.enable_grad():
      benchmark("render_backward", render_backward(), len(cameras))
      benchmark("training_step", training_step(), len(cameras))



  with_trainer(f, args)
