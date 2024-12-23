import argparse
from functools import partial
from time import time
import math
from typing import Optional
import numpy as np
import tinycudann as tcnn
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch import nn

from splat_trainer.scene.mlp.torch_mlp import DirectionalMLP



def torch_model(args):
  model = DirectionalMLP(args.inputs, args.outputs, args.hidden, args.layers, 
            norm=partial(torch.nn.LayerNorm, elementwise_affine=False) if args.norm else nn.Identity, 
            activation=nn.ReLU, 
            output_activation=nn.Sigmoid,
            sh_degree=args.sh_degree)
  
  # model = torch.compile(model, options=dict(max_autotune=True), dynamic=True)
  return model


def mlp_tcnn(args):
  return tcnn.NetworkWithInputEncoding(
    args.inputs + 3, args.outputs,
    encoding_config=dict(
      otype = "composite",
      nested = [
        dict(otype = "SphericalHarmonics", 
            degree = args.sh_degree + 1, 
            n_dims_to_encode = 3
        ),
        
        dict(otype = "Identity",
            n_dims_to_encode = args.inputs)
      ]
    ), 
    
    network_config = dict(
      otype = "FullyFusedMLP",
      activation = "ReLU",
      output_activation = "Sigmoid",
      n_neurons = args.hidden,
      n_hidden_layers = args.layers,
    )
  )



class TCNNModel(torch.nn.Module):
  def __init__(self, mlp, append_sh=False):
    super().__init__()
    self.model = mlp
    self.append_sh = append_sh

  def forward(self, dir, x):
    if self.append_sh:
      x = torch.cat([dir, x], dim=1)

    return self.model(x)

def tcnn_model(args):
  mlp = mlp_tcnn(args)
  return TCNNModel(mlp, args.sh_degree is not None)

def bench_mlp(name, pos, points, features, model, iters=200, device="cuda:0", backward=True, randomize_batch=False):
  run_benchmark(f"{name}_warmup", pos, points, features, model, iters // 10, device, randomize_batch=randomize_batch) # warmup
  time, n = run_benchmark(f"{name}", pos, points, features, model, iters, 
                       device=device, randomize_batch=randomize_batch, backward=backward)
  
  print(f"{name}: N: {n/iters:.2f}Mpts Time: {time:.4f}s rate: {n / (time * 1000000):.2f}Mpts/s")


def run_benchmark(name, pos, points, features, model, iters=200, device="cuda:0", backward=True, randomize_batch=False):
  features = features.to(device).requires_grad_(True)
  pos = pos.to(device) # .requires_grad_(True)
  points = points.to(device) # .requires_grad_(True)
  model = model.to(device)
  # model = ChunkedModel(model, chunk_size=65536)
  num_points = features.shape[0]

  dir = F.normalize(points - pos.unsqueeze(0), dim=1)
  total = 0
  
  torch.cuda.synchronize()
  start = time()
  
  pbar = tqdm(desc = f"{name}: n={num_points}", total=iters)
  
  for i in range(iters):
    if randomize_batch:
      n = np.random.randint(0, num_points // 2)
      out = model.forward(dir[:n], features[:n])
    else:
      n = num_points
      out = model.forward(dir, features)
    err = out.sum()
    if backward:
      err.backward()

    total += n
    pbar.update(1)

  torch.cuda.synchronize()
  end = time()
  pbar.close()

  return end - start, total

def main():
  torch.set_float32_matmul_precision('high')

  parser = argparse.ArgumentParser("Benchmark tcnn")
  parser.add_argument("--n", type=int, default=1000000, help="Number of points")
  parser.add_argument("--hidden", type=int, default=64, help="Number of hidden neurons")
  parser.add_argument("--layers", type=int, default=3, help="Number of hidden layers")

  parser.add_argument("--inputs", type=int, default=32, help="Number of features")
  parser.add_argument("--outputs", type=int, default=3, help="Output dimensions")
  parser.add_argument("--sh_degree", type=int, default=3, help="Spherical harmonics degree")

  parser.add_argument("--device", type=str, default="cuda", help="Device")
  parser.add_argument("--iters", type=int, default=2000, help="Number of iterations")
  parser.add_argument("--norm", action="store_true", help="Use normalization")
  parser.add_argument("--randomize_batch", action="store_true", help="Randomize batch size")
  parser.add_argument("--forward", action="store_true", help="Run forward pass")

  args = parser.parse_args()

  features = torch.randn(args.n, args.inputs)

  points = torch.randn(args.n, 3)  * 100
  pos = F.normalize(torch.randn( (3,) ), dim=0)

  features = features.to(dtype=torch.float32)

  model = torch_model(args)
  print(model)

  print(args)

  with torch.autocast(device_type=args.device, dtype=torch.float16):
    bench_mlp("torch", pos, points, features, model, args.iters, args.device, 
              backward=not args.forward, randomize_batch=args.randomize_batch)

  bench_mlp("tcnn", pos, points, features, tcnn_model(args), args.iters, args.device, 
            backward=not args.forward, randomize_batch=args.randomize_batch)


  

if __name__ == "__main__":
  main()