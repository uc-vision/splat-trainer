import argparse
from time import time
import math
import numpy as np
import tinycudann as tcnn
import torch
import torch.nn.functional as F
from tqdm import tqdm





def main():
  parser = argparse.ArgumentParser("Benchmark tcnn")
  parser.add_argument("--n", type=int, default=1000000, help="Number of points")
  parser.add_argument("--hidden", type=int, default=32, help="Number of hidden neurons")
  parser.add_argument("--layers", type=int, default=2, help="Number of hidden layers")
  parser.add_argument("--num_features", type=int, default=8, help="Number of features")
  

  args = parser.parse_args()

  model = tcnn.NetworkWithInputEncoding(
    args.num_features + 3, 3,
    encoding_config=dict(
      otype = "composite",
      nested = [
        dict(otype = "SphericalHarmonics", 
            degree = 4, 
            n_dims_to_encode = 3
        ),
        
        dict(otype = "Identity",
            n_dims_to_encode = args.num_features)
      ]
    ), 
    
    network_config = dict(
      otype = "FullyFusedMLP",
      activation = "ReLU",
      output_activation = "None",
      n_neurons = args.hidden,
      n_hidden_layers = args.layers,
    )
  )


  
  device = "cuda:0"

  dir = F.normalize(torch.randn( (args.n, 3) ), dim=1)
  features = torch.cat([dir, torch.randn(args.n, args.num_features)], dim=1).to(device)

  features.requires_grad_(True)

  iters = 100
  start = time()
  
  pbar = tqdm(desc = f"n={args.n}", total=iters)
  for i in range(iters):
    out = model.forward(features)

    err = out.sum()
    err.backward()

    pbar.update(1)

  torch.cuda.synchronize()
  end = time()

  print(f"N: {args.n} Rate: {iters / (end - start)}")


  

if __name__ == "__main__":
  main()