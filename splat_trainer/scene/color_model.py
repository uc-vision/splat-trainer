import math
from typing import Tuple
import torch
import torch.nn.functional as F
import torch.nn as nn
from taichi_splatting.perspective import (CameraParams)


def positional_model(hidden:int, layers:int, num_features:int):
  import tinycudann as tcnn

  return tcnn.NetworkWithInputEncoding(
    num_features + 3, 3,
    encoding_config=dict(
      otype = "composite",
      nested = [
        dict(otype = "SphericalHarmonics", 
            degree = 4, 
            n_dims_to_encode = 3
        ),
        
        dict(otype = "Identity",
            n_dims_to_encode = num_features)
      ]
    ), 
    
    network_config = dict(
      otype = "FullyFusedMLP",
      activation = "ReLU",
      output_activation = "None",
      n_neurons = hidden,
      n_hidden_layers = layers,
    )
  )




class EnvironmentModel(torch.nn.Module):
  def __init__(self, 
               num_cameras:int,
               
               image_features:int = 8, 
               hidden:int             = 32,
               layers:int             = 2):
    super().__init__()

    self.color_model = positional_model(hidden, layers, image_features)
    self.image_features = nn.Embedding(num_cameras, image_features)

    # Initialize image features with small standard deviation
    nn.init.normal_(self.image_features.weight, std=0.02)

  def forward(self, feature:torch.Tensor):
    return self.model(feature).to(torch.float32).sigmoid()
  

  def forward(self, dir:torch.Tensor, cam_idx:int):
    cam_feature = self.image_features[cam_idx]  
    cam_feature = cam_feature.unsqueeze(0).expand(dir.shape[0], -1)

    feature = torch.cat([dir, cam_feature], dim=1)
    return self.color_model(feature).to(torch.float32).sigmoid()



class ColorModel(torch.nn.Module):
  def __init__(self, 
               num_cameras:int,
               
               image_features:int = 8, 
               point_features:int       = 8,
               hidden:int             = 32,
               layers:int             = 2):
    
    super().__init__()

    self.color_model = positional_model(hidden, layers, image_features + point_features)
    self.image_features = nn.Embedding(num_cameras, image_features)

    # Initialize image features with small standard deviation
    nn.init.normal_(self.image_features.weight, std=0.02)


  def forward(self, point_features:torch.Tensor, positions:torch.Tensor, cam_pos:torch.Tensor, cam_idx:int):
    cam_feature = self.image_features[cam_idx]  
    cam_feature = cam_feature.unsqueeze(0).expand(positions.shape[0], -1)

    dir = F.normalize(positions.detach() - cam_pos)

    feature = torch.cat([dir, point_features, cam_feature], dim=1)
    return self.color_model(feature).to(torch.float32).sigmoid()
  
  def optimizer(self, lr_nn:float, lr_image_feature:float):

    param_groups = [
      dict(params=self.color_model.parameters(), lr=lr_nn, name="color_model"),
      dict(params=self.image_features.parameters(), lr=lr_image_feature, name="image_features")
    ]

    return torch.optim.Adam(param_groups, betas=(0.9, 0.999))