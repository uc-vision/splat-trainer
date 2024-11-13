import math
from typing import Tuple
import torch
import torch.nn.functional as F
import torch.nn as nn
from taichi_splatting.perspective import (CameraParams)

from splat_trainer.config import Varying, VaryingFloat, eval_varying, schedule_groups, schedule_lr


def color_model(layers:int, hidden_features:int, input_features:int, color_channels:int=3):
  import tinycudann as tcnn
  

  return tcnn.NetworkWithInputEncoding(
    input_features + 3, color_channels,
    encoding_config=dict(
      otype = "composite",
      nested = [
        dict(otype = "SphericalHarmonics", 
            degree = 4, 
            n_dims_to_encode = 3
        ),
        
        dict(otype = "Identity",
            n_dims_to_encode = input_features)
      ]
    ), 
    
    network_config = dict(
      otype = "FullyFusedMLP",
      activation = "ReLU",
      output_activation = "Sigmoid",
      n_neurons = hidden_features,
      n_hidden_layers = layers,
    )
  )

def mlp(layers:int, hidden_features:int, input_features:int, output_features:int):
  import tinycudann as tcnn

  return tcnn.Network(
    input_features, output_features,  
    network_config = dict(
      otype = "FullyFusedMLP",
      activation = "ReLU",
      output_activation = "None",
      n_neurons = hidden_features,
      n_hidden_layers = layers,
    )
  )


class EnvironmentModel(torch.nn.Module):
  def __init__(self, 
               num_cameras:int,
               
               image_features:int = 8, 
               hidden_features:int = 32,
               layers:int             = 2):
    super().__init__()

    self.color_model = color_model(layers=layers, hidden_features=hidden_features, input_features=image_features)
    self.image_features = nn.Parameter(torch.zeros(num_cameras, image_features))

    # Initialize image features with small standard deviation
    nn.init.normal_(self.image_features.weight, std=0.5)

  

  def forward(self, dir:torch.Tensor, cam_idx:int):
    cam_feature = self.image_features[cam_idx].unsqueeze(0)
    cam_feature = cam_feature.expand(dir.shape[0], -1)

    feature = torch.cat([dir, cam_feature], dim=1)
    return self.color_model(feature).to(torch.float32)



class ColorModel(torch.nn.Module):
  def __init__(self, 
               num_glo_embeddings:int,
               
               glo_features:int = 16, 
               point_features:int       = 16,

               hidden_features:int     = 64,
               layers:int             = 2,

               affine_model:bool    = True,
               color_channels:int     = 3):
    
    super().__init__()

    self.feature_size = glo_features + point_features
    
    self.affine_model = None
    if affine_model:
      self.affine_model = mlp(layers=layers, 
        hidden_features=hidden_features, 
        input_features=self.feature_size, 
        output_features=self.feature_size*2)
    
    self.color_model = color_model(layers=layers, 
        hidden_features=hidden_features, 
        input_features=self.feature_size, 
        color_channels=color_channels)

    self.glo_features = nn.Parameter(torch.zeros(num_glo_embeddings, glo_features))

    # Initialize image features with small standard deviation
    nn.init.normal_(self.glo_features, std=0.5)

  def evaluate_with_features(self, point_features:torch.Tensor, positions:torch.Tensor, cam_pos:torch.Tensor, glo_feature:torch.Tensor):
    glo_feature = glo_feature.unsqueeze(0).expand(positions.shape[0], -1)
    feature = torch.cat([point_features, glo_feature], dim=1)
    
    if self.affine_model is not None:
      affine_feature = self.affine_model(feature)
      log_scale, shift = torch.chunk(affine_feature, 2, dim=1)
      feature = torch.exp(log_scale.to(torch.float32)) * feature + shift

    dir = F.normalize(positions.detach() - cam_pos, dim=1)

    feature = torch.cat([dir, feature], dim=1)
    return self.color_model(feature).to(torch.float32)
  
  def lookup_camera(self, cam_idx:int):
    return self.glo_features[cam_idx]

  def forward(self, point_features:torch.Tensor, positions:torch.Tensor, cam_pos:torch.Tensor, cam_idx:int):
    return self.evaluate_with_features(point_features, positions, cam_pos, glo_feature=self.glo_features[cam_idx])

  
  def optimizer(self, lr_nn:VaryingFloat, lr_image_features:VaryingFloat):
    lr_nn = eval_varying(lr_nn, 0.)
    lr_image_features = eval_varying(lr_image_features, 0.)


    param_groups = [
      dict(params=self.color_model.parameters(), lr=lr_nn, name="nn"),
      dict(params=[self.glo_features], lr=lr_image_features, name="image_features")
    ]
    
    if self.affine_model is not None:
      param_groups.append(dict(params=self.affine_model.parameters(), lr=lr_nn, name="affine"))

    return torch.optim.Adam(param_groups, betas=(0.9, 0.999))
  
  def schedule(self, optimizer, lr_nn: VaryingFloat, lr_image_features: VaryingFloat, t:float):
    schedule_groups(dict(nn=lr_nn, image_features=lr_image_features), t, optimizer)