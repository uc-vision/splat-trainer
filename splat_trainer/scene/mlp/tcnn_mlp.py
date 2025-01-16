import tinycudann as tcnn

def directional_mlp(inputs: int, outputs: int, hidden: int, layers: int, sh_degree: int):
  return tcnn.NetworkWithInputEncoding(
    inputs + 3, outputs,
    encoding_config=dict(
      otype = "composite",
      nested = [
        dict(otype = "SphericalHarmonics", 
            degree = sh_degree + 1, 
            n_dims_to_encode = 3
        ),
        
        dict(otype = "Identity",
            n_dims_to_encode = inputs)
      ]
    ), 
    
    network_config = dict(
      otype = "FullyFusedMLP",
      activation = "ReLU",
      output_activation = "Sigmoid",
      n_neurons = hidden,
      n_hidden_layers = layers,
    )
  )



def basic_mlp(inputs: int, outputs: int, hidden: int, layers: int):
  return tcnn.Network(
    inputs, outputs,
    network_config=dict(
      otype = "FullyFusedMLP",
      activation = "ReLU",
      output_activation = "Sigmoid",
      n_neurons = hidden,
      n_hidden_layers = layers,
    )
  )