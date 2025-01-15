import json
from numbers import Number
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, Mapping, Sequence
from beartype import beartype
import pandas as pd
from splat_trainer.logger import Logger

from tensordict import TensorDict

import torch

def log_histograms(logger:Logger, name:str, value:Any, step:int):
  if isinstance(value, torch.Tensor):
    value = value.detach()
    logger.log_histogram(f"{name}/histogram", value, step)

    if value.grad is not None:
      log_histograms(logger, f"{name}/grad", value.grad, step)

  if isinstance(value, dict):
    keys = sorted(value.keys()) 
    for k in keys:
      log_histograms(logger, f"{name}/{k}", value[k], step)

  if isinstance(value, tuple) or isinstance(value, list):
    for k, v in enumerate(value):
      log_histograms(logger, f"{name}/{k}", v, step)



def value_summary(value:Any, state:dict):
  if isinstance(value, torch.Tensor):
    return tuple([*value.shape, value.dtype])

  if isinstance(value, Mapping):
    return {k: value_summary(v, state) for k, v in value.items()}

  if isinstance(value, list) or isinstance(value, tuple): 
    return [value_summary(v, state) for v in value]


  return value

def without_key(d:dict, key:str):
  return {k: v for k, v in d.items() if k != key}

def optimizer_summary(optimizer:torch.optim.Optimizer):
  return {group['name']: value_summary(without_key(group, 'name'), optimizer.state) for group in optimizer.param_groups}


def print_summary(optimizer:torch.optim.Optimizer):
  opt_summary = optimizer_summary(optimizer)
  pprint(opt_summary)
  

@beartype
def log_optimizer(name:str, optimizer:torch.optim.Optimizer, logger:Logger, step:int):
  """Log hyperparameters, parameters and states of the optimizer"""
  opt_summary = optimizer_summary(optimizer)

  logger.log_json(f"{name}/summary", opt_summary, step)


def dump_optimizer(optimizer:torch.optim.Optimizer, filename:Path):
  opt_summary = optimizer_summary(optimizer)
  filename.parent.mkdir(parents=True, exist_ok=True)
  with open(filename, "w") as f:
    json.dump(opt_summary, f, indent=2)



def compare_tensors(x:TensorDict, y:TensorDict, rtol:float=1e-5, atol:float=1e-8):
  x = x.flatten_keys()
  y = y.flatten_keys()

  assert x.keys() == y.keys(), f"Keys do not match {list(x.keys())} != {list(y.keys())}"
  problems = []

  for k, v in x.items():
    if not torch.allclose(v, y[k], rtol=rtol, atol=atol):
      max_rel = torch.max(torch.abs(v - y[k]) / (torch.abs(y[k]) + atol))
      max_abs = torch.max(torch.abs(v - y[k]))
      problems.append(f"{k}: rel={max_rel:.4e}, abs={max_abs:.4e}")
      
  if len(problems) > 0:
    raise ValueError("\n".join(problems))
  
def optimizer_state(optimizer:torch.optim.Optimizer):
  state = {}
  for group in optimizer.param_groups:
    for i, p in enumerate(group['params']):
      state[f"{group['name']}.{i}"] = optimizer.state[p]
  return TensorDict(state)


def compare_optimizers(x:torch.optim.Optimizer, y:torch.optim.Optimizer):
  x_state = optimizer_state(x)
  y_state = optimizer_state(y)

  return compare_tensors(x_state, y_state)


def print_table(x:pd.DataFrame, sig_figs: int = 4):
  print(x.to_string(float_format=lambda x: f'{{:.{sig_figs}e}}'.format(x) if abs(x) < 1e-3 
                    else f'{{:.{sig_figs}g}}'.format(x)))


def print_stats(x:TensorDict):
  x = x.flatten_keys()

  def stats(v:torch.Tensor):
    v = v.float()
    return [v.abs().mean().item(), v.std().item(), v.min().item(), v.max().item()]

  k = sorted(x.keys())
  stats_table = []
  for k in k:
    v = x[k]
    stats_table.append([k, *stats(v)])

    if v.grad is not None:
      stats_table.append([f"{k}.grad", *stats(v.grad)])

  df = pd.DataFrame(stats_table, columns=["param", "mean", "std", "min", "max"])
  print_table(df)


def print_params(optimizer:torch.optim.Optimizer):
  
  summary = optimizer_summary(optimizer)
  df = pd.DataFrame(summary)

  print_table(df)

  