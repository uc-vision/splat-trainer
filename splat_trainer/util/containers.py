
from typing import TypeVar

import numpy as np


T = TypeVar("T")

def transpose_rows(rows: dict[str, list[T]]) -> dict[str, list[T]]:
    return {key: [row[key] for row in rows] for key in rows[0]}

def mean_rows(rows: dict[str, list[T]]) -> dict[str, T]:
  d = transpose_rows(rows)
  return {k:np.mean(v) for k, v in d.items()}

def sum_rows(rows: dict[str, list[T]]) -> dict[str, T]:
  d = transpose_rows(rows)
  return {k:np.sum(v) for k, v in d.items()}


def replace_dict(d, **kwargs):
  return {**d, **kwargs}