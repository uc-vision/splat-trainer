
import math
from numbers import Number
import torch


class Histogram:
  """ A histogram class for torch tensors,
  can be added with other histograms which have the same bins """

  def __init__(self,
               values,
               range=(0, 1),
               num_bins=10,
               trim=True):
    assert len(range) == 2

    self.range = range
    lower, upper = self.range

    bin_indexes = (values - lower) * num_bins / (upper - lower)
    bin_indexes = bin_indexes.long()

    if trim:
      valid = (bin_indexes >= 0) & (bin_indexes < num_bins)

      values = values[valid]
      bin_indexes = bin_indexes[valid]

    bin_indexes.clamp_(0, num_bins - 1)

    self.sum = values.sum().item()
    self.sum_squares = values.norm(2).item()
    self.counts = torch.bincount(bin_indexes, minlength=num_bins)

  @staticmethod
  def empty(range, num_bins, device:torch.device):
    return Histogram(torch.zeros((0,), device=device), range, num_bins)
  
  @property
  def device(self):
    return self.counts.device

  def __repr__(self):
    return self.counts.tolist().__repr__()

  @property
  def bins(self):
    lower, upper = self.range
    d = (upper - lower) / self.counts.size(0)

    return torch.tensor(
        [lower + i * d for i in range(0,
                                      self.counts.size(0) + 1)])
  @property
  def bin_width(self):
    lower, upper = self.range
    return (upper - lower) / self.counts.size(0)
  
  @property
  def bin_ranges(self):
    bins = self.bins()
    return torch.stack([bins[:-1], bins[1:]], dim=1)


  def __add__(self, other):
    assert isinstance(other, Histogram)
    assert other.counts.size(0) == self.counts.size(0) and (
        other.range == self.range), 'mismatched histogram sizes'

    total = Histogram.empty(range=self.range, num_bins=self.counts.size(0), device=self.device)
    total.sum = self.sum + other.sum
    total.sum_squares = self.sum_squares + other.sum_squares
    total.counts = self.counts + other.counts

    return total
  
  def append(self, values:torch.Tensor, trim=True):
    return self + Histogram(values, 
        range=self.range, num_bins=self.counts.size(0), trim=trim)

  def __truediv__(self, x):
    assert isinstance(x, Number)

    total = Histogram(range=self.range, num_bins=self.counts.size(0))
    total.sum = self.sum / x
    total.sum_squares = self.sum_squares / x
    total.counts = self.counts / x

    return total

  @property
  def std(self):

    n = self.counts.sum().item()
    if n > 1:
      sum_squares = self.sum_squares - (self.sum * self.sum / n)
      var = max(0, sum_squares / (n - 1))

      return math.sqrt(var)
    else:
      return 0

  @property
  def mean(self):
    n = self.counts.sum().item()
    if n > 0:
      return self.sum / self.counts.sum().item()
    else:
      return 0
