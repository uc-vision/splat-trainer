
# Copyright 2024 Yuehao Wang (https://github.com/yuehaowang). 
# This part of code is borrowed form ["Bilateral Guided Radiance Field Processing"](https://bilarfpro.github.io/).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch


def mse_to_psnr(mse):
  return 10 * torch.log10(1 / mse)  

def compute_psnr(a, b):
  return mse_to_psnr(torch.nn.functional.mse_loss(a, b))  



def fit_colors(pred_image:torch.Tensor, source_image:torch.Tensor) -> torch.Tensor:
  """ Fit an affine color transform between the two images and return corrected image """
  return fit_colors_batch(pred_image.unsqueeze(0), source_image.unsqueeze(0)).squeeze(0)

def fit_colors_batch(
    img: torch.Tensor, ref: torch.Tensor, num_iters: int = 5, eps: float = 0.5 / 255
) -> torch.Tensor:
    """
    Warp `img` to match the colors in `ref_img` using iterative color matching.

    This function performs color correction by warping the colors of the input image
    to match those of a reference image. It uses a least squares method to find a
    transformation that maps the input image's colors to the reference image's colors.

    The algorithm iteratively solves a system of linear equations, updating the set of
    unsaturated pixels in each iteration. This approach helps handle non-linear color
    transformations and reduces the impact of clipping.

    Args:
        img (torch.Tensor): Input image to be color corrected. Shape: [..., num_channels]
        ref (torch.Tensor): Reference image to match colors. Shape: [..., num_channels]
        num_iters (int, optional): Number of iterations for the color matching process.
                                   Default is 5.
        eps (float, optional): Small value to determine the range of unclipped pixels.
                               Default is 0.5 / 255.

    Returns:
        torch.Tensor: Color corrected image with the same shape as the input image.

    Note:
        - Both input and reference images should be in the range [0, 1].
        - The function works with any number of channels, but typically used with 3 (RGB).
    """
    if img.shape[-1] != ref.shape[-1]:
        raise ValueError(
            f"img's {img.shape[-1]} and ref's {ref.shape[-1]} channels must match"
        )
    num_channels = img.shape[-1]
    img_mat = img.reshape([-1, num_channels])
    ref_mat = ref.reshape([-1, num_channels])

    def is_unclipped(z):
        return (z >= eps) & (z <= 1 - eps)  # z \in [eps, 1-eps].

    mask0 = is_unclipped(img_mat)
    # Because the set of saturated pixels may change after solving for a
    # transformation, we repeatedly solve a system `num_iters` times and update
    # our estimate of which pixels are saturated.
    for _ in range(num_iters):
        # Construct the left hand side of a linear system that contains a quadratic
        # expansion of each pixel of `img`.
        a_mat = []
        for c in range(num_channels):
            a_mat.append(img_mat[:, c : (c + 1)] * img_mat[:, c:])  # Quadratic term.
        a_mat.append(img_mat)  # Linear term.
        a_mat.append(torch.ones_like(img_mat[:, :1]))  # Bias term.
        a_mat = torch.cat(a_mat, dim=-1)
        warp = []
        for c in range(num_channels):
            # Construct the right hand side of a linear system containing each color
            # of `ref`.
            b = ref_mat[:, c]
            # Ignore rows of the linear system that were saturated in the input or are
            # saturated in the current corrected color estimate.
            mask = mask0[:, c] & is_unclipped(img_mat[:, c]) & is_unclipped(b)
            ma_mat = torch.where(mask[:, None], a_mat, torch.zeros_like(a_mat))
            mb = torch.where(mask, b, torch.zeros_like(b))
            w = torch.linalg.lstsq(ma_mat, mb, rcond=-1)[0]
            assert torch.all(torch.isfinite(w))
            warp.append(w)
        warp = torch.stack(warp, dim=-1)
        # Apply the warp to update img_mat.
        img_mat = torch.clip(torch.matmul(a_mat, warp), 0, 1)
    corrected_img = torch.reshape(img_mat, img.shape)
    return corrected_img