import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor, ToPILImage
import numpy as np
from PIL import Image

import pdb


def gaussian_weight(ksize, sigma):
    center = ksize // 2
    x = np.arange(ksize, dtype=np.float32) - center
    kernel_1d = np.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel = kernel_1d[..., None] @ kernel_1d[None, ...]
    kernel = torch.from_numpy(kernel)
    kernel = kernel / kernel.sum()  # Normalization
    return kernel


class GaussianFilter(nn.Module):
    def __init__(self, ksize=5, sigma=None):
        super(GaussianFilter, self).__init__()
        if ksize % 2 == 0:
            ksize = ksize + 1
        self.ksize = ksize
        if isinstance(sigma, (int, float)):
            self.sigma = sigma
        else:
            self.sigma = 0.3 * ((ksize - 1) / 2.0 - 1) + 0.8

        weight = gaussian_weight(self.ksize, self.sigma)
        # Reshape to 2d depthwise convolutional weight
        weight = weight.view(1, 1, ksize, ksize).repeat(3, 1, 1, 1)

        # Create gaussian filter as convolutional layer
        pad = (ksize - 1) // 2
        self.filter = nn.Conv2d(3, 3, ksize, stride=1, padding=pad, groups=3, bias=False, padding_mode="reflect")
        self.filter.weight.data = weight
        self.filter.weight.requires_grad = False

    def forward(self, x):
        return self.filter(x)


class BilateralFilter(nn.Module):
    def __init__(self, ksize=5, sigma_space=None, sigma_density=None):
        super(BilateralFilter, self).__init__()
        if ksize % 2 == 0:
            ksize = ksize + 1
        self.ksize = ksize
        if isinstance(sigma_space, (int, float)):
            self.sigma_space = sigma_space
        else:
            self.sigma_space = 0.3 * ((ksize - 1) / 2.0 - 1) + 0.8
        if isinstance(sigma_density, (int, float)):
            self.sigma_density = sigma_density
        else:
            self.sigma_density = 0.3 * ((ksize - 1) / 2.0 - 1) + 0.8

        self.pad = (ksize - 1) // 2
        self.weight_space = gaussian_weight(self.ksize, self.sigma_space)

    def forward(self, x):
        x_pad = F.pad(x, pad=[self.pad, self.pad, self.pad, self.pad], mode="reflect")
        x_patches = x_pad.unfold(2, self.ksize, 1).unfold(3, self.ksize, 1)

        # Calculate the 2-dimensional gaussian kernel
        diff_density = x_patches - x.unsqueeze(-1).unsqueeze(-1)
        weight_density = torch.exp(-(diff_density ** 2) / (2 * self.sigma_density ** 2))
        weight_density /= weight_density.sum(dim=(-1, -2), keepdim=True)  # Normalization

        # Keep same shape with weight_density
        weight_space_dim = (x_patches.dim() - 2) * (1,) + (self.ksize, self.ksize)
        weight_space = self.weight_space.view(*weight_space_dim).expand_as(weight_density)
        weight_space = weight_space.to(x.device)

        # Get the final kernel weight
        weight = weight_density * weight_space
        weight_sum = weight.sum(dim=(-1, -2))

        return (weight * x_patches).sum(dim=(-1, -2)) / weight_sum


if __name__ == "__main__":
    device = torch.device("cuda")

    img = Image.open("lena.png").convert("RGB")
    img = ToTensor()(img).unsqueeze(0).to(device)

    model = BilateralFilter()
    # model = GaussianFilter()
    model = model.eval()
    model = model.to(device)

    with torch.no_grad():
        output = model(img)

    output = output.detach().squeeze(0).cpu()
    ToPILImage()(output).save("./test.png")
