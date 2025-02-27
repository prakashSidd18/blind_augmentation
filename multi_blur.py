import torch
from laplace_pyramid import LaplacePyramidGenerator
import math

class MultiBlur():
    def __init__(self, device=torch.device("cpu")):
        self.laplace = LaplacePyramidGenerator(device)
    def set_image(self, image, max_blur_sigma=-1):
        pyr_depth = int(math.floor(min(math.log2(image.shape[-1])+1, math.log2(image.shape[-2])+1)))
        if max_blur_sigma != -1:
            max_blur_lod = int(math.ceil(self.get_lod_level(max_blur_sigma)))
            pyr_depth = max_blur_lod + 1
        gauss_pyr = self.laplace.makeGaussPyramid(image, pyr_depth)
        self.fullres_gauss_pyr = []
        for l in range(len(gauss_pyr)):
            upsampled_level = gauss_pyr[l]
            for u in range(l):
                upsampled_level = self.laplace.pyrUp(upsampled_level)
            self.fullres_gauss_pyr.append(upsampled_level)
    def get_lod_level(self, sigma):
        if sigma > 1.675:
            return math.log2(sigma / 1.675) + 1
        else:
            return sigma / 1.675
    def blur(self, sigma):
        lod_level = self.get_lod_level(sigma)
        hf_level = math.floor(lod_level)
        lf_level = hf_level + 1
        alpha = lod_level - hf_level
        if lf_level >= len(self.fullres_gauss_pyr):
            print("WARNING: Pyramid depth unable to support blur of this magnitude.")
            return self.fullres_gauss_pyr[-1]
        return self.fullres_gauss_pyr[hf_level] * (1 - alpha) + \
            self.fullres_gauss_pyr[lf_level] * alpha

if __name__ == "__main__":
    from utils import load_image_torch
    import matplotlib.pyplot as plt
    import torchvision
    import numpy as np

    print(math.log2(2))
    print(math.log2(1))

    image = load_image_torch("images/duplo_blocks_noisy.png")

    blur = MultiBlur()

    blur.set_image(image, 10)

    sigmas = np.linspace(0, 15, 50)
    lod_levels = np.zeros_like(sigmas)
    for i, sigma in enumerate(sigmas):
        lod_levels[i] = blur.get_lod_level(sigma)
        
    plt.plot(sigmas, lod_levels)
    plt.show()


    blurred_image = blur.blur(16)

    for sigma in np.linspace(15, 17, 10):
        blurred = torchvision.transforms.GaussianBlur(kernel_size=51, sigma=sigma)(image)
        diff = torch.nn.MSELoss()(blurred_image, blurred)
        print(sigma, diff)

    exit(0)

    plt.subplot(1,2,1)
    plt.imshow(image[0,...].permute(1,2,0).detach().cpu().numpy())
    plt.subplot(1,2,2)
    plt.imshow(blurred_image[0,...].permute(1,2,0).detach().cpu().numpy())
    plt.show()



