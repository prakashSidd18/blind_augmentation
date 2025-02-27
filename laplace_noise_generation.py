import utils
from laplace_pyramid import LaplacePyramidGenerator
from utils import load_image_torch, load_image_alphas_torch, save_image_torch, load_image_luma_torch
import torch, cv2, torchvision

def calc_local_stdevs(image, kernel_size=9, sigma=4.0):
    """
    Finds local standard deviations of an input image.
    
    Parameters
    ----------
    image : tensor
        13HW tensor containing the inpur image.
    kernel_size: int
        Dimension of kernel to use for computing local stdevs. Must be an odd positive integer.
    sigma: float
        Standard deviation of Gaussian kernel used for computing local stdevs.

    Returns
    -------
    stdevs: torch.Tensor
        The local standard deviation values. Tensor has same dimensions as input image.
    """
    assert(kernel_size % 2 == 1)
    #pad = torch.nn.ReflectionPad2d(kernel_size//2)
    blur = torchvision.transforms.GaussianBlur(kernel_size, sigma)
    variances = blur(image**2) - blur(image)**2
    variances[variances <= 0] = 1e-3
    stdevs = torch.sqrt(variances)
    return stdevs

def fit_noise_model(target_image, target_noise, pyr_generator, n_levels=4, kernel_size=9, sigma=4.0, sample_spacing_fullres=32):
    """
    Fits a linear model mapping from RGB intensities to local standard deviations of a laplacian pyramid.
    
    Parameters
    ----------
    target_image : tensor
        13HW tensor containing the noise-free intensity image.
    target_noise : tensor
        13HW tensor containing the noise to fit to.
    pyr_generator: LaplacePyramidGenerator instance
        Laplacian pyramid generator class instance.
    n_levels: int
        Number of pyramid levels to use. Must be greater than 2.
    kernel_size: int
        Dimension of kernel to use for computing local stdevs. Must be an odd positive integer.
    sigma: float
        Standard deviation of Gaussian kernel used for computing local stdevs at the highest resolution.
        This value is decreased proportionally when estimating stats at lower pyramid levels.
    sample_spacing_fullres: int
        Spacing between samples taken to fit model. To speed up the model fitting and avoid having to compute
        an enormous matrix pseudo-inverse, this is used to skip and only take one sample in every n-th row
        and column of the image.
        This is the skip value at the highest resolution, and will be halved to take the same total sample
        count at each pyramid level.

    Returns
    -------
    intensity_stat_mappings: List of torch.Tensor
        The fitted model from intensity to noise stats.
    """
    target_noise_pyr = pyr_generator.makeLaplacePyramid(target_noise, n_levels)
    target_noise_stats = []
    curr_sigma = sigma
    for i in range(n_levels-1):
        target_noise_stats.append(calc_local_stdevs(target_noise_pyr[i], kernel_size, curr_sigma))
        curr_sigma *= 0.5
    # Mappings are matrices that map from intensity values to stats at each pyramid level
    intensity_stat_mappings = [torch.ones(4,3) for l in target_noise_stats]
    sample_spacing = sample_spacing_fullres
    stat_sample_spacing = sample_spacing
    curr_sigma = sigma
    for l in range(len(intensity_stat_mappings)):
        subsample_image = target_image[:,:,::sample_spacing, ::sample_spacing]
        subsample_stats = target_noise_stats[l][:,:,::stat_sample_spacing, ::stat_sample_spacing]

        if subsample_image.shape[-1] > subsample_stats.shape[-1]:
            subsample_image = torch.nn.functional.interpolate(subsample_image, [subsample_stats.shape[-2], subsample_stats.shape[-1]], mode="area")

        intensity_matrix = subsample_image[0,...].flatten(1,-1).permute(1,0)
        intensity_matrix = torch.cat([intensity_matrix, torch.ones(intensity_matrix.shape[0], 1)], dim=-1)
        stat_matrix = subsample_stats[0,...].flatten(1,-1).permute(1,0)

        mult = intensity_matrix.permute(1,0) @ intensity_matrix
        mult_inv = torch.linalg.pinv(mult)

        soln = mult_inv @ intensity_matrix.permute(1,0) @ stat_matrix
        #print(soln)
        soln[soln < 0] = 0
        intensity_stat_mappings[l] = soln
        #print(intensity_matrix @ soln)
        stat_sample_spacing //= 2
        if stat_sample_spacing < 1:
            stat_sample_spacing = 1
        curr_sigma *= 0.5
    return intensity_stat_mappings

def fit_noise_model_luma_intercept(target_luma, target_noise, pyr_generator, n_levels=4, kernel_size=9, sigma=4.0, sample_spacing_fullres=32, constrain_nonnegative=True):
    """
    Fits a linear model mapping from RGB intensities to local standard deviations of a laplacian pyramid.
    
    Parameters
    ----------
    target_luma : tensor
        11HW tensor containing the noise-free luma map.
    target_noise : tensor
        13HW tensor containing the noise to fit to.
    pyr_generator: LaplacePyramidGenerator instance
        Laplacian pyramid generator class instance.
    n_levels: int
        Number of pyramid levels to use. Must be greater than 2.
    kernel_size: int
        Dimension of kernel to use for computing local stdevs. Must be an odd positive integer.
    sigma: float
        Standard deviation of Gaussian kernel used for computing local stdevs at the highest resolution.
        This value is decreased proportionally when estimating stats at lower pyramid levels.
    sample_spacing_fullres: int
        Spacing between samples taken to fit model. To speed up the model fitting and avoid having to compute
        an enormous matrix pseudo-inverse, this is used to skip and only take one sample in every n-th row
        and column of the image.
        This is the skip value at the highest resolution, and will be halved to take the same total sample
        count at each pyramid level.

    Returns
    -------
    intensity_stat_mappings: List of torch.Tensor
        The fitted model from intensity to noise stats.
    """
    target_noise_pyr = pyr_generator.makeLaplacePyramid(target_noise, n_levels)
    target_noise_stats = []
    curr_sigma = sigma
    for i in range(n_levels-1):
        target_noise_stats.append(calc_local_stdevs(target_noise_pyr[i], kernel_size, curr_sigma))
        curr_sigma *= 0.5
    # Mappings are matrices that map from intensity values to stats at each pyramid level
    intensity_stat_mappings = [torch.ones(2,3) for l in target_noise_stats]
    sample_spacing = sample_spacing_fullres
    stat_sample_spacing = sample_spacing
    curr_sigma = sigma
    for l in range(len(intensity_stat_mappings)):
        subsample_image = target_luma[:,:,::sample_spacing, ::sample_spacing]
        subsample_stats = target_noise_stats[l][:,:,::stat_sample_spacing, ::stat_sample_spacing]

        if subsample_image.shape[-1] > subsample_stats.shape[-1]:
            subsample_image = torch.nn.functional.interpolate(subsample_image, [subsample_stats.shape[-2], subsample_stats.shape[-1]], mode="area")

        intensity_matrix = subsample_image[0,...].flatten(1,-1).permute(1,0)
        intensity_matrix = torch.cat([intensity_matrix, torch.ones(intensity_matrix.shape[0], 1).to(intensity_matrix.device)], dim=-1)

        stat_matrix = subsample_stats[0,...].flatten(1,-1).permute(1,0)

        mult = intensity_matrix.permute(1,0) @ intensity_matrix
        mult_inv = torch.linalg.pinv(mult)

        soln = mult_inv @ intensity_matrix.permute(1,0) @ stat_matrix
        #print(soln)
        if(constrain_nonnegative):
            soln[soln < 0] = 0
        intensity_stat_mappings[l] = soln
        #print(intensity_matrix @ soln)
        stat_sample_spacing //= 2
        if stat_sample_spacing < 1:
            stat_sample_spacing = 1
        curr_sigma *= 0.5
    return intensity_stat_mappings
def fit_noise_model_luma(target_luma, target_noise, pyr_generator, n_levels=4, kernel_size=9, sigma=4.0, sample_spacing_fullres=32, constrain_nonnegative=True):
    """
    Fits a linear model mapping from RGB intensities to local standard deviations of a laplacian pyramid.
    
    Parameters
    ----------
    target_luma : tensor
        11HW tensor containing the noise-free luma map.
    target_noise : tensor
        13HW tensor containing the noise to fit to.
    pyr_generator: LaplacePyramidGenerator instance
        Laplacian pyramid generator class instance.
    n_levels: int
        Number of pyramid levels to use. Must be greater than 2.
    kernel_size: int
        Dimension of kernel to use for computing local stdevs. Must be an odd positive integer.
    sigma: float
        Standard deviation of Gaussian kernel used for computing local stdevs at the highest resolution.
        This value is decreased proportionally when estimating stats at lower pyramid levels.
    sample_spacing_fullres: int
        Spacing between samples taken to fit model. To speed up the model fitting and avoid having to compute
        an enormous matrix pseudo-inverse, this is used to skip and only take one sample in every n-th row
        and column of the image.
        This is the skip value at the highest resolution, and will be halved to take the same total sample
        count at each pyramid level.

    Returns
    -------
    intensity_stat_mappings: List of torch.Tensor
        The fitted model from intensity to noise stats.
    """
    target_noise_pyr = pyr_generator.makeLaplacePyramid(target_noise, n_levels)
    target_noise_stats = []
    curr_sigma = sigma
    for i in range(n_levels-1):
        target_noise_stats.append(calc_local_stdevs(target_noise_pyr[i], kernel_size, curr_sigma))
        curr_sigma *= 0.5
    # Mappings are matrices that map from intensity values to stats at each pyramid level
    intensity_stat_mappings = [torch.ones(1,3) for l in target_noise_stats]
    sample_spacing = sample_spacing_fullres
    stat_sample_spacing = sample_spacing
    curr_sigma = sigma
    for l in range(len(intensity_stat_mappings)):
        subsample_image = target_luma[:,:,::sample_spacing, ::sample_spacing]
        subsample_stats = target_noise_stats[l][:,:,::stat_sample_spacing, ::stat_sample_spacing]

        if subsample_image.shape[-1] > subsample_stats.shape[-1]:
            subsample_image = torch.nn.functional.interpolate(subsample_image, [subsample_stats.shape[-2], subsample_stats.shape[-1]], mode="area")

        intensity_matrix = subsample_image[0,...].flatten(1,-1).permute(1,0)
        stat_matrix = subsample_stats[0,...].flatten(1,-1).permute(1,0)

        mult = intensity_matrix.permute(1,0) @ intensity_matrix
        mult_inv = torch.linalg.pinv(mult)

        soln = mult_inv @ intensity_matrix.permute(1,0) @ stat_matrix
        #print(soln)
        if(constrain_nonnegative):
            soln[soln < 0] = 0
        intensity_stat_mappings[l] = soln
        #print(intensity_matrix @ soln)
        stat_sample_spacing //= 2
        if stat_sample_spacing < 1:
            stat_sample_spacing = 1
        curr_sigma *= 0.5
    return intensity_stat_mappings

def synthesise_noise(initial_noise, noise_free_image, intensity_stat_mappings, pyr_generator, kernel_size=9, sigma=4.0):
    """
    Given an initial random noise, a target noise-free intensity image and a model fitted by fit_noise_model,
    generates a noise map.
    
    Parameters
    ----------
    initial_noise : tensor
        13HW tensor containing an initial random (e.g. Gaussian) noise.
        I suggest generating using torch.randn().
    noise_free_image : tensor
        13HW tensor containing the image to generate noise for.
    intensity_stat_mappings : List of tensor
        Noise model, e.g. one fitted using fit_noise_model above.
    pyr_generator: LaplacePyramidGenerator instance
        Laplacian pyramid generator class instance.
    kernel_size: int
        Dimension of kernel to use for computing local stdevs. Must be an odd positive integer.
    sigma: float
        Standard deviation of Gaussian kernel used for computing local stdevs at the highest resolution.
        This value is decreased proportionally when estimating stats at lower pyramid levels.

    Returns
    -------
    noise_result : torch.Tensor
        The synthesised noise.
    """
    n_levels = len(intensity_stat_mappings) + 1
    noise_pyr = pyr_generator.makeLaplacePyramid(initial_noise, n_levels)
    noise_result = torch.zeros_like(noise_pyr[-2])

    # Make pyramid of noise, normalise to have stdev of 1 everywhere
    curr_sigma = sigma
    for i in range(len(noise_pyr)-1):
        stdevs = calc_local_stdevs(noise_pyr[i], kernel_size, curr_sigma)
        noise_pyr[i] /= stdevs
        curr_sigma *= 0.5

    target_gauss_pyr = pyr_generator.makeGaussPyramid(noise_free_image, n_levels)

    # Scale each level of the noise laplacian pyramid, and 
    for i in reversed(range(len(noise_pyr)-1)):
        curr_image = target_gauss_pyr[i]
        curr_intensities = curr_image[0,...].permute(1,2,0)
        curr_intensities = torch.cat([curr_intensities, torch.ones(curr_intensities.shape[0], curr_intensities.shape[1], 1)], dim=-1)
        curr_stdevs = curr_intensities @ intensity_stat_mappings[i]
        curr_stdevs = curr_stdevs.permute(2,0,1)[None,...]
        curr_noise = noise_pyr[i] * curr_stdevs
        noise_result += curr_noise
        if i != 0:
            noise_result = pyr_generator.pyrUp(noise_result)

    return noise_result

def synthesise_noise_luma(initial_noise, noise_free_luma, intensity_stat_mappings, pyr_generator, kernel_size=9, sigma=4.0):
    """
    Given an initial random noise, a target noise-free intensity image and a model fitted by fit_noise_model,
    generates a noise map.
    
    Parameters
    ----------
    initial_noise : tensor
        13HW tensor containing an initial random (e.g. Gaussian) noise.
        I suggest generating using torch.randn().
    noise_free_image : tensor
        11HW tensor containing the image to generate noise for.
    intensity_stat_mappings : List of tensor
        Noise model, e.g. one fitted using fit_noise_model above.
    pyr_generator: LaplacePyramidGenerator instance
        Laplacian pyramid generator class instance.
    kernel_size: int
        Dimension of kernel to use for computing local stdevs. Must be an odd positive integer.
    sigma: float
        Standard deviation of Gaussian kernel used for computing local stdevs at the highest resolution.
        This value is decreased proportionally when estimating stats at lower pyramid levels.

    Returns
    -------
    noise_result : torch.Tensor
        The synthesised noise.
    """
    n_levels = len(intensity_stat_mappings) + 1
    noise_pyr = pyr_generator.makeLaplacePyramid(initial_noise, n_levels)
    noise_result = torch.zeros_like(noise_pyr[-2])

    # Make pyramid of noise, normalise to have stdev of 1 everywhere
    curr_sigma = sigma
    for i in range(len(noise_pyr)-1):
        stdevs = calc_local_stdevs(noise_pyr[i], kernel_size, curr_sigma)
        noise_pyr[i] /= stdevs
        curr_sigma *= 0.5

    target_gauss_pyr = pyr_generator.makeGaussPyramid(noise_free_luma, n_levels)

    # Scale each level of the noise laplacian pyramid, and 
    for i in reversed(range(len(noise_pyr)-1)):
        curr_image = target_gauss_pyr[i]
        curr_intensities = curr_image[0,...].permute(1,2,0)
        curr_stdevs = curr_intensities @ intensity_stat_mappings[i]
        curr_stdevs = curr_stdevs.permute(2,0,1)[None,...]
        curr_noise = noise_pyr[i] * curr_stdevs
        noise_result += curr_noise
        if i != 0:
            noise_result = pyr_generator.pyrUp(noise_result)

    return noise_result

def synthesise_noise_luma_intercept(initial_noise, noise_free_luma, intensity_stat_mappings, pyr_generator, kernel_size=9, sigma=4.0):
    """
    Given an initial random noise, a target noise-free intensity image and a model fitted by fit_noise_model,
    generates a noise map.
    
    Parameters
    ----------
    initial_noise : tensor
        13HW tensor containing an initial random (e.g. Gaussian) noise.
        I suggest generating using torch.randn().
    noise_free_image : tensor
        11HW tensor containing the image to generate noise for.
    intensity_stat_mappings : List of tensor
        Noise model, e.g. one fitted using fit_noise_model above.
    pyr_generator: LaplacePyramidGenerator instance
        Laplacian pyramid generator class instance.
    kernel_size: int
        Dimension of kernel to use for computing local stdevs. Must be an odd positive integer.
    sigma: float
        Standard deviation of Gaussian kernel used for computing local stdevs at the highest resolution.
        This value is decreased proportionally when estimating stats at lower pyramid levels.

    Returns
    -------
    noise_result : torch.Tensor
        The synthesised noise.
    """
    n_levels = len(intensity_stat_mappings) + 1
    noise_pyr = pyr_generator.makeLaplacePyramid(initial_noise, n_levels)
    noise_result = torch.zeros_like(noise_pyr[-2])

    # Make pyramid of noise, normalise to have stdev of 1 everywhere
    curr_sigma = sigma
    for i in range(len(noise_pyr)-1):
        stdevs = calc_local_stdevs(noise_pyr[i], kernel_size, curr_sigma)
        noise_pyr[i] /= stdevs
        curr_sigma *= 0.5

    target_gauss_pyr = pyr_generator.makeGaussPyramid(noise_free_luma, n_levels)

    # Scale each level of the noise laplacian pyramid, and 
    for i in reversed(range(len(noise_pyr)-1)):
        curr_image = target_gauss_pyr[i]
        curr_intensities = curr_image[0,...].permute(1,2,0)
        curr_intensities = torch.cat([curr_intensities, torch.ones(curr_intensities.shape[0], curr_intensities.shape[1], 1).to(curr_intensities.device)], dim=-1)
        curr_stdevs = curr_intensities @ intensity_stat_mappings[i]
        curr_stdevs = curr_stdevs.permute(2,0,1)[None,...]
        curr_noise = noise_pyr[i] * curr_stdevs
        noise_result += curr_noise
        if i != 0:
            noise_result = pyr_generator.pyrUp(noise_result)

    return noise_result

if __name__ == "__main__":

    def test1():
        pyr_generator = LaplacePyramidGenerator()

        image = load_image_torch("images/duplo_blocks_denoised.png")
        noisy_image = load_image_torch("images/duplo_blocks_noisy.png")

        target_noise = noisy_image - image

        #model = fit_noise_model(image, target_noise, pyr_generator, sample_spacing_fullres=1)
        model = fit_noise_model(image, target_noise, pyr_generator)
        for matrix in model:
            print(matrix)

        initial_noise = torch.randn_like(target_noise)

        noise_result = synthesise_noise(initial_noise, image, model, pyr_generator)

        import matplotlib.pyplot as plt
        plt.subplot(2,2,1)
        plt.title("Image with synth noise")
        plt.imshow((image + noise_result)[0,...].permute(1,2,0).numpy())
        plt.subplot(2,2,2)
        plt.title("Image with real noise")
        plt.imshow((noisy_image)[0,...].permute(1,2,0).numpy())
        plt.subplot(2,2,3)
        plt.title("Synth noise")
        plt.imshow((noise_result)[0,...].permute(1,2,0).numpy() + 0.5)
        plt.subplot(2,2,4)
        plt.title("Real noise")
        plt.imshow((target_noise)[0,...].permute(1,2,0).numpy() + 0.5)
        plt.show()

        save_image_torch("output/our_noisy_image.png", image + noise_result)
        save_image_torch("output/GT_noisy_image.png", noisy_image)
        save_image_torch("output/synth_noise.png", noise_result + 0.5)
        save_image_torch("output/GT_noise.png", target_noise + 0.5)

    def test2():
        pyr_generator = LaplacePyramidGenerator()

        image = load_image_torch("data/denoised/flir_noisy_rainbowballmotion/frame_000047.png")
        noisy_image = load_image_torch("data/original/flir_noisy_rainbowballmotion/frame_000047.png")
        virtual = load_image_torch("data/composites/flir_noisy_rainbowballmotion/0047_with_bg.png")

        virtual_alphas = load_image_alphas_torch("data/composites/flir_noisy_rainbowballmotion/0047.png")

        target_noise = noisy_image - image

        model = fit_noise_model(image, target_noise, pyr_generator)

        initial_noise = torch.randn_like(target_noise)

        noise_result = synthesise_noise(initial_noise, virtual, model, pyr_generator)

        noisy_virtual = virtual + noise_result

        naive_composite = virtual_alphas * virtual + (1 - virtual_alphas) * noisy_image
        our_composite = virtual_alphas * noisy_virtual + (1 - virtual_alphas) * noisy_image

        import matplotlib.pyplot as plt
        plt.subplot(2,2,1)
        plt.title("Virtual with synth noise")
        plt.imshow((noisy_virtual)[0,...].permute(1,2,0).numpy())
        plt.subplot(2,2,2)
        plt.title("Naive composite")
        plt.imshow((naive_composite)[0,...].permute(1,2,0).numpy())
        plt.subplot(2,2,3)
        plt.title("Our composite")
        plt.imshow((our_composite)[0,...].permute(1,2,0).numpy())
        plt.subplot(2,2,4)
        plt.title("Synth noise")
        plt.imshow((noise_result)[0,...].permute(1,2,0).numpy() + 0.5)
        plt.show()

        save_image_torch("output/naive_composite.png", naive_composite)
        save_image_torch("output/our_composite.png", our_composite)
        save_image_torch("output/synth_noise.png", noise_result + 0.5)
        save_image_torch("output/GT_noise.png", target_noise + 0.5)

    def test3():

        pyr_generator = LaplacePyramidGenerator()

        #image = load_image_torch("data/denoised/flir_noisy_rainbowballmotion/frame_000047.png")
        #noisy_image = load_image_torch("data/original/flir_noisy_rainbowballmotion/frame_000047.png")
        #image_luma = load_image_luma_torch("data/denoised/flir_noisy_rainbowballmotion/frame_000047.png")

        # image = load_image_torch("data/denoised/nikon_varying_ISO/frame_000633.png")
        # noisy_image = load_image_torch("data/original/nikon_varying_ISO/frame_000633.png")
        # image_luma = load_image_luma_torch("data/denoised/nikon_varying_ISO/frame_000633.png")
        # image = load_image_torch("data/denoised/nikon_varying_ISO/frame_000250.png")
        # noisy_image = load_image_torch("data/original/nikon_varying_ISO/frame_000250.png")
        # image_luma = load_image_luma_torch("data/denoised/nikon_varying_ISO/frame_000250.png")


        image = load_image_torch("data/denoised/image2.png")
        noisy_image = load_image_torch("data/original/image2.png")
        image_luma = load_image_luma_torch("data/denoised/image2.png")

        #image = load_image_torch("data/denoised/image_1.png")
        #noisy_image = load_image_torch("data/original/image_1.png")
        #image_luma = load_image_luma_torch("data/denoised/image_1.png")

        target_noise = noisy_image - image

        #model = fit_noise_model(image, target_noise, pyr_generator, sample_spacing_fullres=1)
        model = fit_noise_model_luma_intercept(image_luma, target_noise, pyr_generator, n_levels=6, constrain_nonnegative=False)
        for matrix in model:
            print(matrix)

        initial_noise = torch.randn_like(target_noise)

        noise_result = synthesise_noise_luma_intercept(initial_noise, image_luma, model, pyr_generator)
        synth_image = image + noise_result

        import matplotlib.pyplot as plt
        plt.subplot(2,2,1)
        plt.title("Image with synth noise")
        plt.imshow((synth_image)[0,...].cpu().permute(1,2,0).numpy())
        plt.subplot(2,2,2)
        plt.title("Image with real noise")
        plt.imshow((noisy_image)[0,...].cpu().permute(1,2,0).numpy())
        plt.subplot(2,2,3)
        plt.title("Synth noise")

        plt.imshow((noise_result+0.5)[0,...].permute(1,2,0).numpy())
        plt.subplot(2,2,4)
        plt.title("Real noise")
        plt.imshow((target_noise+0.5)[0,...].permute(1,2,0).numpy())
        plt.show()

        save_image_torch("output/our_noisy_image.png", synth_image)
        save_image_torch("output/GT_noisy_image.png", noisy_image)
        save_image_torch("output/synth_noise.png", noise_result + 0.5)
        save_image_torch("output/GT_noise.png", target_noise + 0.5)

    #test1()
    #test2()
    test3()
