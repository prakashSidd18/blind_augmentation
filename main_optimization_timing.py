
import os

import imageio
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import utils
import io_module
import optimization_module

import blur_utils
from laplace_pyramid import LaplacePyramidGenerator
import laplace_noise_generation as laplace_noise_utils

if torch.cuda.is_available():
    device = torch.device("cuda")
    cuda = True
else:
    device = torch.device("cpu")
    cuda = False


os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
imageio.plugins.freeimage.download()

# Toggle visualization and model saving to remove IO overhead
visualize = False
save_model = False

# Run optimization over every `n` frames and report mean timings; default n=10
skip_frames = 10

# Select (uncomment) the scenes to test
dataset_frame_id = {
                    'flir_noisy_rainbowballmotion': {'opt_frame': 5, 'dilation': True, 'erosion': True, 'debug_frame': 26,
                                                     'noise': True, 'mb': True, 'dof': True, 'downsize': False},
                    'flir_noisy_greenballmotion': {'opt_frame': 15, 'dilation': True, 'erosion': True, 'debug_frame': 18,
                                                   'noise': True, 'mb': True, 'dof': True, 'downsize': False},
                    'rainbow_ball_2': {'opt_frame': 13, 'dilation': False, 'erosion': True, 'debug_frame': 14,
                                       'noise': True, 'mb': True, 'dof': True, 'downsize': False},
                    'flir_dof_mb_greenballmotion_scene1_1': {'opt_frame': 15, 'dilation': True, 'erosion': True, 'debug_frame': 46,
                                                             'noise': True, 'mb': True, 'dof': True, 'downsize': False},
                    'flir_noisy_scleich_2': {'opt_frame': 33, 'dilation': True, 'erosion': True, 'debug_frame': 93,
                                             'noise': True, 'mb': True, 'dof': True, 'downsize': False},
                    }



for count, dataset in enumerate(dataset_frame_id.keys()):

    print('############# {} #############'.format(dataset))
    opt_defocus_blur_model = dataset_frame_id[dataset]['dof']
    opt_motion_blur_model = dataset_frame_id[dataset]['mb']
    opt_noise_model = dataset_frame_id[dataset]['noise']
    downsize_dataset = dataset_frame_id[dataset]['downsize']
    # =====================================================================================================================
    '''Get input frames'''
    img_filepath = r"data/original/{}.png".format(dataset)
    img_abspath = os.path.abspath(img_filepath)
    img_name = os.path.basename(img_filepath).split('.')[0]
    img_dir = os.path.join(os.path.dirname(img_abspath), img_name)

    # Initialize IO module for optimization
    image_data = io_module.IO(img_filepath=img_filepath, scale_down=downsize_dataset, single_frame_mode=False,
                              frame_id=dataset_frame_id[img_name]['opt_frame'], debug=False, skip_frames=skip_frames)

    # Read input frames
    input_frames = image_data.read_frames()
    numFrames = input_frames['numFrames']

    # Initialize optimizer
    optimizer = optimization_module.Optimization(image_data=image_data,
                                                 opt_noise_model=opt_noise_model,
                                                 opt_motion_blur_model=opt_motion_blur_model,
                                                 opt_defocus_blur_model=opt_defocus_blur_model,
                                                 visualize=visualize,
                                                 save_model=save_model)

    if visualize:
        utils.create_folder(img_name)


    noise_viz_frames = []
    gen_noise_viz_frames = []
    gt_noise_viz_frames_optim = []
    gt_noise_viz_frames = []
    mb_params = np.zeros((numFrames, 1))
    mb_viz_frames = []
    dof_viz_frames = []
    original_viz_frames = []

    noise_intensity_multiplier = 1
    dof_minima = 0.0
    dof_maxima = 10.0
    # =================================================================================================================
    '''Optimization!!'''
    print('######### {}x{} #########'.format(input_frames['width'], input_frames['height']))
    for frame_id in range(0, numFrames, skip_frames):
        '''Run optimization on current input frame'''
        optimizer.init_opt_models(img_name, frame_id)
        original_image = optimizer.original_image
        fName = optimizer.fName
        img_channel, img_height, img_width = original_image.size()

        if visualize:
            original_viz_frames.append(original_image)

        path = os.path.join("./output", img_name + "_result_", "{}/{}_blur_parameters.npy".format(fName, img_name))

        fNameNoise = '/{}/synth_Noise_image_new'.format(fName)
        noise_ver = '/{}/noise_generated_new'.format(fName)
        if not optimizer.load_opt_model(path):
            optimizer.run_optimization()
            fNameNoise = '/{}/synth_Noise_image_original'.format(fName)
            noise_ver = '/{}/noise_generated_original'.format(fName)

        # --------------------------------------------------------------------------------------------

        if opt_noise_model and visualize:
            '''### Noise Optimization Viz ###'''
            # Generate image noise in shape of original noise
            # input_noise, generated_noise = noise_utils.eval_noise_model(optimizer.noise_image,
            #                                                             optimizer.grayscale_image,
            #                                                             optimizer.noise_model)

            pyr_generator = LaplacePyramidGenerator()
            denoised_img = optimizer.denoised_image.cpu().float()[None, ...]
            grayscale_img = optimizer.grayscale_image.cpu().float()[None, None, :, :]
            input_noise = torch.randn_like(optimizer.noise_image[None, ...]).cpu()
            generated_noise = laplace_noise_utils.synthesise_noise_luma_intercept(input_noise,
                                                                                  grayscale_img,
                                                                                  optimizer.noise_model,
                                                                                  pyr_generator)[0].to(device)

            noise_added_image = (optimizer.denoised_image + generated_noise).clip(0, 1)

            utils.save_image(generated_noise, img_name=img_name, file_desc=noise_ver, fmt='exr')

            # clip noise image to save as png
            optimizer.noise_image = utils.tonemap_noise(optimizer.noise_image) * noise_intensity_multiplier
            utils.save_image(optimizer.noise_image + 0.5, img_name=img_name, file_desc='/{}/noise'.format(fName), fmt='png')
            generated_noise = utils.tonemap_noise(generated_noise) * noise_intensity_multiplier
            utils.save_image(optimizer.noise_image, img_name=img_name, file_desc='/{}/noise_tm'.format(fName), fmt='png')
            utils.save_image(generated_noise, img_name=img_name, file_desc=noise_ver, fmt='png')
            utils.save_image(noise_added_image, img_name=img_name, file_desc=fNameNoise, fmt='png')

            if visualize:
                noise_viz_frames.append(noise_added_image)
                gen_noise_viz_frames.append(generated_noise)
                gt_noise_viz_frames_optim.append(optimizer.noise_image)

        # --------------------------------------------------------------------------------------------
        if opt_motion_blur_model and visualize:
            '''### Motion Blur Optimization Viz ###'''
            if optimizer.optimized:
                optimizer.flow_image = torch.from_numpy(optimizer.flow_image).permute(2, 0, 1).float().to(device)
            flow_image = optimizer.motion_blur_model.item() * optimizer.flow_image
            motion_blur_synthesized_image = blur_utils.motion_blur_optim(optimizer.demotion_blurred_image, flow_image,
                                                                         optimizer.kernel_size)
            utils.save_image(motion_blur_synthesized_image, img_name=img_name,
                            file_desc='/{}/synth_MB_image_{:.3f}'.format(fName, optimizer.motion_blur_model), fmt='png')

            mb_params[frame_id] = optimizer.motion_blur_model.item()
            px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
            fig, ax = plt.subplots(figsize=(img_width*px, img_height*px), facecolor='white')
            ax.set_facecolor("white")
            ax.plot(range(numFrames), mb_params, label='Estimated Exposure')
            plt.xlabel('Frame ID')
            plt.ylabel('Exposure')
            plt.ylim(0, 1)
            plt.legend()
            fig.canvas.draw()

            # Convert the canvas to a raw RGB buffer
            buf = fig.canvas.tostring_rgb()
            ncols, nrows = fig.canvas.get_width_height()
            mb_fig = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3) / 255.0
            # mb_fig = np.transpose(mb_fig, (2, 0, 1))
            mb_viz_frames.append(mb_fig.astype('float32'))

            plt.close()

        # --------------------------------------------------------------------------------------------
        if opt_defocus_blur_model:
            '''### Defocus Blur Optimization Viz ###'''
            if optimizer.optimized:
                dof = optimizer.parameters.item()['dof']
                depth = optimizer.parameters.item()['depth']
                sorted_rad = optimizer.parameters.item()['sorted_rad']
            else:
                dof = optimizer.parameters['dof']
                depth = optimizer.parameters['depth']
                sorted_rad = optimizer.parameters['sorted_rad']

            x = torch.linspace(0, 1, 1000).to(device)

            with torch.no_grad():
                y = optimizer.defocus_blur_model(x)

            dof_minima = torch.min(y)
            dof_maxima = torch.max(y)

            y = ((y - dof_minima) / (dof_maxima - dof_minima)) * dof_maxima

            if visualize:
                px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
                fig, ax = plt.subplots(figsize=(512 * px, 512 * px), facecolor='white')
                ax.set_facecolor("white")
                if not visualize:
                    sb.regplot(dof, x=depth, y=sorted_rad, fit_reg=False, x_jitter=0.05, y_jitter=0.2,
                               scatter_kws={'s': 0.5, 'alpha': 1 / 5}, label='Ground Truth')
                plt.plot(x.detach().cpu().numpy(), y.detach().cpu().numpy(), label='Estimated DOF Model')
                plt.xlabel('Disparity')
                plt.ylabel('Optimal Blur Radius')
                plt.legend()
                plt.ylim(0, np.ceil(dof_maxima.item()))
                # plt.ylim(0, 6)                        # Use for DoF Correctness Parameters (Supp. Fig. 5)
                fig.canvas.draw()
                plt.savefig(os.path.join("./output", img_name + "_result_",
                                         "{}/{}_blur_radius_vs_depth_fit_degree2_lbfgs.png".format(fName, img_name)))
                plt.savefig(os.path.join("./output", img_name + "_result_",
                                         "{}/{}_blur_radius_vs_depth_fit_degree2_lbfgs.pdf".format(fName, img_name)))

                # Convert the canvas to a raw RGB buffer
                buf = fig.canvas.tostring_rgb()
                ncols, nrows = fig.canvas.get_width_height()
                dof_fig = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3) / 255.0
                # dof_fig = np.transpose(dof_fig, (2, 0, 1))
                dof_viz_frames.append(dof_fig.astype('float32'))

                plt.close()

                depth_image = optimizer.depth_image.view(-1)
                sigma = ((optimizer.defocus_blur_model(depth_image) - dof_minima + 1e-3) /
                         (dof_maxima - dof_minima + 1e-3)) * dof_maxima
                sigma = sigma.view(img_height, img_width)
                depth_image = depth_image.view(img_height, img_width)

                defocus_blur_synthesized_image = blur_utils.custom_conv2d_optim(optimizer.focused_image, sigma,
                                                                                optimizer.kernel_size,
                                                                                padding=optimizer.kernel_size // 2)
                utils.save_image(defocus_blur_synthesized_image, img_name=img_name,
                                 file_desc='/{}/synth_DOF_image'.format(fName), fmt='png')

        '''Delete current models if not compositing!'''
        if opt_noise_model:
            optimizer.noise_model = None
        if opt_motion_blur_model:
            optimizer.motion_blur_model = None
        if opt_defocus_blur_model:
            optimizer.defocus_blur_model = None
        torch.cuda.empty_cache()

    # =================================================================================================================
    print('Mean Total optimization time: {:04f} sec. per frame'.format(optimizer.modeling_time['total']))
    print('Mean Noise optimization time: {:04f} sec. per frame'.format(optimizer.modeling_time['noise']))
    print('Mean MB optimization time: {:04f} sec. per frame'.format(optimizer.modeling_time['mb']))
    print('Mean DOF optimization time: {:04f} sec. per frame'.format(optimizer.modeling_time['dof']))

    image_data.free_mem()
