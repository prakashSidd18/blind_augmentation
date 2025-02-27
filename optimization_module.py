
import os

import imageio
import torch
import time
import numpy as np

import utils
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


class Optimization:

    def __init__(self, image_data=None, opt_noise_model=True, opt_motion_blur_model=True, opt_defocus_blur_model=True,
                 visualize=True, debug=False, save_model=True):
        self.image_data = image_data
        self.opt_noise_model = opt_noise_model
        self.opt_motion_blur_model = opt_motion_blur_model
        self.opt_defocus_blur_model = opt_defocus_blur_model

        self.visualize = visualize
        self.debug = debug
        self.save_model = save_model

        self.noise_model = None
        self.motion_blur_model = None
        self.defocus_blur_model = None
        self.parameters = None

        self.dataset_name = None
        self.original_image = None
        self.grayscale_image = None
        self.denoised_image = None
        self.noise_image = None
        self.demotion_blurred_image = None
        self.focused_image = None
        self.depth_image = None
        self.flow_image = None
        self.fName = None

        self.optimized = False
        self.kernel_size = 15

        self.modeling_time = {'noise': 0.0, 'mb': 0.0, 'dof': 0.0, 'total': 0.0}

    def init_opt_models(self, dataset, idx):
        self.dataset_name = dataset
        # print('######### {}x{} #########'.format(self.width, self.height))
        self.original_image, self.grayscale_image, self.denoised_image, self.demotion_blurred_image, \
            self.focused_image, self.noise_image, self.depth_image, \
            self.flow_image = self.image_data.get_single_frame(frame_id=idx)
        self.fName = self.image_data.input_data['frame_names'][idx]

        if self.visualize:
            utils.save_image(self.original_image, img_name=self.dataset_name,
                             file_desc='/{}/original_image'.format(self.fName), fmt='png', cmap='gray')
            utils.save_image(self.grayscale_image, channel=1, img_name=self.dataset_name,
                             file_desc='/{}/grayscale_image'.format(self.fName), fmt='png', cmap='gray')
            utils.save_image(self.denoised_image, img_name=self.dataset_name,
                             file_desc='/{}/denoised_image'.format(self.fName), fmt='png')
            utils.save_image(self.demotion_blurred_image, img_name=self.dataset_name,
                             file_desc='/{}/sharp_MB_image'.format(self.fName), fmt='png')
            utils.save_image(self.focused_image, img_name=self.dataset_name,
                             file_desc='/{}/sharp_DOF_image'.format(self.fName), fmt='png')
            utils.save_image(self.noise_image, img_name=self.dataset_name,
                             file_desc='/{}/noise'.format(self.fName), fmt='exr')
            utils.save_image(self.depth_image, img_name=self.dataset_name, channel=1,
                             file_desc='/{}/depth'.format(self.fName), fmt='png')
            flow_image_viz = utils.visulize_flow_file(self.flow_image)
            utils.save_image(flow_image_viz, img_name=self.dataset_name,
                             file_desc='/{}/flow'.format(self.fName), fmt='png', np_img=True)
            utils.save_image(self.flow_image[:, :, 0], img_name=self.dataset_name, channel=1,
                             file_desc='/{}/flow_u'.format(self.fName), fmt='exr', np_img=True)
            utils.save_image(self.flow_image[:, :, 1], img_name=self.dataset_name, channel=1,
                             file_desc='/{}/flow_v'.format(self.fName), fmt='exr', np_img=True)

    def load_opt_model(self, path):
        if self.opt_defocus_blur_model or self.opt_motion_blur_model:
            dof_model_path = path.replace("blur_parameters.npy", "dof_model.pth")
            if os.path.exists(path):
                self.parameters = np.load(path, allow_pickle=True)
                if self.opt_motion_blur_model and 'mb' in self.parameters.item().keys():
                    self.motion_blur_model = torch.tensor(self.parameters.item()['mb']).to(device)
                    self.optimized = True
                else:
                    print('MB model not available!! Please run optimization and save models before loading!')
                    self.optimized = False
                if self.opt_defocus_blur_model and os.path.exists(dof_model_path):
                    self.defocus_blur_model = blur_utils.PolynomialModel(degree=2).to(device)
                    self.defocus_blur_model.load_state_dict(torch.load(dof_model_path, weights_only=True))
                    self.optimized = True
                else:
                    print('DOF model not available!! Please run optimization and save models before loading!')
                    self.optimized = False
            else:
                self.optimized = False

        if self.opt_noise_model:
            self.optimized = False

        return self.optimized

    def run_optimization(self):
        '''Analysis step'''
        torch.cuda.synchronize()
        modeling_start_time = time.perf_counter()
        kernel_size = self.kernel_size
        parameters = {}

        # --------------------------------------------------------------------------------------------
        '''### Noise Analysis ###'''
        if self.opt_noise_model:
            # Train the model on noise
            pyrGenerator = LaplacePyramidGenerator()
            den_image = self.denoised_image[None, ...].cpu()
            image_luma = self.grayscale_image[None, None, :, :].cpu()
            target_noise = self.noise_image[None, ...].cpu()
            self.noise_model = laplace_noise_utils.fit_noise_model_luma_intercept(image_luma, target_noise,
                                                                                  pyrGenerator, n_levels=4)

            if self.save_model:
                save_path = os.path.join(self.dataset_name+"_result_",
                                                                       "{}/{}_noise_model.pth".format(self.fName,
                                                                                                    self.dataset_name))
                utils.create_folder(save_path)
                torch.save(torch.cat(self.noise_model), "./output/"+save_path)


        noise_modeling_end_time = time.perf_counter()

        # --------------------------------------------------------------------------------------------
        '''### Motion Blur Analysis ###'''
        if self.opt_motion_blur_model:
            optim_strength = 0
            if self.motion_blur_model:
                optim_strength = self.motion_blur_model
                # print('Before:', optim_strength)
            NN_to_blender_factor = 1.0
            self.flow_image = torch.from_numpy(self.flow_image*NN_to_blender_factor).permute(2, 0, 1).float().to(device)

            self.motion_blur_model = blur_utils.motion_blur_analysis(self.denoised_image, self.demotion_blurred_image,
                                                 self.flow_image, kernel_size=kernel_size, img_name=self.dataset_name)
            if self.motion_blur_model == 0.0 and optim_strength > 0.0:
                # print('Optimized:', self.motion_blur_model)
                self.motion_blur_model = optim_strength
                # print('After:', self.motion_blur_model)
            parameters['mb'] = self.motion_blur_model

        mb_modeling_end_time = time.perf_counter()

        # --------------------------------------------------------------------------------------------
        '''### Defocus Blur Analysis ###'''
        if self.opt_defocus_blur_model:
            self.defocus_blur_model = blur_utils.PolynomialModel(degree=2).to(device)
            self.defocus_blur_model, dof, depth, sorted_rad = \
                    blur_utils.defocus_blur_analysis(self.defocus_blur_model, self.demotion_blurred_image,
                                                     self.focused_image, self.depth_image, kernel_size=kernel_size,
                                                     filter_pix=False)
            if self.save_model:
                save_path = os.path.join(self.dataset_name + "_result_",
                           "{}/{}_dof_model.pth".format(self.fName, self.dataset_name))
                utils.create_folder(save_path)
                torch.save(self.defocus_blur_model.state_dict(), './output/'+save_path)
            parameters['dof'] = dof
            parameters['depth'] = depth
            parameters['sorted_rad'] = sorted_rad

        modeling_end_time = time.perf_counter()
        self.modeling_time['total'] = modeling_end_time - modeling_start_time
        self.modeling_time['noise'] = noise_modeling_end_time - modeling_start_time
        self.modeling_time['mb'] = mb_modeling_end_time - noise_modeling_end_time
        self.modeling_time['dof'] = modeling_end_time - mb_modeling_end_time

        if (self.opt_motion_blur_model or self.opt_defocus_blur_model) and self.save_model:
            save_path = os.path.join( self.dataset_name + "_result_",
                                 "{}/{}_blur_parameters.npy".format(self.fName, self.dataset_name))
            utils.create_folder(save_path)
            np.save('./output/'+save_path, parameters, allow_pickle=True)

        self.parameters = parameters


