
import os

import imageio
import torch
import cv2
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


class Compositor:

    def __init__(self, image_data=None,
                 noise_model=None, motion_blur_model=None, defocus_blur_model=None,
                 dilation=True, erosion=True, d_kSize=15, e_kSize=7, scale=1,
                 visualize=False, debug=False):
        self.image_data = image_data

        self.visualize = visualize
        self.debug = debug

        self.noise_model = noise_model
        self.motion_blur_model = motion_blur_model
        self.defocus_blur_model = defocus_blur_model

        self.dataset_name = None
        self.original_image = None
        self.image_depth = None
        self.noise_image = None
        self.fName = None

        self.kernel_size = 15

        self.composite_image = None
        self.mask = None
        self.composite_image_only_bg = None
        self.mask_only_bg = None
        self.composite_image_with_bg = None
        self.mask_with_bg = None
        self.composite_depth = None
        self.composite_flow = None

        self.occlusion_mask = None
        self.final_mask_dilated = None
        self.naive_bg_image = None

        self.width = 0
        self.height = 0

        self.composite_mask = None
        self.dilation = dilation
        self.erosion = erosion

        self.dilation_kSize = d_kSize // scale
        self.erosion_kSize = e_kSize // scale

        self.frame_id = 0

    def init_composite_frame(self, dataset, frame_id=0, skip_frames=1, skip_multiplier=1):
        # read image data
        self.dataset_name = dataset
        self.frame_id = frame_id
        self.original_image = self.image_data.input_data['original_frames'][frame_id]
        self.image_depth = self.image_data.input_data['depth_frames'][frame_id][:, :, 0]
        self.noise_image = self.image_data.input_data['noise_frames'][frame_id]
        self.noise_image = torch.from_numpy(self.noise_image).permute(2, 0, 1).float().to(device)
        self.fName = self.image_data.input_data['frame_names'][frame_id]
        self.width = self.image_data.input_data['width']
        self.height = self.image_data.input_data['height']

        # read composite frames data
        self.composite_image = self.image_data.composite_data['composite_frames_virtual'][(frame_id*skip_multiplier)//skip_frames]
        self.mask = self.image_data.composite_data['composite_mask_virtual'][(frame_id*skip_multiplier)//skip_frames]
        self.composite_image_only_bg = self.image_data.composite_data['composite_frames_real'][(frame_id*skip_multiplier)//skip_frames]
        self.mask_only_bg = self.image_data.composite_data['composite_mask_real'][(frame_id*skip_multiplier)//skip_frames]
        self.composite_image_with_bg = self.image_data.composite_data['composite_frames_virtual_real'][(frame_id*skip_multiplier)//skip_frames]
        self.mask_with_bg = self.image_data.composite_data['composite_mask_virtual_real'][(frame_id*skip_multiplier)//skip_frames]
        self.composite_depth = self.image_data.composite_data['composite_depth_frames'][(frame_id*skip_multiplier)//skip_frames]
        self.composite_flow = self.image_data.composite_data['composite_flow_frames'][(frame_id*skip_multiplier)//skip_frames]

    def depth_test(self, off=False):

        if off:
            self.occlusion_mask = np.ones(self.composite_image.shape)
            self.composite_mask = self.mask * self.occlusion_mask
            self.final_mask_dilated = np.zeros(self.composite_mask.shape)
            return

        self.occlusion_mask = np.zeros(self.composite_image.shape)
        visibility_mask = self.composite_depth > self.image_depth

        self.occlusion_mask[:, :, 0] = visibility_mask
        self.occlusion_mask[:, :, 1] = visibility_mask
        self.occlusion_mask[:, :, 2] = visibility_mask

        # take transparency on edges into account
        self.occlusion_mask = self.occlusion_mask.astype('float32')

        if self.visualize:
            utils.save_image(self.occlusion_mask, img_name=self.dataset_name, channel=1,
                             file_desc='/{:04d}_visibility_mask'.format(self.frame_id), np_img=True, fmt='png')
        if self.debug:
            utils.save_image(self.occlusion_mask, img_name=self.dataset_name,
                             file_desc='/{:04d}_visibility_mask'.format(self.frame_id), np_img=True, fmt='exr')

        self.composite_mask = self.mask * self.occlusion_mask
        self.final_mask_dilated = np.zeros(self.composite_mask.shape)

        if self.dilation:
            # define the dilation kernel
            dilation_kernel = np.ones((self.dilation_kSize, self.dilation_kSize), np.uint8)

            # find where both real and virtual are present
            # it's just the composite mask / mask
            # occlusion mask is where virtual wins over real
            # find where real wins over virtual and both are present i.e. the foreground mask
            foreground_mask = np.zeros(self.mask.shape).astype(('float32'))
            foreground_mask[self.image_depth >= self.composite_depth] = 1.0
            foreground_mask *= self.mask

            self.final_mask_dilated = cv2.dilate(self.composite_mask, dilation_kernel, iterations=1)
            self.final_mask_dilated *= foreground_mask
            self.composite_mask += self.final_mask_dilated

            if self.debug:
                utils.save_image(self.composite_mask.astype('float32'), img_name=self.dataset_name,
                         file_desc='/{:04d}_composite_mask_with_visibility_dilated_real_edges'.format(self.frame_id),
                         np_img=True,
                         fmt='png')

    def naive_compositing(self):
        # Change composite image and background image to include difference in local scene
        erosion_kernel = np.ones((self.erosion_kSize, self.erosion_kSize), np.uint8)

        difference_image = (self.composite_image_with_bg - self.composite_image_only_bg)
        difference_image *= (1 - self.composite_mask)

        if self.erosion:
            difference_image = cv2.erode(difference_image, erosion_kernel, iterations=1)

        self.naive_bg_image = (self.original_image + difference_image).clip(0, 1)

        if self.debug:
            utils.save_image(difference_image, img_name=self.dataset_name,
                             file_desc='/{:04d}_difference_image'.format(self.frame_id), np_img=True, fmt='exr')
            utils.save_image(self.naive_bg_image, img_name=self.dataset_name,
                         file_desc='/{:04d}_background_image_with_diff'.format(self.frame_id), np_img=True, fmt='png')

        naive_composite = utils.image_compositing(self.naive_bg_image, self.composite_image_with_bg,
                                                  self.composite_mask).astype('float32')
        utils.save_image(naive_composite, img_name=self.dataset_name,
                         file_desc='/{:04d}_naive_composite'.format(self.frame_id),
                         fmt='png', np_img=True)

        return naive_composite

    def defocus_blur_composite(self, dof_minima, dof_maxima):
        composite_image, defocus_blurred_mask, composite_sigma, blurred_composite_sigma = \
            blur_utils.defocus_blur_composite(self.composite_image_with_bg, self.composite_depth,
                                              self.defocus_blur_model, dof_minima, dof_maxima,
                                              self.mask, self.occlusion_mask, self.final_mask_dilated,
                                              kernel_size=self.kernel_size)
        composite_image = np.transpose(composite_image.detach().cpu().numpy(), (1, 2, 0))
        defocus_blurred_mask = np.transpose(defocus_blurred_mask.detach().cpu().numpy(), (1, 2, 0))

        if self.debug:
            utils.save_image(composite_sigma, img_name=self.dataset_name, channel=1,
                             file_desc='/{:04d}_composite_radii_defocus'.format(self.frame_id), fmt='exr')
            utils.save_image(blurred_composite_sigma, img_name=self.dataset_name, channel=1,
                             file_desc='/{:04d}_blurred_composite_radii_defocus'.format(self.frame_id), fmt='exr')
            utils.save_image(composite_image, img_name=self.dataset_name,
                             file_desc='/{:04d}_blurred_composite_image_defocus'.format(self.frame_id), fmt='png',
                             np_img=True)
            utils.save_image(defocus_blurred_mask, img_name=self.dataset_name,
                             file_desc='/{:04d}_blurred_composite_mask_defocus'.format(self.frame_id), fmt='exr',
                             np_img=True)
            utils.save_image(self.naive_bg_image, img_name=self.dataset_name,
                             file_desc='/{:04d}_background_image_defocus'.format(self.frame_id), np_img=True, fmt='png')

        blurred_composite = utils.image_compositing(self.naive_bg_image, composite_image,
                                                    defocus_blurred_mask).astype('float32')
        utils.save_image(blurred_composite, img_name=self.dataset_name,
                         file_desc='/{:04d}_DOF_composite'.format(self.frame_id), fmt='png', np_img=True)

        self.composite_image = composite_image
        self.composite_mask = defocus_blurred_mask

        return blurred_composite

    def motion_blur_composite(self):
        # if defocus blur has not been applied to the compositing before, use composite virtual+real as input,
        # else composite image is blurred_composite_image_virtual + real, so do nothing
        if not self.defocus_blur_model:
            self.composite_image = self.composite_image_with_bg

        if torch.is_tensor(self.motion_blur_model):
            self.motion_blur_model = self.motion_blur_model.detach().cpu().numpy()
        composite_flow_image = self.motion_blur_model * self.composite_flow
        composite_image, motion_blurred_mask, composite_flow, blurred_composite_flow = blur_utils.motion_blur_composite(
            self.composite_image, composite_flow_image, self.composite_mask, kernel_size=self.kernel_size)

        composite_image = np.transpose(composite_image.detach().cpu().numpy(), (1, 2, 0))
        composite_image[np.isinf(composite_image)] = 0
        motion_blurred_mask = np.transpose(motion_blurred_mask.detach().cpu().numpy(), (1, 2, 0))
        blurred_composite_flow = np.transpose(blurred_composite_flow.detach().cpu().numpy(), (1, 2, 0))

        if self.debug:
            utils.save_image(self.naive_bg_image, img_name=self.dataset_name,
                             file_desc='/{:04d}_background_image_motion'.format(self.frame_id), np_img=True, fmt='png')
            utils.save_image(composite_image, img_name=self.dataset_name,
                             file_desc='/{:04d}_blurred_composite_image_motion'.format(self.frame_id), fmt='png',
                             np_img=True)
            utils.save_image(motion_blurred_mask, img_name=self.dataset_name,
                             file_desc='/{:04d}_blurred_composite_mask_motion'.format(self.frame_id), fmt='png', np_img=True)
            composite_flow_image_viz = utils.visulize_flow_file(blurred_composite_flow)
            utils.save_image(composite_flow_image_viz, img_name=self.dataset_name,
                             file_desc='/{:04d}_blurred_composite_flow'.format(self.frame_id), fmt='png', np_img=True)
            utils.save_image(self.naive_bg_image, img_name=self.dataset_name,
                             file_desc='/{:04d}_background_image_motion'.format(self.frame_id), np_img=True, fmt='exr')
            utils.save_image(composite_image, img_name=self.dataset_name,
                             file_desc='/{:04d}_blurred_composite_image_motion'.format(self.frame_id), fmt='exr', np_img=True)
            utils.save_image(motion_blurred_mask, img_name=self.dataset_name,
                             file_desc='/{:04d}_blurred_composite_mask_motion'.format(self.frame_id), fmt='exr', np_img=True)
            utils.save_image_np(blurred_composite_flow, img_name=self.dataset_name,
                                file_desc='/{:04d}_flow_composite_blurred'.format(self.frame_id), np_img=True)

        blurred_composite = utils.image_compositing(self.naive_bg_image, composite_image,
                                                    motion_blurred_mask).astype('float32')

        if self.defocus_blur_model:
            utils.save_image(blurred_composite, img_name=self.dataset_name,
                             file_desc='/{:04d}_DOF_MB_composite'.format(self.frame_id), fmt='png', np_img=True)
        else:
            utils.save_image(blurred_composite, img_name=self.dataset_name,
                             file_desc='/{:04d}_MB_composite'.format(self.frame_id), fmt='png', np_img=True)

        self.composite_image = composite_image
        self.composite_mask = motion_blurred_mask

        return blurred_composite

    def noise_composite(self):
        # if no blur has been applied to the compositing before, use composite virtual+real as input,
        # else composite image is blurred_composite_image_virtual + real, so do nothing
        if not self.defocus_blur_model and not self.motion_blur_model:
            self.composite_image = self.composite_image_with_bg

        # Generate image noise in shape of original noise
        composite_rgb = torch.from_numpy(self.composite_image).permute(2, 0, 1).float()[None, ...]
        grayscale_img = cv2.cvtColor(self.composite_image, cv2.COLOR_RGB2GRAY)
        grayscale_img = torch.from_numpy(grayscale_img).float()[None, None, :, :]
        pyr_generator = LaplacePyramidGenerator()
        input_noise = torch.randn_like(self.noise_image[None, ...]).cpu()
        generated_noise = laplace_noise_utils.synthesise_noise_luma_intercept(input_noise,
                                                                              grayscale_img,
                                                                              self.noise_model, pyr_generator)
        grayscale_img = grayscale_img[0, 0, :, :].detach().cpu().numpy()
        grayscale_img = (grayscale_img * self.composite_mask[:, :, 0]).astype('float32')

        # Masked composite noise
        generated_noise = np.transpose(generated_noise[0].detach().cpu().numpy(), (1, 2, 0))

        masked_noise = (generated_noise * self.composite_mask).astype('float32')
        viz_noise = torch.from_numpy(masked_noise).permute(2, 0, 1).float().to(device)

        composite_noisy_img = (self.composite_image + masked_noise).clip(0, 1)
        blurred_composite = utils.image_compositing(self.naive_bg_image, composite_noisy_img,
                                                    self.composite_mask).astype('float32')

        if self.debug:
            utils.save_image(masked_noise, img_name=self.dataset_name,
                             file_desc='/{:04d}_masked_noise'.format(self.frame_id), fmt='exr', np_img=True)
            utils.save_image(composite_noisy_img, img_name=self.dataset_name,
                             file_desc='/{:04d}_composite_noisy_image'.format(self.frame_id), fmt='png', np_img=True)

        utils.save_image(blurred_composite, img_name=self.dataset_name,
                         file_desc='/{:04d}_noisy_composite'.format(self.frame_id), fmt='png', np_img=True)

        return blurred_composite, viz_noise, grayscale_img
