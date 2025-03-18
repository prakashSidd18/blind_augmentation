import glob
import os

import imageio
import torch
import cv2
import numpy as np
import utils

if torch.cuda.is_available():
    device = torch.device("cuda")
    cuda = True
else:
    device = torch.device("cpu")
    cuda = False


os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
imageio.plugins.freeimage.download()


#=====================================================================================================================

class IO():
    def __init__(self, img_filepath=r"data/original/flir_noisy_rainbowballmotion.png", scale_down=True, scale=1,
                 single_frame_mode=False, frame_id=5, debug=False, skip_frames=1):
        # Current data path
        self.img_filepath = img_filepath
        self.img_abspath = os.path.abspath(img_filepath)
        self.img_name = os.path.basename(img_filepath).split('.')[0]
        self.img_dir = os.path.join(os.path.dirname(self.img_abspath), self.img_name)
        self.data_path = os.path.dirname(os.path.dirname(self.img_abspath))

        # Path to input data directory
        self.defocus_img_path = os.path.abspath(f"{self.data_path}/de_focus_blurred/")
        self.demotion_img_path = os.path.abspath(f"{self.data_path}/de_motion_blurred/")
        self.denoised_img_path = os.path.abspath(f"{self.data_path}/denoised/")
        self.composite_img_path = os.path.abspath(f"{self.data_path}/composites/")
        self.depth_img_path = os.path.abspath(f"{self.data_path}/depth/")
        self.flow_img_path = os.path.abspath(f"{self.data_path}/flow/")

        self.scale_down = scale_down
        self.scale = scale

        # Input frames data for optimization
        self.filenames = []
        self.input_data = {}

        # Input frames data for compositing
        self.composite_data = {}

        self.debug = debug
        self.skip_frames = skip_frames

        if os.path.exists(self.img_dir):
            self.filenames = sorted(os.listdir(self.img_dir)[:-1])
            self.totalNumFrames = len(self.filenames)
            self.single_frame_mode = single_frame_mode
            self.frame_ID = frame_id
        else:
            print('[IO Error] Original images directory does not exists!!')
            exit(-1)

    '''Read frame(s)'''
    def read_frames(self):
        frame_name = []
        original_frames = []
        gray_frames = []
        denoised_frames = []
        noise_frames = []
        demotion_blurred_frames = []
        defocus_blurred_frames = []
        depth_frames = []
        flow_frames = []
        height, width, channel = 0, 0, 0
        original_scale = self.scale
        for i, fName in enumerate(self.filenames):
            print('Reading input frame {}/{} ...'.format(i, self.totalNumFrames), end='\r')
            if self.scale_down:
                self.scale = original_scale
            if self.single_frame_mode and i != self.frame_ID:
                continue
            elif i % self.skip_frames != 0:
                original_frames.append([])
                gray_frames.append([])
                denoised_frames.append([])
                demotion_blurred_frames.append([])
                defocus_blurred_frames.append([])
                noise_frames.append([])
                depth_frames.append([])
                flow_frames.append([])
                frame_name.append([])
                continue

            iName = str(fName).split('.')[0]
            original_image = cv2.imread(os.path.join(self.img_dir, str(fName))) / 255.0
            original_image = original_image.astype('float32')
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            height, width, channel = original_image.shape
            or_height, or_width, or_channel = original_image.shape
            resized_image = original_image

            while (height > 512 and width > 512) and self.scale_down:
                self.scale += 1
                height, width = or_width // self.scale, or_height // self.scale

            if self.scale > 1:
                resized_image = cv2.resize(original_image,
                                            (or_width // self.scale, or_height // self.scale),
                                            interpolation=cv2.INTER_LINEAR)
                height, width, channel = resized_image.shape

            original_image = resized_image

            denoised_image = cv2.imread(os.path.join(self.denoised_img_path, self.img_name, str(fName))) / 255.0
            denoised_image = denoised_image.astype('float32')
            denoised_image = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB)
            if original_image.shape != denoised_image.shape:
                denoised_image = cv2.resize(denoised_image, (width, height), interpolation=cv2.INTER_CUBIC)
                # print('Warning!! Image size mismatch 0!! Resizing denoised image!!')

            gray_img_frame = cv2.cvtColor(denoised_image, cv2.COLOR_RGB2GRAY)

            demotion_blurred_image = cv2.imread(
                os.path.join(self.demotion_img_path, self.img_name, '{:03d}.png'.format(i))) / 255.0

            demotion_blurred_image = demotion_blurred_image.astype('float32')
            demotion_blurred_image = cv2.cvtColor(demotion_blurred_image, cv2.COLOR_BGR2RGB)
            if original_image.shape != demotion_blurred_image.shape:
                demotion_blurred_image = cv2.resize(
                    demotion_blurred_image, (width, height), interpolation=cv2.INTER_CUBIC)
                # print('Warning!! Image size mismatch 1!! Resizing sharp MB image!!')

            focused_image = cv2.imread(os.path.join(self.defocus_img_path, self.img_name, '{:03d}.png'.format(i))) / 255.0
            focused_image = focused_image.astype('float32')
            focused_image = cv2.cvtColor(focused_image, cv2.COLOR_BGR2RGB)
            if original_image.shape != focused_image.shape:
                focused_image = cv2.resize(focused_image, (width, height), interpolation=cv2.INTER_CUBIC)
                # print('Warning!! Image size mismatch 2!! Resizing sharp DOF image!!')

            depth = cv2.imread(os.path.join(self.depth_img_path, self.img_name, '{:03d}.png'.format(i))) / 255.0
            depth = depth.astype('float32')
            depth = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)
            if original_image.shape != depth.shape:
                depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_CUBIC)
                # print('Warning!! Image size mismatch 3!! Resizing depth image')

            try:
                flow = utils.readFlow(os.path.join(self.flow_img_path, self.img_name, '{:06d}.flo'.format(i)))
                if flow.shape[0] != height or flow.shape[1] != width:
                    flow = cv2.resize(flow, (width, height), interpolation=cv2.INTER_CUBIC)
                    # print('Warning!! Image size mismatch 4!! Resizing flow image')
                flow_image = utils.visulize_flow_file(flow)
            except:
                flow = np.zeros((height, width, 2)).astype('float32')
                flow_image = np.zeros((height, width, 3)).astype('float32')
                print('Warning!! No flow for {} read!!'.format(fName))

            # compute residual noise
            noise_image = (original_image - denoised_image).astype('float32')

            if self.debug:
                utils.save_image(original_image, img_name=self.img_name, np_img=True,
                                 file_desc='/{:04d}_original_image'.format(self.frame_ID), fmt='png')
                utils.save_image(denoised_image, img_name=self.img_name, np_img=True,
                                 file_desc='/{:04d}_denoised_image'.format(self.frame_ID), fmt='png')
                utils.save_image(demotion_blurred_image, img_name=self.img_name, np_img=True,
                                 file_desc='/{:04d}_sharp_MB_image'.format(self.frame_ID), fmt='png')
                utils.save_image(focused_image, img_name=self.img_name, np_img=True,
                                 file_desc='/{:04d}_sharp_DOF_image'.format(self.frame_ID), fmt='png')
                utils.save_image(noise_image, img_name=self.img_name, np_img=True,
                                 file_desc='/{:04d}_noise'.format(self.frame_ID), fmt='exr')
                utils.save_image(depth, img_name=self.img_name, channel=1, np_img=True,
                                 file_desc='/{:04d}_depth'.format(self.frame_ID), fmt='png')
                utils.save_image(flow_image, img_name=self.img_name,
                                 file_desc='/{:04d}_flow'.format(self.frame_ID), fmt='png', np_img=True)

            original_frames.append(original_image)
            gray_frames.append(gray_img_frame)
            denoised_frames.append(denoised_image)
            demotion_blurred_frames.append(demotion_blurred_image)
            defocus_blurred_frames.append(focused_image)
            noise_frames.append(noise_image)
            depth_frames.append(depth)
            flow_frames.append(flow)
            frame_name.append(iName)

        self.input_data['original_frames'] = original_frames
        self.input_data['gray_frames'] = gray_frames
        self.input_data['denoised_frames'] = denoised_frames
        self.input_data['noise_frames'] = noise_frames
        self.input_data['demotion_blurred_frames'] = demotion_blurred_frames
        self.input_data['defocus_blurred_frames'] = defocus_blurred_frames
        self.input_data['depth_frames'] = depth_frames
        self.input_data['flow_frames'] = flow_frames
        self.input_data['frame_names'] = frame_name
        self.input_data['numFrames'] = len(original_frames)
        self.input_data['width'] = width
        self.input_data['height'] = height

        print('\nDone reading input frames!')

        return self.input_data

    '''Return single frame for optimization'''
    def get_single_frame(self, frame_id=0):

        original_frame = self.input_data['original_frames'][frame_id]
        grayscale_frame = self.input_data['gray_frames'][frame_id]
        denoised_frame = self.input_data['denoised_frames'][frame_id]
        noise_frame = self.input_data['noise_frames'][frame_id]
        demotion_blurred_frame = self.input_data['demotion_blurred_frames'][frame_id]
        defocus_blurred_frame = self.input_data['defocus_blurred_frames'][frame_id]
        depth_frame = self.input_data['depth_frames'][frame_id]
        flow_frame = self.input_data['flow_frames'][frame_id]

        # send to GPU
        original_frame = torch.from_numpy(original_frame).permute(2, 0, 1).float().to(device)
        grayscale_frame = torch.from_numpy(grayscale_frame).float().to(device)
        denoised_frame = torch.from_numpy(denoised_frame).permute(2, 0, 1).float().to(device)
        noise_frame = torch.from_numpy(noise_frame).permute(2, 0, 1).float().to(device)
        demotion_blurred_frame = torch.from_numpy(demotion_blurred_frame).permute(2, 0, 1).float().to(device)
        defocus_blurred_frame = torch.from_numpy(defocus_blurred_frame).permute(2, 0, 1).float().to(device)

        depth_frame = torch.from_numpy(depth_frame).permute(2, 0, 1).float().to(device)
        depth_frame = depth_frame[0]

        return original_frame, grayscale_frame, denoised_frame, demotion_blurred_frame, defocus_blurred_frame, \
            noise_frame, depth_frame, flow_frame

    def free_mem(self):
        del self.input_data
        del self.composite_data
        torch.cuda.empty_cache()

    '''Read compositing frame(s)'''
    def read_composites(self):
        composite_frames_virtual = []
        composite_mask_virtual = []
        composite_frames_real = []
        composite_mask_real = []
        composite_frames_virtual_real = []
        composite_mask_virtual_real = []
        composite_depth_frames = []
        composite_flow_frames = []
        num_composite_frames = len(glob.glob((os.path.join(self.composite_img_path, self.img_name, '*.npz'))))
        if num_composite_frames != self.totalNumFrames:
            skip_frames = (self.totalNumFrames // num_composite_frames) + 1
        else:
            skip_frames = 1

        for i in range(num_composite_frames):
            print('Reading composite frame {}/{} ...'.format(i, num_composite_frames), end='\r')
            if self.debug and i != (self.frame_ID//skip_frames):
                continue
            # Read Composite Image and Mask
            composite_image = cv2.imread(os.path.join(self.composite_img_path, self.img_name, '{:04d}.png'.format(i+1)),
                                         cv2.IMREAD_UNCHANGED)
            composite_height, composite_width, composite_channel = composite_image.shape
            if composite_image.shape[2] == 4:
                composite_mask = composite_image[:, :, 3] / 255.0
                composite_image = composite_image[:, :, :3]
            else:
                composite_mask = np.ones([composite_height, composite_width])
            mask = np.zeros(composite_image.shape)
            if composite_mask is not None:
                mask[:, :, 0] = composite_mask
                mask[:, :, 1] = composite_mask
                mask[:, :, 2] = composite_mask
            composite_image = composite_image / 255.0
            composite_image = composite_image * mask
            composite_image = composite_image.astype('float32')
            composite_image = cv2.cvtColor(composite_image, cv2.COLOR_BGR2RGB)

            # Read composite image with background
            composite_image_with_bg = cv2.imread(
                os.path.join(self.composite_img_path, self.img_name, '{:04d}_with_bg.png'.format(i+1)),
                cv2.IMREAD_UNCHANGED)
            if composite_image_with_bg.shape[2] == 4:
                composite_mask_with_bg = composite_image_with_bg[:, :, 3] / 255.0
                composite_image_with_bg = composite_image_with_bg[:, :, :3]
            else:
                composite_mask_with_bg = np.ones([composite_height, composite_width, 3])

            mask_with_bg = np.zeros(composite_image_with_bg.shape)
            if composite_mask_with_bg is not None:
                mask_with_bg[:, :, 0] = composite_mask_with_bg
                mask_with_bg[:, :, 1] = composite_mask_with_bg
                mask_with_bg[:, :, 2] = composite_mask_with_bg
            composite_image_with_bg = (composite_image_with_bg[:, :, :3] / 255.0) * mask_with_bg
            composite_image_with_bg = composite_image_with_bg.astype('float32')
            composite_image_with_bg = cv2.cvtColor(composite_image_with_bg, cv2.COLOR_BGR2RGB)

            # Read composite image containing only local background
            composite_image_only_bg = cv2.imread(
                os.path.join(self.composite_img_path, self.img_name, '{:04d}_only_bg.png'.format(i+1)),
                cv2.IMREAD_UNCHANGED)
            if composite_image_only_bg.shape[2] == 4:
                composite_mask_only_bg = composite_image_only_bg[:, :, 3] / 255.0
                composite_image_only_bg = composite_image_only_bg[:, :, :3]
            else:
                composite_mask_only_bg = np.ones([composite_height, composite_width, 3])

            mask_only_bg = np.zeros(composite_image_only_bg.shape)
            if composite_mask_with_bg is not None:
                mask_only_bg[:, :, 0] = composite_mask_only_bg
                mask_only_bg[:, :, 1] = composite_mask_only_bg
                mask_only_bg[:, :, 2] = composite_mask_only_bg
            composite_image_only_bg = (composite_image_only_bg[:, :, :3] / 255.0) * mask_only_bg
            composite_image_only_bg = composite_image_only_bg.astype('float32')
            composite_image_only_bg = cv2.cvtColor(composite_image_only_bg, cv2.COLOR_BGR2RGB)

            if self.debug:
                utils.save_image(composite_image, img_name=self.img_name, file_desc='/{:04d}_composite_image'.format(i),
                                 fmt='png', np_img=True)
                utils.save_image(composite_mask.astype('float32'), img_name=self.img_name, channel=1,
                                 file_desc='/{:04d}_composite_mask'.format(i), cmap='gray', np_img=True, fmt='png')
                utils.save_image(composite_image_with_bg, img_name=self.img_name,
                                 file_desc='/{:04d}_composite_image_with_bg'.format(i),
                                 fmt='png', np_img=True)
                utils.save_image(composite_mask_with_bg.astype('float32'), img_name=self.img_name, channel=1,
                                 file_desc='/{:04d}_composite_mask_with_bg'.format(i), cmap='gray', np_img=True,
                                 fmt='png')
                utils.save_image(composite_image_only_bg, img_name=self.img_name,
                                 file_desc='/{:04d}_composite_image_only_bg'.format(i),
                                 fmt='png', np_img=True)
                utils.save_image(composite_mask_only_bg.astype('float32'), img_name=self.img_name, channel=1,
                                 file_desc='/{:04d}_composite_mask_only_bg'.format(i), cmap='gray', np_img=True,
                                 fmt='png')
                utils.save_image(composite_image, img_name=self.img_name, file_desc='/{:04d}_composite_image'.format(i),
                                 fmt='exr', np_img=True)
                utils.save_image(composite_mask.astype('float32'), img_name=self.img_name, channel=1,
                                 file_desc='/{:04d}_composite_mask'.format(i), np_img=True, fmt='exr')
                utils.save_image(composite_image_with_bg, img_name=self.img_name,
                                 file_desc='/{:04d}_composite_image_with_bg'.format(i),
                                 fmt='exr', np_img=True)
                utils.save_image(composite_image_only_bg, img_name=self.img_name,
                                 file_desc='/{:04d}_composite_image_only_bg'.format(i),
                                 fmt='exr', np_img=True)

            # Read Composite Flow
            try:
                composite_flow = utils.read_flow_exr_file(
                    os.path.join(self.composite_img_path, self.img_name, '{:04d}_flow.exr'.format(i+1)))
                if composite_flow.shape[0] != composite_height or composite_flow.shape[1] != composite_width:
                    composite_flow = cv2.resize(composite_flow, (composite_width, composite_height),
                                                      interpolation=cv2.INTER_CUBIC)
                composite_flow_image = utils.visulize_flow_file(composite_flow)
                if self.debug:
                    utils.save_image(composite_flow_image, img_name=self.img_name,
                                        file_desc='/{:04d}_flow_composite'.format(i), np_img=True)
                    utils.save_image(composite_flow[:, :, 0], img_name=self.img_name, channel=1,
                                     file_desc='/{:04d}_flow_u_composite'.format(i), fmt='exr', np_img=True)
                    utils.save_image(composite_flow[:, :, 1], img_name=self.img_name, channel=1,
                                     file_desc='/{:04d}_flow_v_composite'.format(i), fmt='exr', np_img=True)
            except:
                composite_flow = np.zeros((composite_height, composite_width, 2)).astype('float32')
                composite_flow_image = np.zeros((composite_height, composite_width, 3)).astype('float32')
                print('Warning!!! COMPOSITE Flow for image {:04d} not read!!!'.format(i+1))

            # Read Composite Depth
            composite_depth = np.load(os.path.join(self.composite_img_path, self.img_name, '{:04d}.npz'.format(i+1)))
            composite_depth = 1 - composite_depth['dmap']
            if self.debug:
                utils.save_image(composite_depth, img_name=self.img_name, channel=1,
                                 file_desc='/{:04d}_composite_depth_rgb'.format(i), np_img=True, fmt='png')
                utils.save_image(composite_depth, img_name=self.img_name, channel=1,
                                 file_desc='/{:04d}_composite_depth_gs'.format(i), np_img=True, cmap='gray',
                                 fmt='png')
                utils.save_image(composite_depth, img_name=self.img_name, channel=1,
                                 file_desc='/{:04d}_composite_depth'.format(i), np_img=True, fmt='exr')

            if self.scale_down and self.scale > 1:
                composite_image = cv2.resize(composite_image,
                                             (composite_width // self.scale, composite_height // self.scale),
                                             interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask,
                                             (composite_width // self.scale, composite_height // self.scale),
                                             interpolation=cv2.INTER_LINEAR)
                composite_image_only_bg = cv2.resize(composite_image_only_bg,
                                             (composite_width // self.scale, composite_height // self.scale),
                                             interpolation=cv2.INTER_LINEAR)
                mask_only_bg = cv2.resize(mask_only_bg,
                                             (composite_width // self.scale, composite_height // self.scale),
                                             interpolation=cv2.INTER_LINEAR)
                composite_image_with_bg = cv2.resize(composite_image_with_bg,
                                             (composite_width // self.scale, composite_height // self.scale),
                                             interpolation=cv2.INTER_LINEAR)
                mask_with_bg = cv2.resize(mask_with_bg,
                                             (composite_width // self.scale, composite_height // self.scale),
                                             interpolation=cv2.INTER_LINEAR)
                composite_depth = cv2.resize(composite_depth,
                                             (composite_width // self.scale, composite_height // self.scale),
                                             interpolation=cv2.INTER_LINEAR)
                composite_flow = cv2.resize(composite_flow,
                                             (composite_width // self.scale, composite_height // self.scale),
                                             interpolation=cv2.INTER_LINEAR)

            composite_frames_virtual.append(composite_image)
            composite_mask_virtual.append(mask)
            composite_frames_real.append(composite_image_only_bg)
            composite_mask_real.append(mask_only_bg)
            composite_frames_virtual_real.append(composite_image_with_bg)
            composite_mask_virtual_real.append(mask_with_bg)
            composite_depth_frames.append(composite_depth)
            composite_flow_frames.append(composite_flow)

        self.composite_data['composite_frames_virtual'] = composite_frames_virtual
        self.composite_data['composite_mask_virtual'] = composite_mask_virtual
        self.composite_data['composite_frames_real'] = composite_frames_real
        self.composite_data['composite_mask_real'] = composite_mask_real
        self.composite_data['composite_frames_virtual_real'] = composite_frames_virtual_real
        self.composite_data['composite_mask_virtual_real'] = composite_mask_virtual_real
        self.composite_data['composite_depth_frames'] = composite_depth_frames
        self.composite_data['composite_flow_frames'] = composite_flow_frames

        print('\nDone reading composite frames!')
        return self.composite_data


#=====================================================================================================================
