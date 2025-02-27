
import torch
import torch.nn.functional as nn
import time
import numpy as np
import torchvision.transforms

import utils
from multi_blur import MultiBlur

if torch.cuda.is_available():
    device = torch.device("cuda")
    cuda = True
else:
    device = torch.device("cpu")
    cuda = False

loss_func = torch.nn.MSELoss().to(device)



def blur(image, axis, taps = 1):
    result = 0
    weights = np.array([1,2,1]) / 4
    for i in range (-taps, taps+1):
        result += weights[i + taps] * np.roll(image, i, axis = axis)
    return result

def downsample(image):
    image = blur(image, -2)[:,::2,:]
    image = blur(image, -3)[::2,:,:]
    return image

def upsample(image):
    image = np.repeat(image, 2, axis = (-2))
    image = blur(image, -2)
    image = np.repeat(image, 2, axis = (-3))
    image = blur(image, -3)
    return image

def pull_push(image, mask, level_count = 5):
    downsampled_mask = mask
    downsampled_image = image

    image_pyramid = [image]
    mask_pyramid = [mask]

    # Pull
    for i in range(0, level_count):
        premul_image = downsampled_image * downsampled_mask
        downsampled_mask = downsample(downsampled_mask)
        downsampled_image = downsample(premul_image) / downsampled_mask
        downsampled_image = np.nan_to_num(downsampled_image, 0)
        image_pyramid.append(downsampled_image)
        mask_pyramid.append(downsampled_mask)

    # Push
    upsampled_image = downsampled_image
    for i in range(1, level_count+1):
        upsampled_image = upsample(upsampled_image)
        alpha = mask_pyramid[-1 - i]
        upsampled_image = (1-alpha) * upsampled_image + alpha * image_pyramid[-1 - i]
    return upsampled_image


def grid(shape):
    u = np.linspace(0, 1, shape[1])
    v = np.linspace(0, 1, shape[0])
    x, y = np.meshgrid(v, u)
    return np.stack((x, y)).T


def bspline(x, index, order):
  if order == 0:
    return np.where((x >= index) & (x < index + 1), 1, 0)
  else:
    w0 = x - index
    w1 = index + 1 + order - x
    b0 = bspline(x, index, order - 1)
    b1 = bspline(x, index + 1, order - 1)
    return w0 * b0 + w1 * b1


def tensor_product(x, index0, index1, order):
  return bspline(x[:,:,0], -index0, order) * bspline(x[:,:,1], -index1, order)


def sample(image, positions):
    positions, pixel_indices = np.modf(positions * (np.array(image.shape[:2]) - 1))
    pixel_indices = np.array(pixel_indices, dtype = np.int32)

    result = 0
    order = 1
    for i in range(-order, order + 1):
        for j in range(-order, order + 1):
            w = tensor_product(positions, order + i, order + j, order)
            y = np.roll(image, (i, j), axis = (0, 1))[pixel_indices[:,:,0], pixel_indices[:,:,1],:]
            result += y * w[..., np.newaxis]
    return result

def normalize_blur(image, mask, level_count = 6):
  downsampled_mask = mask
  downsampled_image = image

  # Pull
  for i in range(0, level_count):
    premul_image = downsampled_image * downsampled_mask
    downsampled_mask = downsample(downsampled_mask)
    downsampled_image = downsample(premul_image)

    downsampled_image /= downsampled_mask
    downsampled_image = np.nan_to_num(downsampled_image)

  sample_positions = grid(image.shape[:2])
  result = sample(downsampled_image, sample_positions)

  return result


# https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
def gkern(l=5, sig=1.):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = torch.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = torch.exp(-0.5 * torch.square(ax) / torch.square(sig))
    kernel = torch.outer(gauss, gauss)
    result = kernel / torch.sum(kernel)
    return result


def gkern_tensor(l=5, sig=1.):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = torch.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = torch.exp(-0.5 * torch.square(ax[:, None, None]) / torch.square(sig))
    gauss = gauss.permute(1, 2, 0)
    a = gauss.unsqueeze(-1)
    b = gauss.unsqueeze(-2)
    kernel = torch.matmul(a, b)
    kernel_sum = torch.sum(torch.sum(kernel, dim=-1), dim=-1).unsqueeze(-1).unsqueeze(-1)
    result = kernel / kernel_sum
    return result.permute(2, 3, 0, 1)


def zero_padding(input_image, sigmas, padding):
    c, h, w = input_image.size()
    h += padding * 2
    w += padding * 2
    new_input = torch.zeros([c, h, w])
    new_input[:, padding:h - padding, padding:w - padding] = input_image
    new_sigmas = torch.zeros([h, w])
    new_sigmas[padding:h - padding, padding:w - padding] = sigmas

    return new_input, new_sigmas


def custom_conv2d_optim(input_image, sigmas, k_size=3, stride=1, padding=0):

    if padding > 0:
        # modify input_image and sigmas to pad appropriately
        # print('Zero Padding with size {}'.format(str(padding)))
        input_image, sigmas = zero_padding(input_image, sigmas, padding)

    c, h, w = input_image.size()

    out_h = (h - (k_size - 1) // stride)
    out_w = (w - (k_size - 1) // stride)

    result = torch.zeros([c, out_h, out_w]).to(device)

    for channel in range(c):
        img = input_image[channel, :, :]
        # create the copies
        copied_input_image = img.expand(k_size, k_size, h, w)
        shifted_input_image = torch.zeros(copied_input_image.size())

        for i in range(k_size):
            for j in range(k_size):
                # shift current image, truncate boundaries if required
                center_pixel = k_size // 2
                shift_i = center_pixel - i
                shift_j = center_pixel - j

                shift_start_i = 0 if shift_i < 0 else shift_i
                shift_start_j = 0 if shift_j < 0 else shift_j
                shift_end_i = h + shift_i
                shift_end_j = w + shift_j

                copy_start_i = 0 - shift_i if shift_i < 0 else 0
                copy_start_j = 0 - shift_j if shift_j < 0 else 0

                shifted_input_image[i, j, shift_start_i:shift_end_i, shift_start_j:shift_end_j] = \
                    copied_input_image[i, j, copy_start_i:h-shift_i, copy_start_j:w-shift_j]

        # convolve the shifted images with kernel
        kernels = gkern_tensor(k_size, sigmas)

        prod = kernels * shifted_input_image
        result[channel] = (torch.sum(torch.sum(prod, dim=0), dim=0))[k_size//2:h-k_size//2, k_size//2:w-k_size//2]

    result[torch.isnan(result)] = 0
    return result


def custom_conv2d_optim_masked(input_image, sigmas, k_size=3, stride=1, padding=0, mask=None):
    original_image = input_image.to(device)
    if padding > 0:
        # modify input_image and sigmas to pad appropriately
        input_image, sigmas = zero_padding(input_image, sigmas, padding)

    c, h, w = input_image.size()

    out_h = (h - (k_size - 1) // stride)
    out_w = (w - (k_size - 1) // stride)

    result = torch.zeros([c, out_h, out_w]).to(device)

    if mask is None:
        mask = torch.ones(c, h, w)
    else:
        new_mask = torch.zeros([h, w])
        mask = torch.from_numpy(mask).to(device)
        new_mask[padding:h - padding, padding:w - padding] = mask
        mask = new_mask.expand(c, h, w)

    # for channel in range(c):
    img = input_image.to(torch.float16)
    # create the copies
    copied_input_image = img.expand(k_size, k_size, c, h, w)
    shifted_input_image = torch.zeros(copied_input_image.size())

    m = mask[0, :, :].to(torch.int8)
    copied_mask = m.expand(k_size, k_size, c, h, w)
    shifted_mask = torch.zeros(copied_mask.size())

    # print(copied_input_image.size())
    for i in range(k_size):
        for j in range(k_size):
            # shift current image, truncate boundaries if required
            center_pixel = k_size // 2
            shift_i = center_pixel - i
            shift_j = center_pixel - j

            shift_start_i = 0 if shift_i < 0 else shift_i
            shift_start_j = 0 if shift_j < 0 else shift_j
            shift_end_i = h + shift_i
            shift_end_j = w + shift_j

            copy_start_i = 0 - shift_i if shift_i < 0 else 0
            copy_start_j = 0 - shift_j if shift_j < 0 else 0

            shifted_input_image[i, j, :, shift_start_i:shift_end_i, shift_start_j:shift_end_j] = \
                copied_input_image[i, j, :, copy_start_i:h - shift_i, copy_start_j:w - shift_j]

            shifted_mask[i, j, :, shift_start_i:shift_end_i, shift_start_j:shift_end_j] = \
                copied_mask[i, j, :, copy_start_i:h - shift_i, copy_start_j:w - shift_j]

    # convolve the shifted images with kernel
    kernels = gkern_tensor(k_size, sigmas)                                      # (k_size, k_size, h, w)
    if c != 1:
        kernels = kernels.unsqueeze(2).repeat(1, 1, c, 1, 1)                        # (k_size, k_size, c, h, w)
    prod = kernels * shifted_mask * shifted_input_image
    result[:] = (torch.sum(torch.sum(prod, dim=0), dim=0) /
                       (torch.sum(torch.sum(kernels * shifted_mask, dim=0), dim=0)))[:, k_size // 2:h - k_size // 2,
                      k_size // 2:w - k_size // 2]

    mask = torch.isnan(result)
    result[mask] = original_image[mask]
    return result


def custom_batched_conv2d_optim_masked(input_image, sigmas, k_size=3, stride=1, padding=0, mask=None):

    batch_size = 20
    if padding > 0:
        # modify input_image and sigmas to pad appropriately
        input_image, sigmas = zero_padding(input_image, sigmas, padding)

    c, h, w = input_image.size()

    out_h = (h - (k_size - 1) // stride)
    out_w = (w - (k_size - 1) // stride)

    result = torch.zeros([c, out_h, out_w]).to(device)

    if mask is None:
        mask = torch.ones(c, h, w)
    else:
        new_mask = torch.zeros([h, w])
        mask = torch.from_numpy(mask).to(device)
        new_mask[padding:h - padding, padding:w - padding] = mask
        mask = new_mask.expand(c, h, w)

    img = input_image


    n_batches = ((k_size) // (batch_size)) + 1
    img_batch = torch.zeros((c, h, w))
    mask_batch = torch.zeros((c, h, w))
    for hor_batch in range(n_batches):
        for ver_batch in range(n_batches):
            # create the copies
            copied_input_image = img.expand(batch_size, batch_size, c, h, w)
            shifted_input_image = torch.zeros(copied_input_image.size())

            m = mask[0, :, :]
            copied_mask = m.expand(batch_size, batch_size, c, h, w)
            shifted_mask = torch.zeros(copied_mask.size())

            for ii in range(batch_size):
                i = hor_batch * batch_size + ii
                if i > k_size:
                    continue

                for jj in range(batch_size):
                    # shift current image, truncate boundaries if required
                    j = ver_batch * batch_size + jj
                    if j > k_size:
                        continue

                    center_pixel = k_size // 2
                    shift_i = center_pixel - i
                    shift_j = center_pixel - j

                    shift_start_i = 0 if shift_i < 0 else shift_i
                    shift_start_j = 0 if shift_j < 0 else shift_j
                    shift_end_i = h + shift_i
                    shift_end_j = w + shift_j

                    copy_start_i = 0 - shift_i if shift_i < 0 else 0
                    copy_start_j = 0 - shift_j if shift_j < 0 else 0

                    shifted_input_image[ii, jj, :, shift_start_i:shift_end_i, shift_start_j:shift_end_j] = \
                        copied_input_image[ii, jj, :, copy_start_i:h - shift_i, copy_start_j:w - shift_j]

                    shifted_mask[ii, jj, :, shift_start_i:shift_end_i, shift_start_j:shift_end_j] = \
                        copied_mask[ii, jj, :, copy_start_i:h - shift_i, copy_start_j:w - shift_j]

                    # utils.save_image(shifted_input_image[ii][jj], img_name=img_name,
                    #                  file_desc='/h_batch_{:04d}_v_batch_{:04d}_i_{:04d}_j_{:04d}_shifted_image'.format(hor_batch, ver_batch, ii, jj), fmt='exr')
                    # utils.save_image(shifted_mask[ii][jj], img_name=img_name,
                    #                  file_desc='/batch_{:04d}_i_{:04d}_j_{:04d}_shifted_mask'.format(batch, ii, jj), fmt='exr')

            # convolve the shifted images with kernel
            kernels = gkern_tensor(batch_size, sigmas)      # (batch_size, batch_size, h, w)
            kernels = kernels.unsqueeze(2).repeat(1, 1, c, 1, 1)
            prod = kernels * shifted_mask * shifted_input_image
            sum_image_batch = torch.sum(torch.sum(prod, dim=0), dim=0)
            sum_mask_batch = torch.sum(torch.sum(kernels * shifted_mask, dim=0), dim=0)
            img_batch += sum_image_batch
            mask_batch += sum_mask_batch

            # utils.save_image(sum_image_batch[0], img_name=img_name, channel=1,
            #                  file_desc='/h_batch_{:04d}_v_batch_{:04d}_shifted_image'.format(hor_batch, ver_batch), fmt='exr')

    result[:] = (img_batch / mask_batch)[:, k_size // 2:h - k_size // 2, k_size // 2:w - k_size // 2]

    result[torch.isnan(result)] = 0
    return result


def gaussian_blur_image(sharp_image, kernel_size=9, sigma=0.5, padding=4):
    filter = gkern(kernel_size, sigma).to(device)

    filters = torch.zeros(3, 3, kernel_size, kernel_size).to(device)
    filters[0, 0, :, :] = filter
    filters[1, 1, :, :] = filter
    filters[2, 2, :, :] = filter

    blurred_image = nn.conv2d(sharp_image.unsqueeze(0), filters, padding=padding)

    return blurred_image

def box_blur_image(sharp_image, kernel_size=9, sigma=0.5, padding=4):
    filter = torch.ones((kernel_size, kernel_size)).to(device)
    filter /= torch.sum(filter)

    filters = torch.zeros(3, 3, kernel_size, kernel_size).to(device)
    filters[0, 0, :, :] = filter
    filters[1, 1, :, :] = filter
    filters[2, 2, :, :] = filter

    blurred_image = nn.conv2d(sharp_image.unsqueeze(0), filters, padding=padding)

    return blurred_image



def filter_pixels(optim_radius, depth_image, sharp_image, k_size=3):
    c, h, w = sharp_image.size()

    shifted_img = torch.zeros((k_size, k_size, c, h, w))

    img_copies = sharp_image.expand(k_size, k_size, c, h, w)

    for i in range(k_size):
        for j in range(k_size):
            # shift current image, truncate boundaries if required
            center_pixel = k_size // 2
            shift_i = center_pixel - i
            shift_j = center_pixel - j

            shift_start_i = 0 if shift_i < 0 else shift_i
            shift_start_j = 0 if shift_j < 0 else shift_j
            shift_end_i = h + shift_i
            shift_end_j = w + shift_j

            copy_start_i = 0 - shift_i if shift_i < 0 else 0
            copy_start_j = 0 - shift_j if shift_j < 0 else 0

            shifted_img[i, j, :, shift_start_i:shift_end_i, shift_start_j:shift_end_j] = \
                img_copies[i, j, :, copy_start_i:h - shift_i, copy_start_j:w - shift_j]


    shifted_img = shifted_img.view(-1, c, h ,w)

    color_var = torch.var(shifted_img, dim=0)

    result_map = torch.where((color_var > (0.001 * color_var.max())), 1, 0)

    return optim_radius, result_map

def compute_patched_error(blurred_image, gt_image, patch_size=3):
    c, h, w = blurred_image.size()

    blurred_copies = blurred_image.expand(patch_size, patch_size, c, h, w)
    shifted_blurred = torch.zeros(blurred_copies.size())

    gt_copies = gt_image.expand(patch_size, patch_size, c, h, w)
    shifted_gt = torch.zeros(gt_copies.size())

    for i in range(patch_size):
        for j in range(patch_size):
            # shift current image, truncate boundaries if required
            center_pixel = patch_size // 2
            shift_i = center_pixel - i
            shift_j = center_pixel - j

            shift_start_i = 0 if shift_i < 0 else shift_i
            shift_start_j = 0 if shift_j < 0 else shift_j
            shift_end_i = h + shift_i
            shift_end_j = w + shift_j

            copy_start_i = 0 - shift_i if shift_i < 0 else 0
            copy_start_j = 0 - shift_j if shift_j < 0 else 0

            shifted_blurred[i, j, :, shift_start_i:shift_end_i, shift_start_j:shift_end_j] = \
                blurred_copies[i, j, :, copy_start_i:h - shift_i, copy_start_j:w - shift_j]

            shifted_gt[i, j, :, shift_start_i:shift_end_i, shift_start_j:shift_end_j] = \
                gt_copies[i, j, :, copy_start_i:h - shift_i, copy_start_j:w - shift_j]


    shifted_gt = shifted_gt.reshape(-1, h, w)
    shifted_blurred = shifted_blurred.reshape(-1, h, w)

    error = torch.mean(((shifted_blurred - shifted_gt)**2), dim=0).to(device)

    return error


def motion_blur_optim(sharp_image, flow_image, kernel_size=3):
    gathered_image = torch.zeros(sharp_image.size()).to(device)

    channel, height, width = sharp_image.size()

    flow_x = flow_image[0, :, :]
    flow_y = flow_image[1, :, :]

    y = torch.arange(0, height).to(device)
    y = y.reshape(1, height).repeat(width, 1)
    y = torch.transpose(y, 0, 1)

    x = torch.arange(0, width).to(device)
    x = x.reshape(1, width).repeat(height, 1)

    mesh_grid = torch.zeros(sharp_image.size())
    deformed_grid = torch.zeros(sharp_image.size())
    mesh_grid[0] = x / width
    mesh_grid[1] = y / height

    random_image = torch.rand([height, width]).to(device)

    for i in range(kernel_size):
        t = ((i + random_image) / kernel_size) - 0.5
        new_y = (t * flow_y).int() # (h, w)
        ind_y = torch.clamp(y + new_y, 0, height-1) # (1, h, w)
        new_x = (t * flow_x).int() # (h, w)
        ind_x = torch.clamp(x + new_x, 0, width-1) # (1, h, w)

        idx = torch.cat([ind_y.unsqueeze(0), ind_x.unsqueeze(0)], dim=0) # (2, h, w)
        deformed_grid[0] = ind_x / width
        deformed_grid[1] = ind_y / height
        idx = torch.floor(idx.reshape(2, -1).permute(1, 0)).long()

        shifted_sharp_image = sharp_image[:, idx[:, 0], idx[:, 1]].reshape(channel, height, width)
        gathered_image += shifted_sharp_image

    blurred_image = (1 / kernel_size) * gathered_image

    return blurred_image


def motion_blur_optim_masked(sharp_image, flow_image, mask, kernel_size=3):
    gathered_image = torch.zeros(sharp_image.size()).to(device)

    channel, height, width = sharp_image.size()
    m_channel, m_height, m_width = mask.size()

    if m_channel == channel:
        if m_height != height or m_width != width:
            print('Mask dimensions do not match image dimension!!')
            exit(-1)
    else:
        mask = mask[0:channel]

    flow_x = flow_image[0, :, :]
    flow_y = flow_image[1, :, :]

    y = torch.arange(0, height).to(device)
    y = y.reshape(1, height).repeat(width, 1)
    y = torch.transpose(y, 0, 1)

    x = torch.arange(0, width).to(device)
    x = x.reshape(1, width).repeat(height, 1)

    random_image = torch.rand([height, width]).to(device)
    kernel_count = torch.zeros(mask.size()).to(device)

    # utils.save_image(flow_x, img_name=img_name, channel=1, file_desc='/flow_u_composite_to_blur', fmt='exr')
    # utils.save_image(flow_y, img_name=img_name, channel=1, file_desc='/flow_v_composite_to_blur', fmt='exr')
    for i in range(kernel_size):
        t = ((i + random_image) / kernel_size) - 0.5
        new_y = (t * flow_y).int() # (h, w)
        ind_y = torch.clamp(y + new_y, 0, height-1) # (1, h, w)
        new_x = (t * flow_x).int() # (h, w)
        ind_x = torch.clamp(x + new_x, 0, width-1) # (1, h, w)
        idx = torch.cat([ind_y.unsqueeze(0), ind_x.unsqueeze(0)], dim=0) # (2, h, w)
        idx = torch.floor(idx.reshape(2, -1).permute(1, 0)).long()
        shifted_sharp_image = sharp_image[:, idx[:, 0], idx[:, 1]].reshape(channel, height, width)
        shifted_mask = mask[:, idx[:, 0], idx[:, 1]].reshape(channel, height, width)
        gathered_image += shifted_sharp_image * shifted_mask
        kernel_count += shifted_mask

    # utils.save_image(gathered_image, img_name=img_name, file_desc='/{:04d}_gathered_image'.format(i), fmt='exr')
    # utils.save_image(kernel_count, img_name=img_name, file_desc='/{:04d}_kernel_count'.format(i), fmt='exr')

    blurred_image = (1 / kernel_count) * gathered_image
    blurred_image[torch.isnan(blurred_image)] = 0
    return blurred_image


def motion_blur_analysis(image, sharp_image, flow_image, kernel_size, synthetic_flow=False, img_name='', debug=False, j=''):
    exposure_range = np.linspace(0, 1, 10).astype('float32')
    optim_strength = 11
    prev_error = 1000
    channel, height, width = image.shape
    # utils.save_image(image, img_name=img_name, file_desc='/mb_optim_target')
    # utils.save_image(sharp_image, img_name=img_name, file_desc='/mb_optim_sharp')
    loss = []
    a = time.perf_counter()
    for strength in exposure_range:
        current_flow_image = strength * flow_image
        mask = np.ones((height, width, 2))
        # outpaint flow for synthetic validation
        # flow_viz = utils.visulize_flow_file(current_flow_image)
        if synthetic_flow:
            level = 6
            if torch.is_tensor(current_flow_image):
                current_flow_image = np.transpose(current_flow_image.detach().cpu().numpy().astype('float32'), (1, 2, 0))
            blurred_current_flow = normalize_blur(current_flow_image, mask, level)
            flow_viz = utils.visulize_flow_file(blurred_current_flow)
            current_flow_image = torch.from_numpy(blurred_current_flow).permute(2, 0, 1).float().to(device)
        blurred_image = motion_blur_optim(sharp_image, current_flow_image, kernel_size=kernel_size)
        current_error = loss_func(blurred_image, image)
        if debug:
            utils.save_image(blurred_image, img_name=img_name, file_desc='/mb_optim_strength_{}_{:.1f}_{:.5f}'.format(j, strength, current_error))
            utils.save_image(flow_viz, img_name=img_name,
                             file_desc='/{}_recovered_flow_{:.1f}'.format(j, strength), np_img=True)

        loss.append(current_error.item())
        if (current_error < prev_error):
            prev_error = current_error
            optim_strength = strength
    b = time.perf_counter()
    # print('Time taken: {:.4f} seconds total'.format(b-a))
    # print('Time taken: {:.4f} seconds per iteration'.format((b - a)/len(exposure_range)))
    return optim_strength


def motion_blur_composite(composite_image, composite_flow_image, mask, kernel_size, flow_mask=None):

    height, width, channel = composite_image.shape
    # Out-paint composite flow to handle edge artifact
    outpaint_kernel_size = int(np.ceil(np.max(np.abs(composite_flow_image))))

    if flow_mask is None:
        composite_mask = mask[:, :, 0:2]
    else:
        composite_mask = flow_mask[:, :, 0:2]
    level = 5
    while (2 ** level) < outpaint_kernel_size:
        level += 1

    blurred_composite_flow = normalize_blur(composite_flow_image, composite_mask, level)
    blurred_composite_flow = torch.from_numpy(blurred_composite_flow).permute(2, 0, 1).float().to(device)

    # Motion blur composite image and mask
    composite_image = torch.from_numpy(composite_image).permute(2, 0, 1).float().to(device)

    final_mask = mask

    mask = torch.from_numpy(mask).permute(2, 0, 1).float().to(device)
    blurred_composite_image_masked = motion_blur_optim_masked(composite_image, blurred_composite_flow, mask,
                                                              kernel_size=kernel_size)

    mask = torch.from_numpy(final_mask).permute(2, 0, 1).float().to(device)
    motion_blurred_mask = motion_blur_optim(mask, blurred_composite_flow, kernel_size=kernel_size)

    # utils.save_image(blurred_composite_image_masked, img_name=img_name, file_desc='/{:04d}_blurred_composite_image_masked'.format(i), fmt='exr')
    # utils.save_image(blurred_mask, img_name=img_name, file_desc='/{:04d}_blurred_mask'.format(iter), fmt='exr')

    # utils.save_image(blurred_composite_image, img_name=img_name,
    #                  file_desc='/{:04d}_blurred_composite_image'.format(i), fmt='png')
    # utils.save_image(blurred_composite_image_masked, img_name=img_name,
    #                  file_desc='/{:04d}_blurred_composite_image_masked'.format(iter), fmt='png')

    # utils.save_image(blurred_mask, img_name=img_name, file_desc='/{:04d}_blurred_mask'.format(i), fmt='png')

    # Composite bg image with blurred composite image using blurred mask

    return blurred_composite_image_masked, motion_blurred_mask, composite_flow_image, blurred_composite_flow


# Fit a Polynomial Spline curve to Optimal Blur Radius on domain of Depth
    # https://soham.dev/posts/polynomial-regression-pytorch/
class PolynomialModel(torch.nn.Module):
    def __init__(self, degree):
        super().__init__()
        self._degree = degree
        self.linear = torch.nn.Linear(self._degree, 1)

    def forward(self, x):
        features = self._polynomial_features(x)
        return self.linear(features)

    def _polynomial_features(self, x):
        x = x.unsqueeze(1)
        return torch.cat([x ** i for i in range(1, self._degree + 1)], 1)


def defocus_blur_analysis_gd(model, gt_image, sharp_image, depth_image, kernel_size):
    dof, depth, sorted_rad = None, None, None
    c, h, w = gt_image.size()

    optimizer = torch.optim.LBFGS(model.parameters(), history_size=10, max_iter=4)

    criterion = torch.nn.MSELoss()
    print(f"Initial weight values: {model.linear.weight}")
    def train_step(model, depth_image, optimizer, criterion):
        running_loss = 0.0

        def closure():
            # Zero gradients
            optimizer.zero_grad()

            # reshape depth to feed to model
            depth = depth_image.reshape(-1)
            # Forward pass
            pred_radii = model(depth)

            # reshape predicted radii
            pred_radii = pred_radii.view(h, w)
            blurred_prediction = custom_conv2d_optim(sharp_image, pred_radii, k_size=kernel_size,
                                                     padding=kernel_size // 2)

            # Compute loss
            loss = criterion(gt_image, blurred_prediction)

            # Backward pass
            loss.backward()

            return loss

        # Update weights
        optimizer.step(closure)

        # Update the running loss
        loss = closure()
        running_loss += loss.item()
        return running_loss

    losses = []
    for epoch in range(20):
        running_loss = train_step(model=model,
                                  depth_image=depth_image,
                                  optimizer=optimizer,
                                  criterion=criterion)
        print(f"Epoch: {epoch + 1:02}/20 Loss: {running_loss:.5e}")
        losses.append(running_loss)


    print(f"After training weight values: {model.linear.weight}")

    return model, dof, depth, sorted_rad


def defocus_blur_analysis(model, gt_image, sharp_image, depth_image, kernel_size, filter_pix=False):
    c, h, w = gt_image.size()
    minimal_error = torch.ones([h, w]).to(device) * 1000000
    optim_radius = torch.zeros([h, w]).to(device)
    blurred_image = torch.ones(gt_image.size()).to(device)
    threshold = 1e-12
    radii = torch.linspace(10, 0.1, steps=100)
    multi_blur = MultiBlur(device=sharp_image.device)

    multi_blur.set_image(sharp_image[None,...], radii[0])


    for radius in radii:
        current_blurred_image = multi_blur.blur(radius)

        diff = (current_blurred_image[0] - gt_image) ** 2
        current_error = torchvision.transforms.GaussianBlur(5, 1000.0)(diff)
        current_error = torch.mean(current_error, dim=0)
        masked_idx = (current_error - minimal_error) < threshold
        optim_radius[masked_idx] = radius
        minimal_error[masked_idx] = current_error[masked_idx]

    #utils.save_image(optim_radius, img_name=img_name, channel=1, file_desc='optim_radii', fmt='exr', cmap='gray')
    optim_radius, filtered_pixels = filter_pixels(optim_radius, depth_image, sharp_image, k_size=kernel_size)
    o_rad = optim_radius.clone().detach().cpu().numpy()
    depth = depth_image.clone().detach().cpu().numpy()
    if filter_pix:
        dof = []
        for i in range(h):
            for j in range(w):
                map = filtered_pixels[:, i, j]
                if not torch.all(map):
                    optim_radius[i, j] = 0
                    filtered_pixels[:, i, j] = 0
                else:
                    dof.append([depth[i, j], o_rad[i, j]])
        dof = np.stack(dof)
        depth = dof[:, 0]
        rad = dof[:, 1]
    else:
        dof = None
        filtered_pixels = filtered_pixels.detach().cpu().numpy()
        filter_mask = np.any(filtered_pixels, axis=0)
        depth = depth[filter_mask].reshape(-1)
        rad = o_rad[filter_mask].reshape(-1)


    indices = np.argsort(depth)
    depth = np.sort(depth)
    sorted_rad = rad[indices]

    def train_step(model, depth, blur_radius, optimizer, criterion):
        running_loss = 0.0

        y_gt = blur_radius.float()

        def closure():
            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            y_pred = model(depth)

            # Compute loss
            loss = criterion(y_pred, y_gt[:, None])

            # Backward pass
            loss.backward()

            return loss

        # Update weights
        optimizer.step(closure)


        # Update the running loss
        loss = closure()
        #
        # if model.linear.weight[0, 1] < 0:
        #     loss += -model.linear.weight[0, 1]*10

        running_loss += loss.item()
        return running_loss

    depth_map = torch.from_numpy(depth).to(device)
    blur_map = torch.from_numpy(sorted_rad).to(device)
    model = model.to(device)

    # print(f"*** Fitting Polynomial of degree: {degree} ***")
    optimizer = torch.optim.LBFGS(model.parameters(), history_size=10, max_iter=4)

    criterion = torch.nn.MSELoss()

    losses = []
    for epoch in range(20):
        running_loss = train_step(model=model,
                                  depth=depth_map,
                                  blur_radius=blur_map,
                                  optimizer=optimizer,
                                  criterion=criterion)
        # print(f"Epoch: {epoch + 1:02}/20 Loss: {running_loss:.5e}")
        losses.append(running_loss)

    # print(f"After training weight values: {model.linear.weight}")

    return model, dof, depth, sorted_rad



def defocus_blur_composite(composite_image, composite_depth, dof_model, dof_minima, dof_maxima,
                           mask, occlusion_mask, final_mask_dilated, kernel_size):
    h, w, c = composite_image.shape
    composite_mask = mask[:, :, 0]
    composite_depth = torch.from_numpy(composite_depth).float().to(device)
    composite_depth = composite_depth.view(-1)

    composite_sigma = ((dof_model(composite_depth) - dof_minima + 1e-3) / (dof_maxima - dof_minima + 1e-3)) * dof_maxima
    composite_sigma = composite_sigma.view(h, w)
    composite_depth = composite_depth.view(h, w)

    # Outpaint blur radii
    composite_sigmas = composite_sigma.expand(3, h, w)

    composite_sigmas = np.transpose(composite_sigmas.detach().cpu().numpy(), (1, 2, 0))
    blurred_composite_sigma = normalize_blur(composite_sigmas, mask)
    blurred_composite_sigma = torch.from_numpy(blurred_composite_sigma).permute(2, 0, 1).float().to(device)
    blurred_composite_sigma = blurred_composite_sigma[0]

    # Blur composite image
    composite_image = torch.from_numpy(composite_image).permute(2, 0, 1).float().to(device)
    blurred_composite_image = custom_conv2d_optim_masked(composite_image, sigmas=blurred_composite_sigma,
                                                         k_size=kernel_size, padding=kernel_size // 2, mask=composite_mask)

    # Blur composite mask
    final_mask = mask * occlusion_mask
    final_mask += final_mask_dilated

    blurred_mask = torch.from_numpy(final_mask).permute(2, 0, 1).float().to(device)
    blurred_composite_mask = custom_conv2d_optim(blurred_mask, sigmas=blurred_composite_sigma, k_size=kernel_size,
                                                 padding=kernel_size // 2)

    return blurred_composite_image, blurred_composite_mask, composite_sigma, blurred_composite_sigma

