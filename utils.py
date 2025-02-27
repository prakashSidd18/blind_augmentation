from time import time
import os
import matplotlib.pyplot as plt
import imageio
import array as ar
import OpenEXR as exr
import Imath
import numpy as np
import cv2
import torch

from PIL import ImageFont, ImageDraw, Image

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
imageio.plugins.freeimage.download()


'''Read optical flow'''


def read_flow_exr_file(path):
    file = exr.InputFile(path)

    # Compute the size
    dw = file.header()['dataWindow']
    sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    # Read the three color channels as 32-bit floats
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    (R, G, B, A) = [ar.array('f', file.channel(Chan, FLOAT)).tolist() for Chan in ("R", "G", "B", "A")]

    # Use R and G channel for forward flow and B and A channel for backward flow
    flow = np.array([B, A]).reshape(2, sz[1], sz[0]).astype('float32')
    flow = np.transpose(flow, (1, 2, 0))

    return flow


def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))


# ref: https://github.com/sampepose/flownet2-tf/
# blob/18f87081db44939414fc4a48834f9e0da3e69f4c/src/flowlib.py#L240
def visulize_flow_file(flow_data):
    # flow_data = readFlow(flow_filename)
    img = flow2img(flow_data)
    # plt.imshow(img)
    # plt.show()
    # img = np.transpose(img, (2, 0, 1))
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def flow2img(flow_data):
    """
    convert optical flow into color image
    :param flow_data:
    :return: color image
    """
    # print(flow_data.shape)
    # print(type(flow_data))
    u = flow_data[:, :, 0]
    v = flow_data[:, :, 1]

    UNKNOW_FLOW_THRESHOLD = 1e7
    pr1 = abs(u) > UNKNOW_FLOW_THRESHOLD
    pr2 = abs(v) > UNKNOW_FLOW_THRESHOLD
    idx_unknown = (pr1 | pr2)
    u[idx_unknown] = v[idx_unknown] = 0

    # get max value in each direction
    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    maxu = max(maxu, np.max(u))
    maxv = max(maxv, np.max(v))
    minu = min(minu, np.min(u))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))
    u = u / maxrad + np.finfo(float).eps
    v = v / maxrad + np.finfo(float).eps

    img = compute_color(u, v)

    idx = np.repeat(idx_unknown[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)



def compute_color(u, v):
    """
    compute optical flow color map
    :param u: horizontal optical flow
    :param v: vertical optical flow
    :return:
    """

    height, width = u.shape
    img = np.zeros((height, width, 3))

    NAN_idx = np.isnan(u) | np.isnan(v)
    u[NAN_idx] = v[NAN_idx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u ** 2 + v ** 2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - NAN_idx)))

    return img


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255

    return colorwheel



def load_images(root):
    folders = {'image':'images', 'denoised':'denoised', 'deblurred':'denoised_demotion_blurred',
               'defocused':'denoised_demotion_blurred_defocused', 'composite':'composites'}
    image_list = os.listdir(os.path.join(root, 'images'))
    noisy_images = []
    image_set = []
    for img in image_list:
        images = {}
        img_name = os.path.basename(img)
        images['filename'] = img_name
        images['filepath_root'] = os.path.abspath(root)
        for folder in folders:
            try:
                image = cv2.imread(os.path.join(root, folders[folder], img_name)) / 255.0
                image = image.astype('float32')
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except:
                image = np.zeros(images['image'].shape)
                image = image.astype('float32')
            images[folder] = image

        residual = images['image'] - images['denoised']
        images['residual'] = residual
        grayscale = cv2.cvtColor(images['denoised'], cv2.COLOR_RGB2GRAY)
        images['grayscale'] = grayscale
        image_set.append(images)

    return image_set


def image_compositing(bg_img, composite, mask, visibility=None):

    if visibility is not None:
        mask = mask * visibility

    synthetic_img = composite * mask
    background_img = bg_img * (1.0 - mask)

    composited_img = synthetic_img + background_img

    return composited_img


def center_crop(img, crop_size):
    h, w, c = img.shape
    if h < crop_size or w < crop_size:
        print('Crop dimensions ({},{}) are too big for image of size ({},{})!!'.format(str(crop_size), str(crop_size), str(h), str(w)))
        return img

    top = h // 2 - crop_size // 2
    left = w // 2 - crop_size // 2

    cropped__img = img[top:top+crop_size, left:left+crop_size, :]

    return cropped__img


def reverse_center_crop(image, shape):
    crop_shape = image.shape
    uncropped_image = np.zeros((shape[0], shape[1], crop_shape[2]))
    start_h = (shape[0] - crop_shape[0])//2
    start_w = (shape[1] - crop_shape[1])//2
    end_h = (shape[0] + crop_shape[0])//2
    end_w = (shape[1] + crop_shape[1])//2
    uncropped_image[start_h:end_h, start_w:end_w, :] = image
    return uncropped_image.astype('float32')


def load_image_np(img_name, file_desc=''):
    fName = "./output/{}_result_{}.npy".format(img_name, file_desc)
    return np.load(fName)
def save_image_np(image, img_name, file_desc='', np_img=False):
    if not np_img:
        channel = image.size()[0]
        if channel == 3:
            image = np.transpose(image.detach().cpu().numpy(), (1, 2, 0))
        elif channel == 1:
            image = image.detach().cpu().numpy()

    np.save("./output/{}_result_{}".format(img_name, file_desc), image)


def create_folder(img_name, file_desc=''):
    base_path = os.path.dirname("./output/{}_result_{}.png".format(img_name, file_desc))
    if not os.path.isdir(base_path):
        os.makedirs(base_path, exist_ok=True)


def save_image(image, img_name, channel=3, file_desc='', fmt='png', cmap='', np_img=False):
    if not np_img:
        if channel == 3:
            image = np.transpose(image.detach().cpu().numpy(), (1, 2, 0))
        elif channel == 1:
            image = image.detach().cpu().numpy()

    base_path = os.path.dirname("./output/{}_result_{}.png".format(img_name, file_desc))
    if not os.path.isdir(base_path):
        os.makedirs(base_path, exist_ok=True)

    if fmt == 'png':
        if cmap != '':
            plt.imsave("./output/{}_result_{}.png".format(img_name, file_desc), (image * 255.0).astype('uint8'), cmap=cmap)
        else:
            plt.imsave("./output/{}_result_{}.png".format(img_name, file_desc), (image * 255.0).astype('uint8'))
    elif fmt == 'exr':
        imageio.imwrite("./output/{}_result_{}.exr".format(img_name, file_desc), image)


def add_text(image, text=""):
    width, height, channel = image.shape
    # font
    font = cv2.FONT_HERSHEY_PLAIN
    # font = ImageFont.truetype("font/arial.ttf", 18)
    # org
    org = (width - (width//3), (height//10))
    # fontScale
    fontScale = 1.0
    # Blue color in BGR
    color = (0, 0, 255)
    # Line thickness of 2 px
    thickness = 2

    image = cv2.putText(image, text, org, font, fontScale, color, thickness, cv2.LINE_AA)

    return image

def tonemap_noise(image):
    image = image.clip(0, 1)
    return torch.pow(image, 1/2.2)


def create_video(frames, img_name, file_desc, height, width, fps=24, flip=False):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter("./output/{}_result_/{}_result_{}.mp4".format(img_name, img_name, file_desc), fourcc, fps, (int(width), int(height)))

    for frame in frames:
        if torch.is_tensor(frame):
            frame = np.transpose(frame.detach().cpu().numpy().astype('float32'), (1, 2, 0))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) * 255.0
        if flip:
            frame = cv2.flip(frame, 1)
        video.write(frame.astype('uint8'))

    video.release()


def create_concat_video(frames_1, frames_2, img_name, file_desc, height, width, fps=24,
                        left_text="Naive", right_text="Ours", axis=1):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    if axis == 1:
        video = cv2.VideoWriter("./output/{}_result_/{}_result_{}.mp4".format(img_name, img_name, file_desc), fourcc, fps,
                                (int(2*width), int(height)))
    else:
        video = cv2.VideoWriter("./output/{}_result_/{}_result_{}.mp4".format(img_name, img_name, file_desc), fourcc, fps,
                                (int(width), int(2*height)))
    horizontal_frames = []
    if len(frames_1) == len(frames_2):
        for iter, frame1 in enumerate(frames_1):
            frame2 = frames_2[iter]
            if torch.is_tensor(frame1):
                frame1 = np.transpose(frame1.detach().cpu().numpy().astype('float32'), (1, 2, 0))
            if torch.is_tensor(frame2):
                frame2 = np.transpose(frame2.detach().cpu().numpy().astype('float32'), (1, 2, 0))

            if axis == 1:
                frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB) * 255.0
                frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB) * 255.0

                # font
                font = cv2.FONT_HERSHEY_COMPLEX
                # font = ImageFont.truetype("arial.ttf", 18)
                # org
                org = ((width // 2) - 50, height - 50)
                # fontScale
                fontScale = 1
                # Blue color in BGR
                color = (0, 0, 255)
                # Line thickness of 2 px
                thickness = 2

                frame1 = cv2.putText(frame1, left_text, org, font, fontScale, color, thickness, cv2.LINE_AA).astype('uint8')
                frame2 = cv2.putText(frame2, right_text, org, font, fontScale, color, thickness, cv2.LINE_AA).astype('uint8')
            frame = np.concatenate((frame1, frame2), axis=axis)
            video.write(frame)
            horizontal_frames.append(frame)
    else:
        print('Cannot create concatenated video! Number of frames are different!')
    video.release()
    return horizontal_frames

def load_image_torch(filename):
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.tensor(image).float() / 255
    image = image.permute(2,0,1)[None,...]
    return image

def load_image_luma_torch(filename):
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    image = torch.tensor(image).float() / 255
    image = image[:,:,0][None,None,:,:]
    return image

def load_image_alphas_torch(filename):
    image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    image = image[:,:,3][None,None,...]
    image = torch.tensor(image).float() / 255
    return image

def save_image_torch(filename, image):
    image = torch.clip(image, 0, 1)
    image_cv = image[0,...].permute(1,2,0) * 255
    image_cv = image_cv.detach().cpu().numpy()
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, image_cv)
