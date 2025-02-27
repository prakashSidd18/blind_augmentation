import torch

class LaplacePyramidGenerator():
    def __init__(self, device=torch.device('cpu')):
        self.device = device
        self.pyrDown_kernel = 1/256 * torch.tensor([\
            [1,4,6,4,1],\
            [4,16,24,16,4],\
            [6,24,36,24,6],\
            [4,16,24,16,4],\
            [1,4,6,4,1],\
            ]).to(device)
        self.pyrDown_kernel_1chan = self.pyrDown_kernel[None,None,...].to(device)
        self.pyrDown_kernel_3chan = torch.zeros([3,3,5,5]).to(device)
        for i in range(3):
            self.pyrDown_kernel_3chan[i,i,...] = self.pyrDown_kernel
        self.pad = torch.nn.ReflectionPad2d(2).to(device)
    def pyrDown(self, image):
        if (image.shape[-1] == 2 or image.shape[-2] == 2):
            return torch.nn.functional.interpolate(image, scale_factor=0.5, mode="area")

        if(image.shape[1] == 1):
            blurred = torch.nn.functional.conv2d(self.pad(image), self.pyrDown_kernel_1chan)
        if(image.shape[1] == 3):
            blurred = torch.nn.functional.conv2d(self.pad(image), self.pyrDown_kernel_3chan)
        return blurred[:,:,::2,::2]
    
    def pyrUp(self, image):
        if image.shape[-1] <= 2 or image.shape[-2] <= 2:
            return torch.nn.functional.interpolate(image, scale_factor=2.0, mode="bilinear")

        output = torch.zeros([image.shape[0], image.shape[1], image.shape[2]*2, image.shape[3]*2]).to(self.device)
        output[:,:,::2,::2] = image
        if(image.shape[1] == 1):
            output = torch.nn.functional.conv2d(self.pad(output), self.pyrDown_kernel_1chan * 4)
        if(image.shape[1] == 3):
            output = torch.nn.functional.conv2d(self.pad(output), self.pyrDown_kernel_3chan * 4)
        return output

    def makeGaussPyramid(self, image, n_levels):
        pyramid = [image]
        for i in range(n_levels-1):
            pyramid.append(self.pyrDown(pyramid[-1]))
        return pyramid
    
    def makeLaplacePyramid(self, image, n_levels):
        pyramid = self.makeGaussPyramid(image, n_levels)
        lPyramid = []
        for i in range(n_levels-1):
            lPyramid.append(pyramid[i] - self.pyrUp(pyramid[i+1]))
        lPyramid.append(pyramid[-1])
        return lPyramid
    
    def reconLaplacePyramid(self, pyramid):
        n_levels = len(pyramid)
        output = pyramid[-1]
        for i in reversed(range(n_levels-1)):
            output = self.pyrUp(output) + pyramid[i]
        return output