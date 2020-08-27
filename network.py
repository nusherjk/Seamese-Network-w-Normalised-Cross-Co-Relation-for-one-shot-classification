import torch
import torch.nn as nn
import numpy as np


import cv2

from torch.utils.data import DataLoader
from torch.autograd import Variable


# Still does not have batch training.

#Work in progress

# similar to nanonets arcitecture

# NCC is not Great!

img = "img.jpg"

def get_test_input(img):
    img = cv2.imread(img)
    #img = cv2.resize(img, (64,128))          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    #img_ = img_[ :, :, :] / 255.0  # Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = Variable(img_)                     # Convert to Variable
    return img_





def patch_mean(images, patch_shape):

    channels, *patch_size = patch_shape
    dimensions = len(patch_size)
    padding = tuple(side // 2 for side in patch_size)

    conv = (F.conv1d, F.conv2d, F.conv3d)[dimensions - 1]

    # Convolution with these weights will effectively compute the channel-wise means
    patch_elements = torch.Tensor(patch_size).prod().item()
    weights = torch.full((channels, channels, *patch_size), fill_value=1 / patch_elements)
    weights = weights.to(images.device)

    # Make convolution operate on single channels
    channel_selector = torch.eye(channels).byte()
    weights[1 - channel_selector] = 0

    result = conv(images, weights, padding=padding, bias=None)

    return result


def patch_std(image, patch_shape):
    return (patch_mean(image**2, patch_shape) - patch_mean(image, patch_shape)**2).sqrt()


def channel_normalize(template):
    reshaped_template = template.clone().view(template.shape[0], -1)
    reshaped_template.sub_(reshaped_template.mean(dim=-1, keepdim=True))
    reshaped_template.div_(reshaped_template.std(dim=-1, keepdim=True, unbiased=False))

    return reshaped_template.view_as(template)


class Convdev(nn.Module):
    def __init__(self):
        super(Convdev, self).__init__()
        self.layer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.layer2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=0, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.layer3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=0, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.layer4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=2, padding=0, bias=False)
        self.batchnorm4 = nn.BatchNorm2d(256)
        self.layer5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=2, padding=0, bias=False)
        self.batchnorm5 = nn.BatchNorm2d(256)
        self.layer6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=0, bias=False)
        self.batchnorm6 = nn.BatchNorm2d(512)
        self.layer7 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, stride=2, padding=0, bias=False)
        self.batchnorm7 = nn.BatchNorm2d(1024)
        self.layer8 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=2, padding=0, bias=False)
        self.batchnorm8 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(2,2)
        self.lReLU = nn.LeakyReLU()


    def forward_one(self, x):
        #out = self.lReLU(self.batchnorm1(self.layer1(x)))
        #out = self.pool(out)
        #out = self.lReLU(self.batchnorm2(self.layer2(out)))
        #out = self.pool(out)

        out = self.lReLU(self.batchnorm1(self.layer1(x)))
        out = self.lReLU(self.batchnorm2(self.layer2(out)))
        out = self.lReLU(self.batchnorm3(self.layer3(out)))
        out = self.lReLU(self.batchnorm4(self.layer4(out)))
        out = self.lReLU(self.batchnorm5(self.layer5(out)))
        out = self.lReLU(self.batchnorm6(self.layer6(out)))
        out = self.lReLU(self.batchnorm7(self.layer7(out)))
        #out = self.lReLU(self.batchnorm8(self.layer8(out)))
        out = self.pool(out)
        return out

    def forward(self, input1, input2):
        output_1 = self.forward_one(input1)
        output_2 = self.forward_one(input2)
        output_1 = channel_normalize(output_1)
        output_2 = channel_normalize(output_2)
        #output_1 = output_1.view(1024)
        #output_2 = output_2.view(1024)
        output = torch.tensordot(output_1,output_2)


        return output




'''
class CrossPatch(nn.Module):
    def __init__(self):
        super(CrossPatch, self).__init__()
        self.layer1 = nn.conv2D(in_channels = 1500, out_channels=25, kernel_size= )

'''


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # Outputs batch X 512 X 1 X 1
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            # nn.Dropout2d(p=0.4),

            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            # nn.Dropout2d(p=0.4),

            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            # nn.Dropout2d(p=0.4),


            nn.Conv2d(128, 256, kernel_size=1, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            # nn.Dropout2d(p=0.4),

            nn.Conv2d(256, 256, kernel_size=1, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            # nn.Dropout2d(p=0.4),

            nn.Conv2d(256, 512, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),

            # 1X1 filters to increase dimensions
            nn.Conv2d(512, 1024, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1024),

        )

    def forward_once(self, x):
        output = self.net(x)
        # output = output.view(output.size()[0], -1)
        # output = self.fc(output)

        output = torch.squeeze(output)
        return output

    def forward(self, input1, input2, input3=None):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        if input3 is not None:
            output3 = self.forward_once(input3)
            return output1, output2, output3

        return output1, output2


if __name__ == '__main__':
    img1 = get_test_input(img)
    img2 = get_test_input(img)
    model = Convdev()
    out1  = model(img1, img2)
    out = out1.view(1024)
    n = len(out)
    ncc_value = torch.sum(out)/n
    print(ncc_value)
    #print(out2.shape)