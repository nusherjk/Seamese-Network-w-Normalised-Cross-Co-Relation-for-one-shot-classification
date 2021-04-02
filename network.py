import torch
import torch.nn as nn
import numpy as np

import torchvision
from torchvision.utils import make_grid, save_image
import cv2
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable

# setting default to gpu
if torch.cuda.is_available():
    #torch.set_default_tensor_type('torch.FloatTensor')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Still does not have batch training.[FIXED]

#Work in progress

# similar to nanonets arcitecture

# NCC is not Great!

img = "img.jpg"

def get_test_input(img):
    img = cv2.imread(img)
    img = cv2.resize(img, (128,128))          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    #img_[1] = img[:,:,::-1].transpose((2,0,1))
    #img_ = img_[ :, :, :] / 255.0  # Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    print(img_.shape)
    img_ = Variable(img_).cuda()                   # Convert to Variable
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


'''def channel_normalize(template):
    reshaped_template = template.clone().reshape(template.shape[0], -1)
    reshaped_template.sub_(reshaped_template.mean(dim=-1, keepdim=True))
    reshaped_template.div_(reshaped_template.std(dim=-1, keepdim=True, unbiased=False))

    return reshaped_template.view_as(template)
'''
def channel_normalize(template):
    template= template.reshape([template.shape[0], template.shape[1]])
    mean = template.mean(dim =-1)
    mean = mean.unsqueeze(1).repeat(1, template.shape[1] )
    stddev = template.std(dim =-1)
    stddev = stddev.unsqueeze(1).repeat(1, template.shape[1] )
    return template.sub(mean).div(stddev)


def ncc(embedding1 , embedding2):
    norm_embedding_1  = channel_normalize(embedding1)
    norm_embedding_2 = channel_normalize(embedding2)
    dot_product = torch.bmm(norm_embedding_1.view(embedding1.shape[0],1,embedding2.shape[1]),
                            norm_embedding_2.view(embedding2.shape[0],embedding2.shape[1], 1))
    dot_product = dot_product.reshape([dot_product.shape[0]]).div(embedding2.shape[1])
    #changed to sigmoid function instead of absolute value.
    #return  torch.sigmoid(dot_product)
    return torch.abs(dot_product)

class Ncc(nn.Module):
    def __init__(self, ):
        super(Ncc, self).__init__()
        self.layer = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=5, stride=1, padding=0, bias=False)

    def channl_normalize(self, template):
        template = template.reshape([template.shape[0], template.shape[1]])
        mean = template.mean(dim=-1)
        mean = mean.unsqueeze(1).repeat(1, template.shape[1])
        stddev = template.std(dim=-1)
        stddev = stddev.unsqueeze(1).repeat(1, template.shape[1])
        return template.sub(mean).div(stddev)

    def ncc(self, embedding1, embedding2):
        norm_embedding_1 = self.channl_normalize(embedding1)
        norm_embedding_2 = self.channl_normalize(embedding2)
        dot_product = torch.bmm(norm_embedding_1.view(embedding1.shape[0], 1, embedding2.shape[1]),
                                norm_embedding_2.view(embedding2.shape[0], embedding2.shape[1], 1))
        dot_product = dot_product.reshape([dot_product.shape[0]]).div(embedding2.shape[1])
        # changed to sigmoid function instead of absolute value.
        return torch.sigmoid(dot_product)

    def forward(self, emb1, emb2):
        #return self.ncc(emb1, emb2)
        #out1 = self.layer(emb1)
        #out2 = self.layer(emb2)
        return self.ncc(emb1, emb2)




class Convdev(nn.Module):
    def __init__(self):
        super(Convdev, self).__init__()
        self.layer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0, bias=True)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.layer2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=0, bias=True)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.layer3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=0, bias=True)
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.layer4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=2, padding=0, bias=True)
        self.batchnorm4 = nn.BatchNorm2d(256)
        self.layer5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=2, padding=0, bias=True)
        self.batchnorm5 = nn.BatchNorm2d(256)
        self.layer6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=0, bias=True)
        self.batchnorm6 = nn.BatchNorm2d(512)
        self.layer7 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, stride=2, padding=0, bias=True)
        self.batchnorm7 = nn.BatchNorm2d(1024)
        self.layer8 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=2, padding=0, bias=True)
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


        out = self.lReLU((self.layer8(out)))
        #out = self.pool(out)
        #print(out.shape)
        return out

    def forward(self, input1, input2, input3=None):
        output_1 = self.forward_one(input1)
        output_2 = self.forward_one(input2)

        #output_1 = channel_normalize(output_1)
        #output_2 = channel_normalize(output_2)

        #pos_distance = torch.tensordot(output_1,output_2)
        #pos_distance = pos_distance.reshape([pos_distance.shape[0], pos_distance.shape[1]]).mean(dim=-1)
        #print(output.shape)

        #pos_distance = torch.sum(pos_distance, (2,3))
        #print(output.shape)
        #pos_distance = pos_distance.sum()/1024
        #print(output.shape)
        if input3 != None:
            output_3 = self.forward_one(input3)
            #output_3 = channel_normalize(output_3)
            #print(output_3)
            #neg_distance = torch.tensordot(output_1,output_3)
            #neg_distance = neg_distance.reshape([neg_distance.shape[0], neg_distance.shape[1]]).mean(dim=-1)
            #neg_distance = torch.sum(neg_distance, (2, 3))
            # print(output.shape)
            #neg_distance = neg_distance.sum() / 1024
            #print(neg_distance.shape)
            #return pos_distance.abs(), neg_distance.abs()
            return output_1, output_2, output_3


        return output_1, output_2


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.relu = nn.ReLU()

    def forward(self, anchor, positive, negative ):
        #Have to change this output
        #distance_positive = F.cosine_similarity(anchor, positive)  # Each is batch X 512
        #distance_negative = F.cosine_similarity(anchor, negative)
        #nccval = ncc(anchor, positive)
        #print(nccval)
        distance_positive = 1-ncc(anchor, positive)
        distance_negative = 1-ncc(anchor, negative)


        losses = self.relu((distance_positive) - (distance_negative) + self.margin)

        return losses.sum()




"""
if __name__ == '__main__':
    img1 = get_test_input(img)
    img2 = get_test_input(img)
    model = Convdev().cuda()
    tpl = TripletLoss()

    out1, out2,out3  = model(img1, img2, img2)
    loss = tpl(out1,out2, out3 )
    print(out1.shape)

    print(loss)

    #ncc_value = out1 #torch.sum(out)/n
    #print(ncc_value)
    #print(out2.shape)'''"""