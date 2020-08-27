import torch
import torch.nn as nn
import numpy as np

from NCC import patch_mean, patch_std, channel_normalize
import cv2

from torch.utils.data import DataLoader
from torch.autograd import Variable



img = "img.jpg"

def get_test_input(img):
    img = cv2.imread(img)
    img = cv2.resize(img, (60,160))          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W
    #img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = img_[ np.newaxis,:, :, :] / 255.0  # Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = Variable(img_)                     # Convert to Variable
    return img_



class Convdev(nn.Module):
    def __init__(self):
        super(Convdev, self).__init__()
        self.layer1 = nn.Conv2d(in_channels=3, out_channels=20, kernel_size=5, stride=1, padding=0, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(20)
        self.layer2 = nn.Conv2d(in_channels=20, out_channels=25, kernel_size=5, stride=1, padding=0, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(25)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)





    def forward_one(self, x):

        #out = self.relu(self.batchnorm1(self.layer1(x)))
        out = self.relu(self.layer1(x))
        out = self.pool(out)
        #print(out.shape)
        #out = self.relu(self.batchnorm2(self.layer2(out)))
        out = self.relu(self.layer2(out))
        out = self.pool(out)

        patch_size = 5
        stride = 1

        # out shape = 1 X 25 X 37 X 12 >>> no. of image X no. of channels X height X Width
        # have to run Inexact matching of 5X5 map
        image, channels, *shape_frame = out.shape
        for i in len(image):
            for j in len(channels):
                x = 0
                y = 0
                if (x+patch_size > shape_frame[0]) or (y+patch_size > shape_frame[1]):
                    pass
                else:
                    #calculate the mean and the std of the 5X5 neigbourhood
                    # then calculate F(x,y) for each mean
                    #patchx = [x : x + patch_size]
                    #patchy = [y : y + patch_size]







        return out

    def forward(self, input_1,input_2):
        out_1 = self.forward_one(input_1)
        out_2 = self.forward_one(input_2)


        return out_1, out_2



if __name__ == '__main__':
    img1 = get_test_input(img)
    img2 = get_test_input(img)
    print(img1.shape)
    model = Convdev()
    out1 , out2 = model(img1, img2)
    print(out1.shape)
    #print(out2.shape)


