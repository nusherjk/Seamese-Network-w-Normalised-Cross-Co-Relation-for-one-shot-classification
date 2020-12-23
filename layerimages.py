import cv2
import numpy as np
import torch
import torchvision
from torchvision.utils import save_image, make_grid
from torch.autograd import Variable
from torch import optim
from network import Convdev


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


if __name__ == '__main__':
    input1 =  get_test_input(img)
    input2 =  get_test_input(img)
    net = Convdev()
    optimizer = optim.Adam(net.parameters(),lr = 0.0005)
    PATH = 'ckpts/model20.pt'

    checkpoint = torch.load(PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']


    out1, out2 = net(input1, input2)


