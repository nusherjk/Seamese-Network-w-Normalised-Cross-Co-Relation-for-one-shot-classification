import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import os
import random


from dataloader import *
from network import *
from scipy.stats import multivariate_normal


def get_test_input(img):
    img = cv2.imread(img)
    img = cv2.resize(img, (128,128))          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    #img_[1] = img[:,:,::-1].transpose((2,0,1))
    #img_ = img_[ :, :, :] / 255.0  # Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    # print(img_.shape)
    img_ = Variable(img_).cuda()                   # Convert to Variable
    return img_



class Config():
    training_dir = "crops/"
    testing_dir = "campus/"
    data_dir = "data/"
    galaryid = "data/gallary_id.txt"
    galarycam = "data/galary_cam.txt"
    query_cam = "data/query_cam.txt"
    query_id = "data/query_id.txt"
    query_gallery_distance = "data/query_gallery_distance.txt"


transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((128, 128)),
    torchvision.transforms.ToTensor()
])


def get_gaussian_mask():
    # 128 is image size
    x, y = np.mgrid[0:1.0:128j, 0:1.0:128j]
    xy = np.column_stack([x.flat, y.flat])
    mu = np.array([0.5, 0.5])
    sigma = np.array([0.22, 0.22])
    covariance = np.diag(sigma ** 2)
    z = multivariate_normal.pdf(xy, mean=mu, cov=covariance)
    z = z.reshape(x.shape)

    z = z / z.max()
    z = z.astype(np.float32)

    mask = torch.from_numpy(z)

    return mask




def get_distance(query_img, gallary_img):
    net = Convdev().cuda()
    optimizer = optim.Adam(net.parameters(),lr = 0.0005)
    PATH = 'ckpts/model40.pt'

    checkpoint = torch.load(PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    """query_img = Image.open(query_img)
    query_img = query_img.convert('RGB')
    query_img = transforms(query_img)

    gallary_img = Image.open(gallary_img)
    gallary_img = gallary_img.convert('RGB')
    gallary_img = transforms(gallary_img)
    """

    query_img = get_test_input(query_img)
    gallary_img = get_test_input(gallary_img)

    gaussian_mask = get_gaussian_mask().cuda()
    out1, out2 = net(query_img*gaussian_mask,gallary_img* gaussian_mask)
    distance = ncc(out1, out2)
    return distance

randomid = random.sample(range(1, 971), 100)


for i in range(len(randomid)):
    """with open(Config.query_id, "a") as nqi:
        nqi.write(str(randomid[i]) + "\n")

    with open(Config.query_cam, "a") as nqi:
        nqi.write(str(randomcam[i]) + "\n")
    """

    query_img = os.path.join(Config.testing_dir, "{0:04d}".format(randomid[i]))

    qimg = random.choices(os.listdir(query_img), k=1)
    if int(qimg[0].split('.')[0])%4 == 0 or int(qimg[0].split('.')[0])%4 == 3:
        qcam = "0"
    else:
        qcam = "1"


    #Store qcam and randomid

    with open(Config.query_id, "a") as nqi:
        nqi.write(str(randomid[i]) + "\n")

    with open(Config.query_cam, "a") as nqi:
        nqi.write(str(qcam) + "\n")


    for gid in os.listdir(Config.testing_dir):
        for gimg in os.listdir(os.path.join(Config.testing_dir, gid)):
            qimg1 = query_img + '/' + qimg[0]
            gimg2 = Config.testing_dir + '/' + gid +'/'+ gimg

            print(qimg1)
            print(gimg2)
            distance = get_distance(qimg1, gimg2)
            print("{0:0.4f}".format(distance.item()))
            with open(Config.query_gallery_distance, "a") as f:
                f.write("{0:0.4f}".format(distance.item()) + " ")


    # Store distance with space.









"""
for f in os.listdir(Config.testing_dir):
    path = os.path.join(Config.testing_dir, f)
    for id in os.listdir(path):
        with open(Config.galaryid, 'a') as file:

            g =str(int(f))
            file.write(g + "\n")
        with open(Config.galarycam, 'a') as file:
            name = id.split('.')[0]
            if(int(name)%4 == 3 or int(name)%4==0):
                file.write("0" + "\n")
            else:
                file.write("1" + "\n")

"""
