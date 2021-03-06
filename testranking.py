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
from config import Config
from dataloader import *
from network import *
from scipy.stats import multivariate_normal
from tqdm import tqdm


"""class Config():
    training_dir = "crops/"
    testing_dir = "campus/test/"
"""

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


class Siamese_Triplet_Test(Dataset):

    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)

        negative_images = set()
        while len(negative_images) < 32:  # get 800 negative inputs not 32
            # keep looping till a different class image is found. Negative image.
            img1_tuple = random.choice(self.imageFolderDataset.imgs)
            if img0_tuple[1] == img1_tuple[1]:
                continue
            else:
                negative_images.update([img1_tuple[0]])

        negative_images = list(negative_images)
        # Selecting positive image.
        anchor_image_name = img0_tuple[0].split('/')[-1]
        anchor_class_name = img0_tuple[0].split('/')[-2]
        anchor_class = img0_tuple[0].split('\\')[0]

        #print(self.imageFolderDataset.root)
        all_files_in_class = os.listdir(anchor_class)
        # all_files_in_class = glob.glob(self.imageFolderDataset.root + anchor_class_name + '/*')
        all_files_in_class = [self.imageFolderDataset.root + x[:4] + '/' + x for x in all_files_in_class if
                              x != img0_tuple[0].split('\\')[-1]]

        if len(all_files_in_class) == 0:
            positive_image = img0_tuple[0]
        else:
            positive_image = random.choice(all_files_in_class)
        # print(len(positive_image),anchor_class_name,positive_image)

        #if anchor_class_name != positive_image.split('/')[-2]:
        #    print("Error")

        anchor = Image.open(img0_tuple[0])
        # negative = Image.open(img1_tuple[0])
        positive = Image.open(positive_image)

        anchor = anchor.convert("RGB")
        # negative = negative.convert("RGB")
        positive = positive.convert("RGB")

        if self.should_invert:
            anchor = PIL.ImageOps.invert(anchor)
            positive = PIL.ImageOps.invert(positive)
        # negative = PIL.ImageOps.invert(negative)

        if self.transform is not None:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
        # negative = self.transform(negative)

        negs = []
        for i in range(len(negative_images)):
            neg_image = Image.open(negative_images[i])
            if self.should_invert:
                neg_image = PIL.ImageOps.invert(neg_image)

            if self.transform is not None:
                neg_image = self.transform(neg_image)
            negs.append(neg_image)

        negatives = torch.squeeze(torch.stack(negs))

        return anchor, positive, negatives

    def __len__(self):
        return len(self.imageFolderDataset.imgs)



def main():
    folder_dataset_test = dset.ImageFolder(root=Config.testing_dir)
    #folder_dataset_test = dset.ImageFolder(root=Config.training_dir)
    siamese_dataset = Siamese_Triplet_Test(imageFolderDataset=folder_dataset_test, transform=transforms,
                                           should_invert=False)
    test_dataloader = DataLoader(siamese_dataset, num_workers=0, batch_size=1, shuffle=False)
    dataiter = iter(test_dataloader)
    x0, _, _ = next(dataiter)
    net = Convdev().cuda()
    optimizer = optim.Adam(net.parameters(),lr = 0.0005)
    PATH = 'ckpts/model180.pt'

    checkpoint = torch.load(PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    #net = model.load_state_dict(torch.load(PATH)).cuda()
    net.eval()


