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
import gc


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
    img_ = Variable(img_).cuda()                 # Convert to Variable
    return img_



class Config():
    training_dir = "campus/train/"
    testing_dir = "campus/test/"
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



def cmc(querys, gallery, topk):
    ret = np.zeros(topk)
    valid_queries = 0
    all_rank = []
    sum_rank = np.zeros(topk)
    for query in querys:
        q_id = query[0]
        q_feature = query[1]
        # Calculate the distances for each query
        distmat = []
        for img, feature in gallery:
            # Get the label from the image
            name = img

            #name,_,_ = get_info(img)  # id of the gallary image.
            dist = get_distance(q_feature, feature)
            distmat.append([name, dist])
            gc.collect()

        # Sort the results for each query
        distmat.sort(key=lambda x: x[1])
        # Find matches
        matches = np.zeros(len(distmat))
        # Zero if no match 1 if match
        for i in range(0, len(distmat)):
            if distmat[i][0] == q_id:
                # Match found
                matches[i] = 1
        rank = np.zeros(topk)
        for i in range(0, topk):
            if matches[i] == 1:
                rank[i] = 1
                # If 1 is found then break as you dont need to look further path k
                break
        all_rank.append(rank)
        valid_queries +=1
    #print(all_rank)
    sum_all_ranks = np.zeros(len(all_rank[0]))
    for i in range(0,len(all_rank)):
        my_array = all_rank[i]
        for g in range(0, len(my_array)):
            sum_all_ranks[g] = sum_all_ranks[g] + my_array[g]
    sum_all_ranks = np.array(sum_all_ranks)
    #print("NPSAR", sum_all_ranks)
    cmc_restuls = np.cumsum(sum_all_ranks) / valid_queries
    print(cmc_restuls)
    return cmc_restuls



def get_distance(query_img, gallary_img):
    net = Convdev().cuda()
    optimizer = optim.Adam(net.parameters(),lr = 0.0005)
    PATH = 'ckpts/model190.pt'

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

    #query_img = get_test_input(query_img)
    #gallary_img = get_test_input(gallary_img)

    gaussian_mask = get_gaussian_mask().cuda()
    out1, out2 = net(query_img*gaussian_mask,gallary_img* gaussian_mask)
    distance = ncc(out1, out2)
    return distance


if __name__ == '__main__':
    querylist = []
    galarylist = []
    topk = 20

    randomid = random.sample(range(1, 485), 100)
    for i in range(len(randomid)):
        randomid[i] = randomid[i]*2

    for i in range(len(randomid)):

        query_img = os.path.join(Config.testing_dir, "{0:04d}".format(randomid[i]))
        q_id = "{0:04d}".format(randomid[i])
        q_path = query_img + '/' + random.choices(os.listdir(query_img), k=1)[0]
        q_feature = get_test_input(q_path)
        querylist.append([q_id, q_feature])
        # Store qcam and randomid

    for gid in os.listdir(Config.testing_dir):
        for gimg in os.listdir(os.path.join(Config.testing_dir, gid)):
            g_id = gid
            g_path = os.path.join(Config.testing_dir, gid)+ '/' + gimg
            g_feature = get_test_input(g_path)
            galarylist.append([g_id, g_feature])
    #print(galarylist)
    cmc(querylist,galarylist,topk)
