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
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataloader import *
from network import *
from scipy.stats import multivariate_normal

def show_plot(iteration, loss, path='loss.png'):
    plot, ac = plt.subplots()
    ac.plot(iteration, loss)
    ac.set_title('CMC')
    ac.set_xlabel('rank')
    ac.set_ylabel('Matching rate [%]')

    # plt.show()
    plt.savefig(path)

def takeSecond(elem):
    return elem[1]

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
    training_dir = "prid2011/train/"
    testing_dir = "prid2011/test/"
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



def cmc(querys, gallery,q_cam, g_cam,distance, topk):
    """
    TODO: make  q_cam and g_cam working so that it does not check the same cameras
    :param querys: set of query ids in numpy array
    :param gallery: set of all gallary ids numpy array
    :param q_cam: query camera either 1 or 0
    :param g_cam: gallary camera either 1 or 0
    :param distance: dist[i][j] where i == query_id and j == gallary_id, value: distance
    :param topk: rank top k default = 20
    :return:
    """
    ret = np.zeros(topk)
    valid_queries = 0
    all_rank = []
    sum_rank = np.zeros(topk)
    for query in range(len(querys)):
        q_id = int(querys[query])
        qCam = int(q_cam[query])
        distmat =[]
        for gal in range(len(gallery)):

            gId = int(gallery[gal])

            gCam = int(g_cam[gal])
            if (qCam == gCam):
                valid = False
            else:
                valid = True

            dist = distance[query][gal]
            distmat.append((gId, dist, valid))

        # Sort the results for each query

        distmat.sort(key= takeSecond)
        distmat = distmat[::-1]
        """for id in distmat:
            if(id[1]!=0.9990):
                with(open("work.txt", "a")) as file:
                    file.write('query id: {}, galary_id and distance: {} \n'.format(q_id, id))"""

        # Find matches
        matches = np.zeros(len(distmat))
        # Zero if no match 1 if match
        for i in range(1, len(distmat)):

            if distmat[i][0] == q_id:
                #print('rank: {}'.format(i))

                # Match found
                matches[i-1] = 1
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
    cmc_restuls = np.cumsum(sum_all_ranks) / (valid_queries)
    print(cmc_restuls)
    #print(valid_queries)
    return cmc_restuls



def get_distance(query_img, gallary_img):
    net = Convdev().cuda()
    optimizer = optim.Adam(net.parameters(),lr = 0.0005)
    PATH = 'ckpts/model180.pt'


    net.eval()

    checkpoint = torch.load(PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #epoch = checkpoint['epoch']
    #loss = checkpoint['loss']

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

def tabruns():
    randomid = random.sample(range(1, 192), 100)
    for i in range(len(randomid)):
        randomid[i] = randomid[i]*2



    for f in os.listdir(Config.testing_dir):
        path = os.path.join(Config.testing_dir, f)
        for id in os.listdir(path):
            with open(Config.galaryid, 'a') as file:

                g =str(int(f))
                file.write(g + "\n")
            with open(Config.galarycam, 'a') as file:
                name = id.split('.')[0]
                if(name[4] == '0'):
                    file.write("0" + "\n")
                else:
                    file.write("1" + "\n")


    for i in tqdm(range(len(randomid))):
        #print("working on {} th query".format(i))
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

                #print(qimg1)
                #print(gimg2)
                distance = get_distance(qimg1, gimg2)
                #print("{0:0.4f}".format(distance.item()))
                with open(Config.query_gallery_distance, "a") as f:
                    f.write("{0:0.4f}".format(distance.item()) + " ")

        with open(Config.query_gallery_distance, "a") as f:
            f.write("\n")




def cmc_ranking():
    querys = np.loadtxt(Config.query_id)
    querycams = np.loadtxt(Config.query_cam)
    gallays = np.loadtxt(Config.galaryid)
    gallaycams = np.loadtxt(Config.galarycam)
    distmat = np.loadtxt(Config.query_gallery_distance)
    #print(distmat[0][996])
    #print(distmat[0][998])
    #print(distmat[0][999])
    #print(distmat[0][997])
    #print(np.sort(distmat[0])[::-1])
    """for i in range(len(distmat[0])d):
        
        if(distmat[0][i]==0.9990):
            print(i)
    print(max(distmat[0]))"""
    ranks = cmc(querys,gallays,querycams,gallaycams,distmat,800)
    return ranks



if __name__== '__main__':
    tabruns()
    accuracy = cmc_ranking()
    acc = [ ac*100 for ac in accuracy]
    ranks = [r for r in range(1,len(accuracy)+1)]
    show_plot(ranks,acc, "pridmodel.png")





