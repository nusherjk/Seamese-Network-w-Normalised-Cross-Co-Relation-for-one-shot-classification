import torch
import cv2
import numpy as np
from torch.autograd import Variable

img = "img.jpg"

def get_test_input(img):
    img = cv2.imread(img)
    img = cv2.resize(img, (128,128))          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    #img_[1] = img[:,:,::-1].transpose((2,0,1))
    #img_ = img_[ :, :, :] / 255.0  # Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    #print(img_.shape)
    img_ = Variable(img_).cuda()                   # Convert to Variable
    return img_

def channel_normalize(template):
    reshaped_template = template.clone().reshape(template.shape[0], -1)
    #reshaped_template.sub(reshaped_template.mean(dim=-1, keepdim=True))
    reshaped_template.div(reshaped_template.std(dim=-1, keepdim=True, unbiased=False))

    return reshaped_template.view_as(template)



def channel_normalize(template):
    mean = template.mean()
    stddev = template.std()
    return template.sub(mean).div(stddev)


def ncc(embedding1 , embedding2):
    norm_embedding_1  = channel_normalize(embedding1)
    norm_embedding_2 = channel_normalize(embedding2)
    dot_product = torch.tensordot(norm_embedding_1,norm_embedding_2)

    # get resized..

    dot_product = dot_product.reshape([dot_product.shape[0], dot_product.shape[1]]).mean(dim=-1)

    return  dot_product.abs()

bn = 12

#imgAemb = torch.randn([bn,1024,1,1])
#imgBemb = torch.randn([bn,1024,1,1])
#print(imgAemb.reshape([1,1024]))
imgA = get_test_input(img)#.reshape([1,3*128*128])#.reshape([1,1024])
imgB = get_test_input(img)#.reshape([1,3*128*128]) #.reshape([1,1024])

#meanA = imgA.mean()
#meanB = imgB.mean()
#stdA = imgA.std()
#stdB = imgB.std()
#nimg = channel_normalize(torchimg)
#FA = imgA.sub(meanA).div(stdA)
#FB = imgB.sub(meanB).div(stdB)
#print(FA.reshape(3*128*128))
#ncc = torch.tensordot(FA,FB)
#print(ncc.item()/(3*128*128))
#t = imgA.div(stdd)
#print(ncc.reshape([bn,1024]).mean(dim=-1))
#print(nimg)

print(ncc(imgA, imgB))
