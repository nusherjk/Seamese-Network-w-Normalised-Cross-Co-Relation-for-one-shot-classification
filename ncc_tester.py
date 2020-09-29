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
"""
def channel_normalize(template):
    reshaped_template = template.clone().reshape(template.shape[0], -1)
    #reshaped_template.sub(reshaped_template.mean(dim=-1, keepdim=True))
    reshaped_template.div(reshaped_template.std(dim=-1, keepdim=True, unbiased=False))

    return reshaped_template.view_as(template)

"""

def channel_normalize(template):
    template= template.reshape([template.shape[0], template.shape[1]])
    mean = template.mean(dim =-1)
    mean = mean.unsqueeze(1).repeat(1, template.shape[1] )
    stddev = template.std(dim =-1)
    stddev = stddev.unsqueeze(1).repeat(1, template.shape[1] )
    return template.sub(mean).div(stddev)


def ncc(embedding1 , embedding2):
    #embedding1 = embedding1.reshape([embedding1.shape[0], embedding1.shape[1]])
    norm_embedding_1  = channel_normalize(embedding1)
    norm_embedding_2 = channel_normalize(embedding2)
    #print(norm_embedding_2)
    #dot_product = torch.tensordot(norm_embedding_1,norm_embedding_2)
    dot_product = torch.bmm(norm_embedding_1.view(embedding1.shape[0],1,embedding2.shape[1]),
                            norm_embedding_2.view(embedding2.shape[0],embedding2.shape[1], 1))
    dot_product = dot_product.reshape([dot_product.shape[0]]).div(embedding2.shape[1])
    # get resized..

    #dot_product = dot_product.mean(dim=1)

    return  dot_product.abs()

bn = 12

imgAemb = torch.randn([bn,1024,1,1])
imgBemb = torch.randn([bn,1024,1,1])
#imgBemb = imgAemb
"""
imgA = imgAemb.reshape([bn,1024])
imgB = imgBemb.reshape([bn,1024])
#print(imgAemb.reshape([1,1024]))
#imgA = get_test_input(img).reshape([1,3*128*128])#.reshape([1,1024])
#imgB = get_test_input(img).reshape([1,3*128*128]) #.reshape([1,1024])

meanA = imgA.mean(dim=-1)

meanA =  meanA.unsqueeze(1).repeat( 1,1024)
meanB = imgB.mean(dim=-1)
meanB = meanB.unsqueeze(1).repeat(1,1024)
stdA = imgA.std(dim=-1)
stdA = stdA.unsqueeze(1).repeat(1,1024)
stdB = imgB.std(dim=-1)
stdB = stdB.unsqueeze(1).repeat(1,1024)
#nimg = channel_normalize(torchimg)
FA = imgA.sub(meanA).div(stdA)
FB = imgB.sub(meanB).div(stdB)


#ncc = torch.tensordot(FA,FB)
ncc = torch.bmm(FA.view(bn,1,1024),FB.view(bn,1024,1))
print(ncc.div(1024))"""
#t = imgA.div(stdd)
#print(ncc.reshape([bn,1024]).mean(dim=-1))
#print(nimg)
#print(imgAemb)
#print(imgBemb)
print(ncc(imgAemb, imgBemb))

