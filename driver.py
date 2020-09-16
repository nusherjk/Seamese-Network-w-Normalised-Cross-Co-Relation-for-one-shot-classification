#Add Inference here
import torch

import numpy as np
from network import *


net = Convdev()
optimizer = optim.Adam(net.parameters(),lr = 0.0005)
PATH = 'ckpts/model140.pt'

checkpoint = torch.load(PATH)
net.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']


