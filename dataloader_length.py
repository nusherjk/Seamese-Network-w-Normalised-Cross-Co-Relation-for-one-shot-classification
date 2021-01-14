

'''net = Convdev()
optimizer = optim.Adam(net.parameters(),lr = 0.0005)
PATH = 'ckpts/model140.pt'

checkpoint = torch.load(PATH)
net.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
'''
#Training codes
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import optim
import torchvision.datasets as dset
from torch.utils.data import DataLoader, Dataset
from scipy.stats import multivariate_normal
import os
import gc

torch.autograd.set_detect_anomaly(True)
from config import Config
from network import Convdev, TripletLoss
from dataloader import *

import warnings
if __name__ == '__main__':

	if (torch.cuda.is_available()):
		# setting default to gpu
		torch.set_default_tensor_type('torch.cuda.FloatTensor')
		print("cuda is available")

	print("Load Config")

	#Load Dataloader
	folder_dataset = dset.ImageFolder(root=Config.training_dir)
	print("load Dataset imagefolder")

	transforms = torchvision.transforms.Compose([
		torchvision.transforms.Resize((128, 128)),  # Important. make size= 128
		torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
		torchvision.transforms.RandomHorizontalFlip(),
		torchvision.transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
		torchvision.transforms.ToTensor()
	])
	print("load Dataset")

	siamese_dataset = SiameseTriplet(imageFolderDataset=folder_dataset, transform=transforms, should_invert=False)
	print("load Dataset Done")

	model = Convdev()
	#model = Convdev()
	criterion = TripletLoss()

	#Optimizer
	optimizer = optim.Adam(model.parameters(),lr = .005)

	counter = []
	loss_history = []
	iteration_number = 0

	print("load Dataloader")

	train_dataloader = DataLoader(siamese_dataset, shuffle=True, num_workers=0, batch_size=Config.train_batch_size)
	print(len(train_dataloader.dataset))

	for i, data in enumerate(train_dataloader):
		# print("Step " + str(i))
		# print("Step " + str(i))

		anchor, positive, negative = data
		print(anchor.shape)
		print(positive.shape)