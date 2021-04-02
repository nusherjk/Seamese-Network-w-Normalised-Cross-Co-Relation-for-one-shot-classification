# Training codes
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

warnings.filterwarnings("ignore")
writer = SummaryWriter()


def get_gaussian_mask():
    # 128 is image size
    x, y = np.mgrid[0:1.0:128j, 0:1.0:128j]  # 128 is input size.
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


# torch.cuda.empty_cache()

if __name__ == '__main__':

    if (torch.cuda.is_available()):
        # setting default to gpu
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        print("cuda is available")

    print("Load Config")

    # Load Dataloader
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

    # Load Models

    model = Convdev().cuda()
    # model = Convdev()
    criterion = TripletLoss(margin=1.0)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=.0005)

    counter = []
    loss_history = []
    iteration_number = 0

    # load checkpoints
    PATH = 'ckpts/model1500.pt'
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epochs = checkpoint['epoch']
    loss = checkpoint['loss']
    print("previous loss :" + str(loss))

    print("load Dataloader")

    train_dataloader = DataLoader(siamese_dataset, shuffle=True, num_workers=4, batch_size=Config.train_batch_size)
    print("load Dataloader Done")

    # Multiply each image with mask to give attention to center of the image.

    print("load gaussian mask")
    gaussian_mask = get_gaussian_mask().cuda()
    # gaussian_mask = get_gaussian_mask()
    print("load gaussian mask Done")

    for epoch in range(epochs, Config.train_number_epochs):
        print("epoch " + str(epoch))
        # torch.cuda.empty_cache()
        for i, data in enumerate(train_dataloader):
            # print("Step " + str(i))
            # print("Step " + str(i))

            anchor, positive, negative = data
            # print(anchor.shape)
            # print(positive.shape)
            # anchor, positive, negative = anchor.cuda(), positive.cuda(), negative.cuda()
            anchor, positive, negative = anchor.cuda(), positive.cuda(), negative.cuda()

            concatenated = torch.cat((anchor, positive, negative), 0)
            grid = torchvision.utils.make_grid(concatenated)
            # writer.add_image('images', grid, 0)
            # writer.add_graph(model, (anchor, positive, negative) )

            anchor, positive, negative = anchor * gaussian_mask, positive * gaussian_mask, negative * gaussian_mask

            optimizer.zero_grad()

            anchor, positive_out, negative_out = model(anchor, positive, negative)

            triplet_loss = criterion(anchor, positive_out, negative_out)
            # print(triplet_loss)
            triplet_loss.backward()
            optimizer.step()
            # print(triplet_loss)
            # writer.add_scalar('Loss/step', triplet_loss.item(), iteration_number)

            if i % 10 == 0:

                print("Epoch number {}\n Current loss {}\n ".format(epoch, triplet_loss.item()))

                iteration_number += 10
                #counter.append(iteration_number)
                #loss_history.append(triplet_loss.item())
        writer.add_scalar('Loss/epoch', triplet_loss.item(), epoch)
        if epoch % 10 == 0:
            if not os.path.exists('ckpts/'):
                os.mkdir('ckpts')

            PATH = 'ckpts/model' + str(epoch) + '.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': triplet_loss.item(),
            }, PATH)

        gc.collect()
    # torch.save(model,  'ckpts/model' + str(epoch) + '.pt')

    #show_plot(counter, loss_history, path='ckpts/loss.png')
    writer.close()
    '''for n_iter in range(100):

        writer.add_scalar('Loss/test', np.random.random(), n_iter)
        writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
        writer.add_scalar('Accuracy/test', np.random.random(), n_iter)'''