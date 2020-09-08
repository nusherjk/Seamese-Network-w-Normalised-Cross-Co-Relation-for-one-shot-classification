from config import Config
from network import Convdev, TripletLoss
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
from dataloader import *
if __name__ == '__main__':
	config_data = Config()


	'''for n_iter in range(100):
		writer.add_scalar('Loss/train', np.random.random(), n_iter)
		writer.add_scalar('Loss/test', np.random.random(), n_iter)
		writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
		writer.add_scalar('Accuracy/test', np.random.random(), n_iter)'''

