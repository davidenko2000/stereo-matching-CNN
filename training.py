import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from utilities import paths, hyperparameters
from utilities.PatchesExtractor import PatchesExtractor
from models.StereoCNN import StereoCNN

device = torch.device('cuda' if torch.cuda.is_available())

model = StereoCNN(in_channels=3)
model.to(device)

RGB_transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.4819, 0.5089, 0.5009), (0.3106, 0.3247, 0.3347))
	])
GRAY_transform = transforms.Compose([
	transforms.ToTensor(),
    	transforms.Normalize((0.4999), (0.3180))
	])

train_dataset = PatchesExtractor(train=True, transform=RGB_transform)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=hyperparameters.BATCH_SIZE,
                      shuffle=True, num_workers=4)

criterion = nn.TripletMarginLoss(margin=hyperparameters.MARGIN)
optimizer = torch.optim.Adam(params=model.parameters(), lr=hyperparameters.LR)

train_report = np.zeroes((hyperparameter.EPOCHS, ))
num_batches = 0

def train(epoch):
	print('\nEpoch: %d' % epoch)
	train_loss = 0
	epoch_batches = 0
	model.train()
	for left_path, right_pos_patch, right_neg_patch in train_dataloader:
		num_batches += 1
		epoch_batches += 1
		left_patch, right_pos_patch, right_neg_patch = left_patch.to(device), right_pos_patch.to(device), right_neg_patch.to(device)
		left_output, right_pos_output, right_neg_output = model(left_patch), model(right_pos_patch), model(right_neg_patch)

		optimizer.zero_grad()
            	loss = criterion(left_output, right_pos_output, right_neg_output)
            	loss.backward()
            	optimizer.step()
		train_loss += loss.item()

		if epoch_batches % 100 == 0:
			print('Train -> Loss: %.3f ' % (train_loss/ epoch_batches))

	train[report] = train_loss / epoch_batches
	torch.save(model, f"train_model_{epoch}.pth")

for epoch in range(hyperparameter.EPOCHS):
	train(epoch)
	if epoch == hyperparameters.LR_CHANGE_AT_EPOCH:
        	for param in optimizer.param_groups:
            		param['lr'] = hyperparameters.LR_AFTER_10_EPOCHS

np.save(paths.TRAIN_DATA + 'training_report.npy', train_report)
