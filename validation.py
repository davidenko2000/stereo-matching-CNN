import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from utilities import paths, hyperparameters
from utilities.PatchesExtractor import PatchesExtractor
from models.StereoCNN import StereoCNN

device = torch.device('cuda' if torch.cuda.is_available())

RGB_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4819, 0.5089, 0.5009), (0.3106, 0.3247, 0.3347))
        ])
GRAY_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4999), (0.3180))
        ])

validation_dataset = PatchesExtractor(train=False, transform=RGB_transform)
validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=hyperparameters.BATCH_SIZE,
                      shuffle=True, num_workers=4)

criterion = nn.TripletMarginLoss(margin=hyperparameters.MARGIN)

validation_report = np.zeroes((hyperparameter.EPOCHS, ))

def validate(epoch):
        print('\nValidation epoch: %d' % epoch)
        valid_loss = 0
        epoch_batches = 0
        model = torch.load('./train_model_{epoch}.pth')
	model.eval()
        for left_path, right_pos_patch, right_neg_patch in validation_dataloader:
                epoch_batches += 1
                left_patch, right_pos_patch, right_neg_patch = left_patch.to(device), right_pos_patch.to(device), rig>
                left_output, right_pos_output, right_neg_output = model(left_patch), model(right_pos_patch), model(ri>

                loss = criterion(left_output, right_pos_output, right_neg_output)
                valid_loss += loss.item()

                if epoch_batches % 100 == 0:
                        print('Valid -> Loss: %.3f ' % (valid_loss/ epoch_batches))

        validation[report] = valid_loss / epoch_batches


for epoch in range(hyperparameter.EPOCHS):
        validate(epoch)

np.save(paths.VALID_DATA + 'validation_report.npy', validation_report)
