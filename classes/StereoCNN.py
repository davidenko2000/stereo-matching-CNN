import torch.nn as nn
import torch.nn.functional as F

class StereoCNN(nn.Module):
	def __init__(self, in_channels=3, features=64, ksize=3, padding=1):
          super().__init__()
          self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=ksize, padding=padding)
          self.conv2 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=ksize, padding=padding)
          self.conv3 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=ksize, padding=padding)
          self.conv4 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=ksize, padding=padding)
	def forward(self, x):
          x = F.relu(self.conv1(x))
          x = F.relu(self.conv2(x))
          x = F.relu(self.conv3(x))
          x = self.conv4(x)
          x = x.squeeze(3).squeeze(2) #dimensions of tensor at second and third dimension are 1, therefore they will be removed
          return  F.normalize(x) #it normalizes vector with euclidean norm
