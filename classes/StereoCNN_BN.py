import torch.nn as nn
import torch.nn.functional as F

class StereoCNN_BN(nn.Module):
	def __init__(self, in_channels=3, features=64, ksize=3, padding=1):
          super().__init__()
          self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=ksize, padding=padding)
	  self.bn1 = nn.BatchNorm2d(features)
	  self.conv2 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=ksize, padding=padding)
          self.bn2 = nn.BatchNorm2d(features)
	  self.conv3 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=ksize, padding=padding)
          self.bn3 = nn.BatchNorm2d(features)
	  self.conv4 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=ksize, padding=padding)
	  self.bn4 = nn.BatchNorm2d(features)
	def forward(self, x):
          x = F.relu(self.bn1(self.conv1(x)))
          x = F.relu(self.bn2(self.conv2(x)))
          x = F.relu(self.bn3(self.conv3(x)))
          x = self.bn4(self.conv4(x))
          x = x.squeeze(3).squeeze(2)
          return  F.normalize(x)
