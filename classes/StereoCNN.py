import torch
import torch.nn as nn
import torch.nn.functional as F

class StereoCNN(nn.Module):
	def __init__(self, in_channels=3, features=64, ksize=3, padding=0):
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
		x = x.squeeze(3).squeeze(2) #dimenzije tenzora na 2. i 3. su 1, stoga ih možemo ukloniti
		return  F.normalize(x) #normalizira vektor, dijeli s euklidskom normom

