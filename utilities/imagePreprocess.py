import numpy as np
import torch
import torchvision.datasets
from torch.utils.data import DataLoader
import utils
import paths
from torchvision import transforms
from PIL import Image

def convert_to_grayscale():
	for idx in range(config.NUM_IMAGES):
		filename = utils.get_filename(idx)
		Image.open(paths.RGB_LEFT_DIR + filename).convert("L").save(paths.GRAY_LEFT_DIR + filename)
		Image.open(paths.RGB_RIGHT_DIR + filename).convert("L").save(paths.GRAY_RIGHT_DIR + filename)


def disparity_save(train=True):
	patch_size = paths.PATCH_SIZE
	for idx in range(paths.TRAIN_START if train else paths.VALID_START, paths.TRAIN_END if train else paths.VALID_END):
		disp_image = utils.get_disp_image(idx)
		disparities = np.concatenate(result)

	np.save(paths.TRAIN_DATA if train else paths.VALID_DATA, disparities)
