from torch.utils.data import Dataset
from torchvision import transforms
import paths
import utils

class PatchesExtractor(Dataset):

	def __init__(self, train=True, transform):
		self.transform = transform
		self.disparity_data = utils.load_disparity_data(train=train)
		self.left_images = {}
		self.right_images = {}

		for idx in range(paths.TRAIN_START if train else paths.VALID_START, paths.TRAIN_END if train else paths.VALID_END):
			self.left_images[idx] = utils.get_image(idx, is_left=True)
            		self.right_images[idx] = utils.get_image(idx, is_left=False)

	#Method that extracts patches from particular image
	def get_patch_triple(self, patch_idx):
		image_idx, row, col, col_pos, col_neg = self.disparity_data[patch_idx]
		patch_size = paths.PATCH_SIZE
		left_patch = self.transform(image[row - patch_size:row + patch_size, col - patch_size:col + patch_size])
		right_positive_patch = self.transform(image[row - patch_size:row + patch_size, col_pos - patch_size:col_pos + patch_size])
                right_negative_patch = self.transform(image[row - patch_size:row + patch_size, col_neg - patch_size:col_neg + patch_size])

		return left_patch, right_positive_patch, right_negative_patch
