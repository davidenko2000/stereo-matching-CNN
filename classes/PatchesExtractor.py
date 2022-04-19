
from torch.utils.data import Dataset

class PatchesExtractor(Dataset):

  def __init__(self, tform, train=True):
          self.transform = tform
          self.disparity_data = load_disparity_data(train=train)
          self.len = self.disparity_data.size
          self.left_images = {}
          self.right_images = {}
          
          for idx in range(TRAIN_START if train else VALID_START, TRAIN_END if train else VALID_END):
                        self.left_images[idx] = get_image(idx, is_left=True)
                        self.right_images[idx] = get_image(idx, is_left=False)
	#Method that extracts patches from particular image
  def __getitem__(self, patch_idx):
          patch_data = self.disparity_data[patch_idx]
          image_idx = patch_data['idx']
          row = patch_data['row']
          col = patch_data['col']
          col_pos = patch_data['col_pos']
          col_neg = patch_data['col_neg']

          patch_size = PATCH_SIZE
          left_image = self.left_images[image_idx]
          right_pos_image = self.right_images[image_idx]
          rigth_neg_image = self.right_images[image_idx]

          if self.transform:
              left_patch = self.transform(left_image[(row - patch_size // 2):(row + patch_size//2 + 1 ), (col - patch_size // 2):(col + patch_size//2 + 1)])
              right_positive_patch = self.transform(right_pos_image[(row - patch_size // 2):(row + patch_size//2 + 1 ), (col_pos - patch_size // 2):(col_pos + patch_size//2 + 1)])
              right_negative_patch = self.transform(rigth_neg_image[(row - patch_size // 2):(row + patch_size//2 + 1 ), (col_neg - patch_size // 2):(col_neg + patch_size//2 + 1)])
                                                
          return left_patch, right_positive_patch, right_negative_patch
  def __len__(self):
    return self.len
