import os
import numpy as np
import skimage
import torch
from matplotlib import image as mpimg
from torch import nn
import torchvision.datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

#Method returns a filename which is made of prefix, which length is 6
def get_filename(idx):
      return str.zfill(str(idx), 6) + "_10.png"

# Method returns an disparity image at index (HxW)
def get_disp_image(idx):
      return skimage.util.img_as_ubyte(mpimg.imread(DISP_DIR + get_filename(idx)))

# Method returns an image at index (HxWxC)
def get_image(idx, is_left):
    	return mpimg.imread((GRAY_LEFT_DIR if IS_GRAY else RGB_LEFT_DIR) + get_filename(idx)) if is_left else mpimg.imread((GRAY_RIGHT_DIR if IS_GRAY else RGB_RIGHT_DIR) + get_filename(idx))
  
#Method which computes a disparity map of the left and right image
def compute_disparity_map(idx, model):
      max_disp = MAX_DISP
      model.to('cuda').eval()

      left_img = get_image(idx, is_left=True)
      right_img = get_image(idx, is_left=False)

      #Adding the padding, as a result the output image will have the same dimensions as input
      padding = PATCH_SIZE // 2
      left_pad = transforms.ToTensor(np.pad(left_img, ((padding, padding),(padding, padding)))).unsqueeze(0)
      right_pad = transforms.ToTensor(np.pad(right_img, ((padding, padding),(padding, padding)))).unsqueeze(0)

      #Making a ndarray of the tensor, which is the output of the model
      left_array = left_pad.squeeze(0).permute(1, 2, 0).detach().numpy()
      right_array = right_pad.squeeze(0).permute(1, 2, 0).detach().numpy()

      #ndarray[HxWxD]
      stacked_disp =  np.stack([np.sum(left_array * np.roll(right_array, d, axis=1) , axis=2) for d in range(max_disp)], axis=2)
      #ndarray[HxW]-using argmax to extract the most similar
      disp_map =  np.argmax(stacked_disp, axis=2)

      return disp_map

#Method which computes mean and standard deviation, used for normalization
def get_mean_std():
      RGB_transform = transforms.Compose([
              transforms.Resize(256),
              transforms.CenterCrop(256),
              transforms.ToTensor()
          ])
      GRAY_transform = transforms.Compose([
          transforms.Resize(256),
          transforms.CenterCrop(256),
          transforms.Grayscale(),
          transforms.ToTensor()
      ])

      images_data = torchvision.datasets.ImageFolder(root=(GRAY_DIR if IS_GRAY else RGB_DIR),
                transform=(GRAY_transform if IS_GRAY else RGB_transform))
      data_loader = DataLoader(images_data, batch_size=len(images_data), shuffle=False, num_workers=0)
      images, _ = next(iter(data_loader))
      mean, std = images.mean([0, 2, 3]), images.std([0, 2, 3])
      return mean, std

#Method returns disparity file, which consists of narrays
def load_disparity_data(train=True):
    	return np.load((TRAIN_DATA if train else VALID_DATA) + 'disparities.npy')

#Method which converts RGB images to grayscale
def convert_to_grayscale():
      for idx in range(NUM_IMAGES):
            filename = get_filename(idx)
            Image.open(RGB_LEFT_DIR + filename).convert("L").save(GRAY_LEFT_DIR + filename)
            Image.open(RGB_RIGHT_DIR + filename).convert("L").save(GRAY_RIGHT_DIR + filename)

#Method which makes narrays and saves disparity data to file
def make_disparity_data(train=True):
      distance = PATCH_SIZE // 2
      def filter(idx):
              disp_image = get_disp_image(idx)
              rows, cols = disp_image.nonzero() #returns non zero pixels (pixels with known disparity)
              rows = rows.astype(np.uint16)
              cols = cols.astype(np.uint16) 
              disparity_values = disp_image[rows, cols]
              
              pos_cols = cols - disparity_values #computes cols with correct disparity
              neg_cols = pos_cols + np.random.choice([-8, -7, -6, -5, -4, 4, 5, 6, 7, 8], size=pos_cols.size).astype(np.uint16) #computes cols with incorrect disparity

              #Calibrations which will be used to discard changes to a pixel which are not allowed
              calibrate_rows = (rows >= distance) & (rows < disp_image.shape[0] - distance)
              calibrate_cols = (cols >= distance) & (cols < disp_image.shape[1] - distance)
              calibrate_pos_cols = (pos_cols >= distance) & (pos_cols < disp_image.shape[1] - distance)
              calibrate_neg_cols = (neg_cols >= distance) & (neg_cols < disp_image.shape[1] - distance)
              calibrations = calibrate_rows & calibrate_cols & calibrate_pos_cols & calibrate_neg_cols

              #Making narray of image indexes and corresponding rows, cols, pos_cols and neg_cols
              rows = rows[calibrations]
              cols = cols[calibrations]
              pos_cols = pos_cols[calibrations]
              neg_cols = neg_cols[calibrations]

              result = np.empty(len(rows), dtype=np.dtype([('idx', 'uint8'), ('row', 'uint16'), ('col', 'uint16'), ('col_pos', 'uint16'), ('col_neg', 'uint16'), ]))
              result['idx'] = np.full(rows.shape, idx, dtype=np.uint8)
              result['row'] = rows
              result['col'] = cols
              result['col_pos'] = pos_cols
              result['col_neg'] = neg_cols

              return result

      disparities = np.concatenate([filter(idx) for idx in range(TRAIN_START if train else VALID_START, TRAIN_END if train else VALID_END)])
      np.save((TRAIN_DATA if train else VALID_DATA) + 'disparities.npy', disparities)
