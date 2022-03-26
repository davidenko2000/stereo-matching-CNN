import os
import numpy as np
import skimage
import torch
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
import paths

#Method return filename which is made of prefix, which length is 6
def get_filename(idx):
	return str.zfill(str(idx), 6) + "_10.png"

# Method returns a disparity image at index
def get_disp_image(idx):
        return skimage.util.img_as_ubyte(mpimg.imread(paths.DISP_DIR + get_filename(idx)))

# Method return a left image at index (HxWxC)
def get_left_image(idx):
    	return mpimg.imread((paths.GRAY_LEFT_DIR if paths.IS_GRAY else paths.RGB_LEFT_DIR) + get_filename(idx))

# Method return a right image at index (HxWxC)
def get_right_image(idx):
	return mpimg.imread((paths.GRAY_RIGHT_DIR if paths.IS_GRAY else paths.RGB_RIGHT_DIR) + get_filename(idx))

def load_disparity_data(train=True):
    return np.load(paths.TRAIN_DATA if train else paths.VALID_DATA)


def load_model(path=path.TRAINED_DIR):
    model = torch.load(path)
    model.eval()
    return model
