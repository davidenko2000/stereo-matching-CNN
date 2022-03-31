import numpy as np
import torch

import utils
import hyperparameters
import paths
import evaluator
from matplotlib import pyplot as plt, colors

def plot_acc_byimage(idx, model)
	real_disp = utils.get_disp_image(idx)
	predicted_disp = utils.compute_disparity_map(idx, model)
	acc = disparity_accuracy_byimage(real_disp=real_disp, predicted_disp=predicted_disp)

	plt.figure(figsize(20,10))
	color_map = colors.ListedColormap(['black', 'red', 'green'])
	plt.imshow(acc, cmap=color_map)
