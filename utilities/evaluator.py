import numpy as np
import torch

import utils
import hyperparameters
import paths

"""
Method which uses real disparity, predicted disparity and pixel error to calcuate accuracy of prediction.
Returns:
        -1 -> if the real disparity is unknown
        0  -> if the prediction in incorrect
        1  -> if the prediction is correct (the prediction must be in the interval which uses allowed pixel error
"""
def disparity_accuracy_byimage(pxl_error=hyperparameter.PIXEL_ERROR, real_disp, predicted_disp):
	acc = np.zeroes(real_disp.shape)
	acc[real_disp == 0] = -1
	acc[(real_disp != 0) & (np.abs(predicted_disp - real_disp) < pxl_error)] = 1

	return acc
"""
Method which calculates accuracy percentage, by using real and predicted disparity.
"""
def compute_accuracy(model, train=True):
	counter_correct = 0
	counter_total = 0
	pxl_error=hyperparameter.PIXEL_ERROR
	for idx in range(paths.TRAIN_START if train else paths.VALID_START,paths.TRAIN_END if train else paths.VALID_END):
		real_disp = utils.get_disp_image(idx)
		predicted_disp = utils.compute_disparity_map(idx, model)

		counter_correct += np.count_nonzero((real_disp != 0) & (np.abs(predicted_disp - real_disp) < pxl_error))
		counter_total += np.count_nonzero(real_disp)

	return counter_correct / counter_total
