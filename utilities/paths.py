#used on Colab
IMAGES_DIR = 'drive/MyDrive/data_scene_flow/'

DISP_DIR = IMAGES_DIR + 'disparity/'
RGB_DIR = IMAGES_DIR + 'RGB/'
GRAY_DIR = IMAGES_DIR + 'GRAY/'

RGB_LEFT_DIR = RGB_DIR + 'left/'
RGB_RIGHT_DIR = RGB_DIR + 'right/'
GRAY_LEFT_DIR = GRAY_DIR + 'left/'
GRAY_RIGHT_DIR = GRAY_DIR + 'right/'

IS_GRAY = False
BNORM = False

PATCH_SIZE = 9
MAX_DISPARITY = 100 #povecat mozda

NUM_IMAGES = 200
TRAIN_START = 0
TRAIN_END = 160
VALID_START = 160
VALID_END = 200


TRAIN_DATA = IMAGES_DIR + 'train/'
VALID_DATA = IMAGES_DIR + 'valid/'
