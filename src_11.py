import imageio as im
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage import color, filters, exposure, feature, morphology, segmentation, measure, data, draw
from skimage.transform import hough_circle, hough_circle_peaks
from helpers import show_gray, crop_image

# load data

path_to_dataset = '/Users/michal/PycharmProjects/bacteria_colony_counter/dataset/*.jpg'

images = {}  # dictionary to keep images
paths = glob(path_to_dataset)

for path in paths:
    # save dataset name
    name = int(path.split('/')[-1].split('.')[0])

    file = im.imread(path)
    file = color.rgb2gray(file)
    file = np.array(file * 255).astype('uint16')

    images.update({name: file})

crop_image(images[5], track_progress=True)

