import imageio as im
from glob import glob
import os
import numpy as np
from skimage import color
from helpers import show_gray, crop_image
from tqdm import tqdm

# read path
path_to_dataset = '/Users/michal/PycharmProjects/bacteria_colony_counter/dataset/'

images = {}  # dictionary to keep images
paths = glob(path_to_dataset + '*.jpg')

for path in paths:
    # save dataset name
    name = int(path.split('/')[-1].split('.')[0])

    # read file
    file = im.imread(path)
    file = color.rgb2gray(file)
    file = np.array(file * 255).astype('uint16')

    images.update({name: file})


# crop images
os.mkdir(path_to_dataset + '/cropped')
for name, image in tqdm(images.items()):
    cropped = crop_image(image, track_progress=False)
    show_gray(cropped)
    im.imwrite(path_to_dataset + '/cropped/{}.png'.format(name), cropped)


# demo which shows cropping one image
# show_gray(crop_image(images[6], track_progress=True))
