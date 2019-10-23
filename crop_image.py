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
    images.update({name: file})


# crop images
os.mkdir(path_to_dataset + '/cropped')
for name, image in tqdm(images.items()):
    # convert image to gray scale
    img_gray = np.array(color.rgb2gray(image) * 255).astype('uint16')
    # perform cropping on gray image
    cropped, coords = crop_image(img_gray, track_progress=False)
    show_gray(cropped, 'cropped')
    # crop original image
    cropped_original = image[coords[0]:coords[1], coords[2]:coords[3]]
    # save it to file
    im.imwrite(path_to_dataset + '/cropped/{}.png'.format(name), cropped_original)

'''
# demo which shows cropping one image
img = images[2]
img_gray = np.array(color.rgb2gray(img) * 255).astype('uint16')
cropped, coords = crop_image(img_gray, track_progress=True)
show_gray(img[coords[0]:coords[1], coords[2]:coords[3]])
'''
