import imageio as im
from glob import glob

from helpers import print_images, show_gray

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import color, filters, exposure, feature, morphology, segmentation, measure

# load data
path_to_dataset = '/Users/michal/PycharmProjects/bacteria_colony_counter/dataset/*.jpg'

images = {}  # dictionary to keep images
paths = glob(path_to_dataset)

for path in paths:
    # save dataset name
    name = int(path.split('/')[-1].split('.')[0])

    image = im.imread(path)
    #image = color.rgb2gray(image)
    #image = (image * 255).astype('uint16')

    images.update({name: image})

# perform filtering

'''
# check depth of images -> [0, 255]
for image in images.values():
    print(np.unique(image))
'''

track_changes = False

image_original = images[1]
im = image_original
print(im.shape)

for i in range(0, im.shape[2]):
    channel = im[:, :, i]

    channel = exposure.equalize_adapthist(channel)
    if track_changes:
        show_gray(im, 'equal')

    channel = filters.median(channel)
    if track_changes:
        show_gray(im, 'median')

    channel = np.where(channel > 100, channel, 0)
    channel = np.where(channel < 201, channel, 0)
    print(np.unique(channel))
    show_gray(channel, i)
    plt.hist(channel.ravel(), bins=265, range=[0, 255])
    plt.show()



'''
#im_otsu = filters.threshold_otsu(im)
#im = im > im_otsu



if track_changes:
    show_gray(im, 'thresh')
'''


'''
im = segmentation.clear_border(im)
if track_changes:
    show_gray(im, 'noborder')

im = morphology.erosion(im)
if track_changes:
    show_gray(im, 'eroded')


labels = measure.label(im)
image_label_overlay = color.label2rgb(labels, image=im)
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(image_original)

for region in measure.regionprops(labels):
    # take regions with large enough areas
    # print(region.area)
    if region.area >= 10:
        # draw rectangle around segmented coins
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

ax.set_axis_off()
plt.tight_layout()
plt.show()
'''



'''
for region in measure.regionprops(labels):
    for prop in region:
        print(prop, region[prop])
'''

'''
blobs_log = feature.blob_log(im, max_sigma=15, min_sigma=10, threshold=.15)
blobs_log[:, 2] = blobs_log[:, 2] * np.sqrt(2)
print(blobs_log)
ax, fig = plt.subplots(1, 1)
ax = plt.imshow(image_original)
for blob in blobs_log:
    if blob[2] > 1.5:
        y, x, r = blob
        c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
        fig.add_patch(c)
plt.title('blobs')
plt.show()
'''

