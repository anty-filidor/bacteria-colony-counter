import imageio as im
from glob import glob

from helpers import print_images, show_gray

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import color, filters, exposure, feature, morphology, segmentation, measure
from scipy import ndimage as ndi

# load data

path_to_dataset = '/Users/michal/PycharmProjects/bacteria_colony_counter/dataset/*.jpg'

images = {}  # dictionary to keep images
paths = glob(path_to_dataset)

for path in paths:
    # save dataset name
    name = int(path.split('/')[-1].split('.')[0])

    image = im.imread(path)
    image = color.rgb2gray(image)
    image = (image * 255).astype('uint16')

    images.update({name: image})

# perform filtering

'''
# check depth of images -> [0, 255]
for image in images.values():
    print(np.unique(image))
'''


im = images[1]
show_gray(im, 'normal')

im = exposure.equalize_adapthist(im)
show_gray(im, 'equal')

im = filters.median(im)
im = filters.median(im)
show_gray(im, 'median')

im_otsu = filters.threshold_otsu(im)
im = im > im_otsu
show_gray(im, 'thresh')

im = morphology.binary_closing(im)
show_gray(im, 'dilated')

im = segmentation.clear_border(im)
show_gray(im, 'noborder')

im = morphology.erosion(im)
show_gray(im, 'eroded')

labels = measure.label(im)
image_label_overlay = color.label2rgb(labels, image=im)
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(images[1])

for region in measure.regionprops(labels):
    # take regions with large enough areas
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
for region in measure.regionprops(labels):
    for prop in region:
        print(prop, region[prop])
'''


#im = morphology.erosion(im)
#show_gray(im, 'eroded')


'''

blobs_log = feature.blob_log(im, max_sigma=10, num_sigma=10, threshold=.1)
blobs_log[:, 2] = blobs_log[:, 2] * np.sqrt(2)
print(blobs_log)
ax, fig = plt.subplots(1, 1)
ax = plt.imshow(im)
for blob in blobs_log:
    y, x, r = blob
    c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
    fig.add_patch(c)
plt.title('blobs')
plt.show()
'''

'''
for name, image in images.items():
    fig, axes = plt.subplots(nrows=2, ncols=2)

    axes[0][0].imshow(image, cmap='gray')

    hist = ndi.histogram(image, bins=256, min=0, max=255)
    axes[1][0].plot(hist)

    thresholds = skimage.filters.(image)
    regions = np.digitize(image, bins=thresholds)
    axes[0][1].imshow(regions, cmap='gray')


    plt.title(name)
    fig.show()
'''

