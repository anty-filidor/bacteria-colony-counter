import imageio as im
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import color, filters, exposure, morphology, measure, segmentation, transform
from glob import glob
from helpers import show_gray

# load data
path_to_dataset = '/Users/michal/PycharmProjects/bacteria_colony_counter/dataset/cropped/*.png'

images = {}  # dictionary to keep images
paths = glob(path_to_dataset)

for path in paths:
    # save dataset name
    name = int(path.split('/')[-1].split('.')[0])

    image = im.imread(path)
    images.update({name: image})
    print(name, image.shape)

# perform filtering

track_changes = True

'''
# check depth of images -> [0, 255]
for image in images.values():
    print(np.unique(image))
'''

'''
# print histograms of each channel
image_original = images[6]
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

    show_gray(channel, i)
    plt.hist(channel.ravel(), bins=265, range=[0, 255])
    plt.show()
'''

# convert images to gray scale and equalise with median filtering
for name, img in images.items():

    # convert image to gray scale
    img = color.rgb2gray(img)
    img = (img * 255).astype('uint16')

    img = exposure.equalize_adapthist(img)
    img = filters.median(img)

    if track_changes and False:
        show_gray(img, name)
        plt.hist(img.ravel(), bins=265, range=[0, 255])
        plt.show()

    images[name] = img

# select photo to process
image_original = images[6]
im = image_original
if track_changes:
    show_gray(im, 'orig')

# threshold image

#fig, ax = filters.try_all_threshold(im, figsize=(20, 16), verbose=False)
#plt.show()

im = filters.rank.tophat(im, morphology.disk(5))

if track_changes:
    show_gray(im, 'tophat')

thresh = filters.threshold_otsu(im)
im = im > thresh

if track_changes:
    show_gray(im, 'otsu')

im = segmentation.clear_border(im)
if track_changes:
    show_gray(im, 'no border')

im = morphology.opening(im)
if track_changes:
    show_gray(im, 'opened')

'''
contours = measure.find_contours(im, 0.1)

fig, ax = plt.subplots()
ax.imshow(image_original, cmap='gray')

for n, contour in enumerate(contours):
    ax.plot(contour[:, 1], contour[:, 0], linewidth=1)
plt.axis('off')
plt.show()
'''


labels = measure.label(im)

#image_label_overlay = color.label2rgb(labels, image=image_original)
#show_gray(image_label_overlay)

#fig, ax = plt.subplots()
new_labels = np.array(labels)
for region in measure.regionprops(labels):
    # take regions with large enough areas
    minr, minc, maxr, maxc = region.bbox
    squareability_bbox = (maxc - minc) / (maxr - minr)
    relative_area_bbox = region.area/(image_original.shape[0] * image_original.shape[1])
    if 0.01 > relative_area_bbox >= 0.0001 and 1.5 >= squareability_bbox >= 0.7:
        #rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
        #ax.add_patch(rect)
        new_labels = np.where(new_labels == region.label, 1, new_labels)
    else:
        new_labels = np.where(new_labels == region.label, 0, new_labels)

#ax.set_axis_off()
#plt.tight_layout()
#plt.show()


print(new_labels.shape)
show_gray(new_labels)

# Detect biggest radius on the image
hough_radius = np.arange(5, 100, 2)
hough_residua = transform.hough_circle(image, hough_radius)
_, cx, cy, r = transform.hough_circle_peaks(hough_residua, hough_radius, total_num_peaks=1)


