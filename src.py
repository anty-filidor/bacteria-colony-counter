import imageio as im
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from skimage import color, filters, exposure, morphology, measure, segmentation, draw
from glob import glob
from helpers import show_gray
from sklearn import cluster
import pandas as pd


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

# mark if you want to track changes
track_changes = False

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

'''
# check depth of images to consider histogram equalisation -> [0, 255]
for image in images.values():
    print(np.unique(image))
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

#####                 #####
####                   ####
###                     ###
##                       ##
# select photo to process #
image_original = images[1]
im = image_original
if track_changes:
    show_gray(im, 'orig')
#                         #
##                       ##
###                     ###
####                   ####
#####                 #####

# apply top hat transform
im = filters.rank.tophat(im, morphology.disk(5))

if track_changes:
    show_gray(im, 'tophat')

# threshold image
thresh = filters.threshold_otsu(im)
im = im > thresh

if track_changes:
    show_gray(im, 'otsu')

# clear borders of image
im = segmentation.clear_border(im)
if track_changes:
    show_gray(im, 'no border')

# apply morphology opening
im = morphology.opening(im)
if track_changes:
    show_gray(im, 'opened')


'''
# to delete
contours = measure.find_contours(im, 0.1)

fig, ax = plt.subplots()
ax.imshow(image_original, cmap='gray')

for n, contour in enumerate(contours):
    ax.plot(contour[:, 1], contour[:, 0], linewidth=1)
plt.axis('off')
plt.show()
'''

# label object on the image
labels = measure.label(im)
new_labels = np.array(labels)

#image_label_overlay = color.label2rgb(labels, image=image_original)
#show_gray(image_label_overlay)

fig, ax = plt.subplots()
ax.imshow(image_original)

# initialise dictionary to keep candidates
candidates_attributes = []

# iterate through all detected regions
for region in measure.regionprops(labels):

    # calculate region of area and its squareability to deny non candidates regions
    minr, minc, maxr, maxc = region.bbox
    squareability_bbox = (maxc - minc) / (maxr - minr)
    relative_area_bbox = region.area/(image_original.shape[0] * image_original.shape[1])

    # if region is a candidate save its attributes and mask of image
    if 0.001 > relative_area_bbox >= 0.0001 and 1.5 >= squareability_bbox >= 0.7:
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        candidates_attributes.append([region.label, region.area, region.convex_area, region.extent])
        #new_labels = np.where(new_labels == region.label, 1, new_labels)
    # if no - delete it
    else:
        new_labels = np.where(new_labels == region.label, 0, new_labels)

ax.set_axis_off()
plt.tight_layout()
plt.show()


show_gray(new_labels >= 1)

# convert attributes to pandas dataframe
candidates_attributes = pd.DataFrame(data=candidates_attributes, columns=['label', 'area', 'convex_area', 'extent'])

# perform clustering of candidates
km = cluster.KMeans(init='random', n_init=10, max_iter=300)
pred = km.fit(candidates_attributes[['area', 'convex_area', 'extent']])

# update candidates dataframe by cluster label
candidates_attributes['class'] = pred.labels_

# visualise clustering effect
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(candidates_attributes.area, candidates_attributes.convex_area, candidates_attributes.extent, marker='o',
           c=candidates_attributes['class'])
ax.set_xlabel('area')
ax.set_ylabel('convex area')
ax.set_zlabel('extent')
ax.set_title('Clustering effect')
plt.tight_layout()
plt.show()


# plot classes on the image
fig, ax = plt.subplots()
ax.imshow(image_original)
for region in measure.regionprops(new_labels):
    minr, minc, maxr, maxc = region.bbox
    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(rect)
ax.set_axis_off()
plt.tight_layout()
plt.show()

