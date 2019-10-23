import matplotlib.pyplot as plt
from skimage import filters, morphology, transform, color, draw
import numpy as np


def print_images(images_dict):
    for name, image in images_dict.items():
        plt.imshow(image)
        plt.title(name)
        plt.axis('off')
        plt.show()


def show_gray(image, name='img'):
    """
    This function plots gray image. It is just shortcut
    :param image: image to plot
    :param name: title of image
    """
    plt.imshow(image, cmap='gray')
    plt.title(name)
    plt.axis('off')
    plt.show()


def crop_image(image_to_crop, track_progress=False):
    """
    This function crops image to size of detected petri dish. We assume that petri dish is on the image and it is the
    biggest object.
    :param image_to_crop: image to crop
    :param track_progress: boolean value; default False; if true stats and images from each step of processing are
    being shown
    :return: cropped image
    """

    # preprocess image
    image = filters.median(image_to_crop)
    th = filters.threshold_otsu(image)
    image = image > th
    image = morphology.dilation(image)
    if track_progress:
        show_gray(image, 'median filter, otsu thresh, dilat')
    image = filters.sobel(image)
    if track_progress:
        show_gray(image, 'sobel edge detection')

    # Estimate radius of petri dish
    x, y = image.shape
    r_x = 0.7 * (x/2)
    r_y = 0.7 * (y/2)
    min_radius = min(r_x, r_y)
    max_radius = max(x/2, y/2)

    # Detect biggest radius on the image
    hough_radius = np.arange(min_radius, max_radius, 2)
    hough_residua = transform.hough_circle(image, hough_radius)
    _, cx, cy, r = transform.hough_circle_peaks(hough_residua, hough_radius, total_num_peaks=1)

    # Draw petri dish
    if track_progress:
        fig, ax = plt.subplots(ncols=1, nrows=1)
        image = color.gray2rgb(image_to_crop)
        circ_y, circ_x = draw.circle_perimeter(int(cy), int(cx), int(r), shape=image.shape)
        image[circ_y, circ_x] = (220, 20, 20)
        ax.imshow(image, cmap=plt.cm.gray)
        plt.title('Detected petri dish')
        plt.show()

    # Crop image
    if track_progress:
        print('Image shape: ', image_to_crop.shape)
        print('Detected dish: x_center - ', int(cx), ' y_center - ', int(cy), ' radius - ', int(r))
        print('Cropping to shape: [', abs(int(cy - r)), ':', int(cy + r), ',', abs(int(cx - r)), ':', int(cx + r), ']')

    final_image = image_to_crop[abs(int(cy - r)):int(cy + r), abs(int(cx - r)):int(cx + r)]

    if track_progress:
        show_gray(final_image, 'cropped')

    return final_image


