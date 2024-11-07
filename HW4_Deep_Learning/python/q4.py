import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation
import matplotlib.pyplot as plt


# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> grayscale -> threshold -> morphology -> label -> skip small boxes
    # this can be 10 to 15 lines of code using skimage functions

    # Denoise (Gaussian blur)
    denoised = skimage.filters.gaussian(image, sigma=1)
    # Denoise (Bilateral filter)
    # denoised = skimage.restoration.denoise_bilateral(image, sigma_color=0.05, sigma_spatial=15, channel_axis=-1)

    # Convert to greyscale
    greyscale = skimage.color.rgb2gray(denoised)

    # Threshold
    threshold = skimage.filters.threshold_otsu(greyscale)
    binary = greyscale < threshold

    # Morphology
    bw = skimage.morphology.dilation(binary, skimage.morphology.square(5))
    bw = skimage.morphology.closing(binary, skimage.morphology.disk(3))

    # Label
    label_image = skimage.measure.label(bw, background=0, connectivity=2)

    # Skip small boxes
    for region in skimage.measure.regionprops(label_image):
        if region.area < 100:
            continue

        minr, minc, maxr, maxc = region.bbox
        bboxes.append((minr, minc, maxr, maxc))

    return bboxes, bw