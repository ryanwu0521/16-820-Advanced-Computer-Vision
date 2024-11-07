import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation
import pickle
import string

import scipy.io

from nn import *
from q4 import *

# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)


def group_bounding_boxes_by_row(bounding_boxes, row_threshold=20):
    """
    Groups bounding boxes into rows based on their vertical position.
    """
    bounding_boxes.sort(key=lambda x: x[2])  # sort by vertical position
    row = [[]]
    current_row_bottom = bboxes[0][2]
    line_num = 1

    for box in bounding_boxes:
        min_row, min_col, max_row, max_col = box
        if min_row >= current_row_bottom + row_threshold:
            current_row_bottom = max_row
            row.append([])  # start a new row
        row[-1].append(box)

    return row


def extract_and_resize_character(image, bounding_box, padding=30, output_size=(32, 32)):
    """
    Extracts a character region from the image using the bounding box, applies padding,
    resizes it to a fixed size, and normalizes pixel values to the range [0, 1].
    """
    min_row, min_col, max_row, max_col = bounding_box
    cropped_char = image[min_row:max_row, min_col:max_col]
    
    # add padding
    padded_char = np.pad(cropped_char,((padding, padding), (padding, padding)), mode='constant', constant_values=1)

    # resize and normalize
    resized_char = skimage.transform.resize(padded_char, output_size).T
    normalized_char = (resized_char - np.min(resized_char)) / (np.max(resized_char) - np.min(resized_char))

    return normalized_char

def predict_characters(character_images, model_params, letters):
    """
    Runs predictions on a list of character images using the provided neural network model parameters.
    """
    predictions = []
    for character_image in character_images:
        flattened_image = character_image.reshape(1, -1)  # flatten image for neural network input
        h = forward(flattened_image, model_params, 'layer1')
        prob = forward(h, model_params, 'output', softmax)

        predicted_index = np.argmax(prob[0, :])
        predictions.append(letters[predicted_index])

    return predictions

letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
params = pickle.load(open("../data/q3_weights.pickle", "rb"))

for img in os.listdir("../images"):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join("../images", img)))
    bboxes, bw = findLetters(im1) 
    bw = 1 - bw  # Invert the image

    plt.imshow(bw, cmap = "gray")  # convert to black and white
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor="red", linewidth=2)
        plt.gca().add_patch(rect)
    plt.show()

    print('\nProcessing image:', img)  # Print the image name

    rows_of_boxes = group_bounding_boxes_by_row(bboxes)
    for row_boxes in rows_of_boxes:
        row_boxes.sort(key=lambda x: x[1]) # sort by horizontal position
        character_images = [extract_and_resize_character(bw, bbox) for bbox in row_boxes]

        predictions = predict_characters(character_images, params, letters)
        print("".join(predictions))