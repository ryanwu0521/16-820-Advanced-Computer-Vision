import numpy as np
import cv2
import scipy.ndimage
import matplotlib.pyplot as plt
from matchPics import matchPics
from opts import get_opts
from helper import plotMatches

#Q2.1.6

def rotTest(opts):

    # TODO: Read the image and convert to grayscale, if necessary
    img = cv2.imread('../data/cv_cover.jpg') 
    img_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Initialize histogram data (rotation angle & number of matches)
    angles = []
    matches = []

    for i in range(36):

        # TODO: Rotate Image
        img_rotated = scipy.ndimage.rotate(img, i*10)

        # TODO: Compute features, descriptors and Match features
        matches_, locs1, locs2 = matchPics(img, img_rotated, opts)
        
        # TODO: Update histogram
        angles.append(i*10)
        matches.append(len(matches_))
        

    pass 


    # TODO: Display histogram
    plt.hist(angles, weights=matches)
    plt.xlabel('Rotation Angle (Degrees)')
    plt.ylabel('Number of Matches')
    plt.title('Number of Feature Matches vs. Rotation Angle')
    plt.show()


    # visualize feature match results at 0, 90, 180, 270 degrees
    for angle in [0, 90, 180, 270]:
        img_rotated = scipy.ndimage.rotate(img, angle)
        matches_, locs1, locs2 = matchPics(img, img_rotated, opts)
        plotMatches(img, img_rotated, matches_, locs1, locs2)
        plt.show()

    return

if __name__ == "__main__":

    opts = get_opts()
    rotTest(opts)