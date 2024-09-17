import numpy as np
import cv2

# Import necessary functions
import matplotlib.pyplot as plt
from opts import get_opts
from matchPics import matchPics
from planarH import computeH_ransac, compositeH
from displayMatch import displayMatched


# Q4

def assemble_panorma(img_left_path, img_right_path,opts): 

    # Read images
    img_left = cv2.imread(img_left_path)
    img_right = cv2.imread(img_right_path)

    # Check if images are loaded properly
    if img_left is None or img_right is None:
        print("Error: Image loading failed.")
        return
    
    # Compute homography
    matches, locs1, locs2 = matchPics(img_left, img_right, opts)

    if matches is None or locs1 is None or locs2 is None:
        print("Error: Feature matching failed.")
        return
    
    # Implement RANSAC
    bestH2to1, best_inliers = computeH_ransac(locs1[matches[:,0]][:, [1, 0]], locs2[matches[:,1]][:, [1, 0]], opts)

    # Warp images
    panorama_img = compositeH(bestH2to1, img_left, img_right)

    # display matched features For debugging
    # displayMatched(opts, img_left, img_right)


    # save and display parnorama image
    cv2.imwrite('../results/Q4_result.png', panorama_img)
    cv2.imshow('Panorma Image', panorama_img)
    cv2.waitKey(0)

    pass


if __name__ == "__main__":
    
        opts = get_opts()
        # given images
        # img_left_path = '../data/pano_left.jpg'
        # img_right_path = '../data/pano_right.jpg'

        # my images
        img_left_path = '../data/mypano_left.jpg'
        img_right_path = '../data/mypano_right.jpg'
        assemble_panorma(img_left_path, img_right_path, opts)
