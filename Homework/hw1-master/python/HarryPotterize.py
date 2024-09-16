import numpy as np
import cv2
import skimage.io 
import skimage.color
from opts import get_opts

# Import necessary functions
from matchPics import matchPics
from planarH import computeH_ransac, compositeH

# Q2.2.4

def warpImage(opts):
    # read images
    cv_cover = cv2.imread('../data/cv_cover.jpg')
    cv_desk = cv2.imread('../data/cv_desk.png')
    hp_cover = cv2.imread('../data/hp_cover.jpg')

    # Check if images are loaded properly
    if cv_cover is None or cv_desk is None or hp_cover is None:
        print("Error: One or more images could not be loaded. Please check the file paths.")
        return
    
    # compute homography
    matches, locs1, locs2 = matchPics(cv_cover, cv_desk, opts)

    if matches is None or locs1 is None or locs2 is None:
        print("Error: Feature matching failed.")
        return
    
    bestH2to1, inliers = computeH_ransac(locs1[matches[:, 0]], locs2[matches[:, 1]], opts)

    # warp images
    composite_img = compositeH(bestH2to1, hp_cover, cv_desk)
    cv2.imwrite('../results/Q2.2.4_result.png', composite_img)

    # display images
    cv2.imshow('Composited Image', composite_img)
    cv2.waitKey(0)

    pass



if __name__ == "__main__":

    opts = get_opts()
    warpImage(opts)


