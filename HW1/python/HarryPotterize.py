import numpy as np
import cv2
import skimage.io 
import skimage.color
from opts import get_opts

# Import necessary functions
from matchPics import matchPics
from planarH import computeH_ransac, compositeH
from displayMatch import displayMatched


# Q2.2.4

def warpImage(opts):
    # Read images
    cv_cover = cv2.imread('../data/cv_cover.jpg')
    cv_desk = cv2.imread('../data/cv_desk.png')
    hp_cover = cv2.imread('../data/hp_cover.jpg')

    # Check if images are loaded properly
    if cv_cover is None or cv_desk is None or hp_cover is None:
        print("Error: Image loading failed.")
        return
    

    # Resize hp_cover to cv_cover size
    hp_cover_resized = cv2.resize(hp_cover, (cv_cover.shape[1], cv_cover.shape[0]))


    # Compute homography
    matches, locs1, locs2 = matchPics(cv_cover, cv_desk, opts)

    if matches is None or locs1 is None or locs2 is None:
        print("Error: Feature matching failed.")
        return

    # Implement RANSAC
    bestH2to1, best_inliers = computeH_ransac(locs1[matches[:,0]][:, [1, 0]], locs2[matches[:,1]][:, [1, 0]], opts)
  
    print(f"Best Homography Matrix:\n{bestH2to1}")
    print(f"Number of inliers: {np.sum(best_inliers)}")

    # Warp images
    composite_img = compositeH(bestH2to1, hp_cover_resized, cv_desk)
    cv2.imwrite('../results/Q2.2.4_result.png', composite_img)

    # display matched features for debugging
    # displayMatched(opts, cv_cover, cv_desk)

    # display images
    cv2.imshow('Composited Image', composite_img)
    cv2.waitKey(0)

    pass



if __name__ == "__main__":

    opts = get_opts()
    warpImage(opts)

