import numpy as np
import cv2

#Import necessary functions
from matchPics import matchPics
from planarH import computeH_ransac, compositeH
from helper import loadVid
from opts import get_opts



#Write script for Q3.1

def creating_AR(opts):
    # Load video
    ar_source_video = loadVid('../data/ar_source.mov')
    book_video = loadVid('../data/book.mov')

    # Load cv cover image
    cv_cover = cv2.imread('../data/cv_cover.jpg')

    # Check if videos are loaded properly
    if ar_source_video is None or book_video is None or cv_cover is None:
        print("Error: Video loading failed.")
        return

    pass

if __name__ == "__main__":
    opts = get_opts()
    creating_AR(opts)
