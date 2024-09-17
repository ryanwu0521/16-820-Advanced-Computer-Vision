import numpy as np
import cv2

#Import necessary functions
from matchPics import matchPics
from planarH import computeH_ransac, compositeH
from helper import loadVid
from opts import get_opts

#Write script for Q3.1

# def creating_AR(opts, frame_skip=10):
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
    
    # determine video frame size
    frame_height, frame_width = book_video[0].shape[:2]

    # crop out the top and bottom border of  ar_source video
    ar_source_video = [frame[60:-60, :] for frame in ar_source_video]

    # initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('../results/ar.mp4', fourcc, 20.0, (frame_width, frame_height))

    # make sure the videos have the same number of frames
    num_frames = min(len(ar_source_video), len(book_video))
    print (f"Total number of frames: {num_frames}")

    # Process each frame
    # for i in range(0,num_frames, frame_skip):
    for i in range(num_frames):
        print(f"Processing frame {i}...")
        ar_frame = ar_source_video[i]
        book_frame = book_video[i]

        # crop ar_source video to match cv_cover size
        cv_cover_height, cv_cover_width = cv_cover.shape[:2]
        ar_frame_height, ar_frame_width = ar_frame.shape[:2]

        # determine crop size 
        crop_height = min(cv_cover_height, ar_frame_height)
        crop_width = min(cv_cover_width, ar_frame_width)

        # center crop and match
        ar_frame_crop = ar_frame[(ar_frame_height - crop_height) // 2 : (ar_frame_height + crop_height) // 2, (ar_frame_width - crop_width) // 2 : (ar_frame_width + crop_width) // 2]
        ar_frame_crop = cv2.resize(ar_frame_crop, (cv_cover_width, cv_cover_height))

        # Compute homography
        matches, locs1, locs2 = matchPics(cv_cover, book_frame, opts)

        if matches is None or locs1 is None or locs2 is None:
            print("Error: Feature matching failed.")
            return

        # Implement RANSAC
        try: 
            bestH2to1, best_inliers = computeH_ransac(locs1[matches[:,0]][:, [1, 0]], locs2[matches[:,1]][:, [1, 0]], opts)
        except ValueError as e:
            print(f"Error: {e}")
            continue
        
        # Warp images
        composite_img = compositeH(bestH2to1, ar_frame_crop, book_frame)
        composite_img = cv2.resize(composite_img, (frame_width, frame_height))

        # Write to video
        out.write(composite_img)

        # save every 50 frames
        if i % 50 == 0:
            cv2.imwrite(f'../results/video_frames/frame_{i}.png', composite_img)
    out.release()
    cv2.destroyAllWindows()
    
    pass


if __name__ == "__main__":
    opts = get_opts()
    # creating_AR(opts, frame_skip=0)
    creating_AR(opts)
