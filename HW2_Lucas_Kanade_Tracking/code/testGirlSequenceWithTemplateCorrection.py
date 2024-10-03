import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade

# Argument parser for customizable parameters
parser = argparse.ArgumentParser()
parser.add_argument(
    '--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade'
)
parser.add_argument(
    '--threshold',
    type=float,
    default=1e-2,
    help='dp threshold of Lucas-Kanade for terminating optimization',
)
parser.add_argument(
    '--template_threshold',
    type=float,
    default=5,
    help='threshold for determining whether to update template',
)
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
template_threshold = args.template_threshold

# load the girl sequence
seq = np.load("../data/girlseq.npy")

# initial template in the first frame
rect_wdc = np.array([280, 152, 330, 318])   # with drift correction   
rect_ndc = rect_wdc.copy()                      # no drift correction
rects_wdc = [rect_wdc.copy()]           
rects_ndc = [rect_ndc.copy()]   

# number of frames for the sequence
num_frames = seq.shape[2]

# initial template in the first frame
It = seq[:, :, 0]
It0 = seq[:, :, 0]

for i in range(num_frames - 1):
    print(f"Tracking at frame {i + 1}/{num_frames - 1}")

    # get the first, current, and next frame images
    It1 = seq[:, :, i + 1]
 
    # run Lucas-Kanade tracking without drift correction
    p_ndc = LucasKanade(It, It1, rect_ndc, threshold, num_iters)

    # update the rectangle coordinates without drift correction
    rect_ndc = rect_ndc + np.array([p_ndc[0], p_ndc[1], p_ndc[0], p_ndc[1]])
    rects_ndc.append(rect_ndc.copy())

    # run Lucas-Kanade tracking (with drift correction)
    p = LucasKanade(It, It1, rect_wdc, threshold, num_iters)

    # update the rectangle coordinates with drift correction
    p_n = np.array(rects_wdc[-1][:2]) - np.array(rects_wdc[0][:2]) + p
    p_star = LucasKanade(It0, It1, rects_wdc[0], threshold, num_iters, p_n)

    # drift correction condition
    if np.linalg.norm(p_star - p_n) <= template_threshold:
        It = It1
        p_star = np.array(rects_wdc[0][:2]) - np.array(rects_wdc[-1][:2]) + p_star
        p = p_star

    # update the rectangle coordinates
    rect_wdc = rect_wdc + np.array([p[0], p[1], p[0], p[1]])
    rects_wdc.append(rect_wdc.copy())

# save the rectangle coordinates to file
rects_wdc = np.array(rects_wdc)         
rects_ndc = np.array(rects_ndc)       
np.save("../results/girlseqrects-wcrt.npy", rects_wdc)
np.save("../results/girlseqrects.npy", rects_ndc)

# visualize the tracking results @ frame 1, 20, 40, 60, 80
frames = [1, 20, 40, 60, 80]
fig, ax = plt.subplots(1, len(frames), figsize=(20, 5))
fig.suptitle("Girl Sequence Tracking Results")
plt.subplots_adjust(wspace=0.1, hspace=0.1)

for i, frame in enumerate(frames):
    ax[i].imshow(seq[:, :, frame], cmap="gray")
    ax[i].axis("off")

    # bounding box without drift correction (blue)
    ax[i].add_patch(
        patches.Rectangle(
            (rects_ndc[frame][0], rects_ndc[frame][1]),
            rects_ndc[frame][2] - rects_ndc[frame][0],
            rects_ndc[frame][3] - rects_ndc[frame][1],
            linewidth=1,
            edgecolor="b",
            facecolor="none",
        )
    )

    # bounding box with drift correction (red)
    ax[i].add_patch(
        patches.Rectangle(
            (rects_wdc[frame][0], rects_wdc[frame][1]),
            rects_wdc[frame][2] - rects_wdc[frame][0],
            rects_wdc[frame][3] - rects_wdc[frame][1],
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
    )
    ax[i].set_title(f"Frame {frame}", fontsize=12)
    ax[i].legend(["Without Drift Correction", "With Drift Correction"])

plt.savefig("../results/Q1.4_girl.png")
plt.show()