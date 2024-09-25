import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade as LKT

# write your script here, we recommend the above libraries for making your animation

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
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold

seq = np.load("../data/girlseq.npy")
rect = [280, 152, 330, 318]

# number of frames for the sequence
num_frames = seq.shape[2]

# initialize the rectangle coordinates
rects = np.zeros((num_frames, 4))

# set the rectangle coordinates for the first frame
rects[0] = rect

# iterate through the frames
for i in range(num_frames - 1):
    # get the current frame and the next frame
    It = seq[:, :, i]
    It1 = seq[:, :, i + 1]

    # run Lucas-Kanade tracking
    p = LKT(It, It1, rect, threshold, num_iters)

    # update the rectangle coordinates for the next frame
    rect[0] += p[0]
    rect[2] += p[0]
    rect[1] += p[1]
    rect[3] += p[1]

    # update template image
    It = It1
    
    # save the rectangle coordinates
    rects[i + 1] = rect.copy()

# save the rectangle coordinates to file
np.save("../results/girlseqrects.npy", rects)

# visualize the tracking results @ frame 1, 20, 40, 60, 80
frames = [1, 20, 40, 60, 80]
fig, ax = plt.subplots(1, len(frames), figsize=(20, 5))
fig.suptitle("Girl Sequence Tracking Results") 
plt.subplots_adjust(wspace=0.1, hspace=0.1)

for i, frame in enumerate(frames):
    ax[i].imshow(seq[:, :, frame], cmap="gray")
    ax[i].axis("off")
    ax[i].add_patch(
        patches.Rectangle(
            (rects[frame][0], rects[frame][1]),
            rects[frame][2] - rects[frame][0],
            rects[frame][3] - rects[frame][1],
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
    )
    ax[i].set_title(f"Frame {frame}", fontsize=12)

plt.savefig("../results/Q1.3_girl.png")
plt.show()
