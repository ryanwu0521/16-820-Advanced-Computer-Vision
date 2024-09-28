import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade

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

# load the car sequence
seq = np.load("../data/carseq.npy")

# initial rectangle coordinates in the first frame
rect = [59, 116, 145, 151]

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
    p = LucasKanade(It, It1, rect, threshold, num_iters)
    
    # debug print
    # print(f"tracking at frame {i + 1}, movement vector: {p}")

    # update the rectangle coordinates for the next frame
    rect[0] += p[0]
    rect[2] += p[0]
    rect[1] += p[1]
    rect[3] += p[1]

    # save the rectangle coordinates
    rects[i + 1] = rect.copy()

# save the rectangle coordinates to file
np.save("../results/carseqrects.npy", rects)

# visualize the tracking results @ frame 1, 100, 200, 300 and 400
frames = [1, 100, 200, 300, 400]
fig, ax = plt.subplots(1, len(frames), figsize=(20, 5))
fig.suptitle("Car Sequence Tracking Results") 
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

plt.savefig("../results/Q1.3_car.png")
plt.show()

