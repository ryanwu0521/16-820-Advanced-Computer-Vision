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

# load the car sequence
seq = np.load("../data/carseq.npy")

# templet in the first frame
rect_ndc = [59, 116, 145, 151]   # no drift correction
rect_wdc = [59, 116, 145, 151]   # with drift correction

# number of frames for the sequence
num_frames = seq.shape[2]

# initialize the rectangle coordinates (no drift correction)
rects_ndc = np.zeros((num_frames, 4))
rects_ndc[0] = rect_ndc.copy()

# initialize the rectangle coordinates (with drift correction)
rects_wdc = np.zeros((num_frames, 4))
rects_wdc[0] = rect_wdc.copy()

# iterate through the frames
for i in range(num_frames - 1):
    print(f"Tracking at frame {i + 1}")

    # get the first, current, and next frame images
    It = seq[:, :, i]
    It1 = seq[:, :, i + 1]
    It0 = seq[:, :, 0]

    # run Lucas-Kanade tracking without drift correction
    p_ndc = LucasKanade(It, It1, rect_ndc, threshold, num_iters)

    # update the rectangle coordinates (no drift correction)
    rect_ndc += np.array([p_ndc[0], p_ndc[1], p_ndc[0], p_ndc[1]])

    # Save the rectangle coordinates (no drift correction)
    rects_ndc[i + 1] = rect_ndc.copy()

    # run Lucas-Kanade tracking (with drift correction)
    p_wdc = LucasKanade(It0, It1, rect_wdc, threshold, num_iters, p_ndc)

    # update the rectangle coordinates (with drift correction)
    rect_wdc += np.array([p_wdc[0], p_wdc[1], p_wdc[0], p_wdc[1]])

    # calculate drift
    drift = rect_wdc - rect_ndc

    # drift correction condition
    if np.linalg.norm(drift) <= template_threshold:
        # apply drift correction
        # rect_wdc += np.array([drift[0], drift[1], drift[0], drift[1]])
        rect_wdc += np.array([p_wdc[0], p_wdc[1], p_wdc[0], p_wdc[1]])
        print(f"Frame {i+1}: Drift correction applied with {drift}")

    else:
        # keep previous template
        rect_wdc = rects_wdc[i].copy()
        print(f"Frame {i+1}: No drift correction applied")
        
    # Save the rectangle coordinates with drift correction
    rects_wdc[i + 1] = rect_wdc.copy()

# save the rectangle coordinates to file
np.save("../results/carseqrects.npy", rects_ndc)
np.save("../results/carseqrects-wcrt.npy", rects_wdc)

# visualize the tracking results @ frame 1, 100, 200, 300, 400 
frames = [1, 100, 200, 300, 400]
fig, ax = plt.subplots(1, len(frames), figsize=(20, 5))
fig.suptitle("Car Sequence Tracking Results") 
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
    ax[i].legend(["No Drift Correction", "With Drift Correction"])

plt.savefig("../results/Q1.4_car.png")
plt.show()
