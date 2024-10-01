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

# load the girl sequence
seq = np.load("../data/girlseq.npy")

# templet in the first frame
rect_nc = [280, 152, 330, 318]   # no drift correction
rect_wc = [280, 152, 330, 318]   # with drift correction

# initialize the rectangle coordinates without drift correction
rects_nc = np.zeros((seq.shape[2], 4))
rects_nc[0] = rect_nc.copy()

# initialize the rectangle coordinates with drift correction
rects_wcrt = np.zeros((seq.shape[2], 4))
rects_wcrt[0] = rect_wc.copy()

# number of frames for the sequence
num_frames = seq.shape[2]

# iterate through the frames
for i in range(num_frames - 1):
    print(f"tracking at frame {i + 1}")
    # get the current frame and the next frame
    It = seq[:, :, i]
    It1 = seq[:, :, i + 1]

    # run Lucas-Kanade tracking without drift correction
    p = LucasKanade(It, It1, rect_nc, threshold, num_iters)

    # update the rectangle coordinates without drift correction
    rect_nc[0] += p[0]
    rect_nc[2] += p[0]
    rect_nc[1] += p[1]
    rect_nc[3] += p[1]

    # Save the rectangle coordinates without drift correction
    rects_nc[i + 1] = rect_nc.copy()

    # Update the rectangle coordinates with drift correction
    rect_prev_wc = rect_wc.copy()
    rect_wc[0] += p[0]
    rect_wc[2] += p[0]
    rect_wc[1] += p[1]
    rect_wc[3] += p[1]

    # calculate drift
    p_n = p + [rect_prev_wc[0] - rect_wc[0], rect_prev_wc[1] - rect_wc[1]]
    # p_n = [rect_wc[0] - rect_prev_wc[0], rect_wc[1] - rect_prev_wc[1]]
    # p_n = np.array(p + [rect_prev_wc[0] - rect_wc[0], rect_prev_wc[1] - rect_wc[1]])

    # update the rectangle coordinates with drift correction
    # p_n = drift

    # run Lucas-Kanade tracking with drift correction
    p_n_star = LucasKanade(It, It1, rect_wc, threshold, num_iters, p_n)

    # drift correction condition
    if np.linalg.norm(p_n_star - p_n) <= template_threshold:
        drift_correction = p_n_star - p_n
        rect_wc[0] += drift_correction[0]
        rect_wc[2] += drift_correction[0]
        rect_wc[1] += drift_correction[1]
        rect_wc[3] += drift_correction[1]
    else:
        # keep previous template
        rect_wc = rect_prev_wc.copy()
        
    # Save the rectangle coordinates with drift correction
    rects_wcrt[i + 1] = rect_wc.copy()

# save the rectangle coordinates to file
np.save("../results/girlseqrects.npy", rects_nc)
np.save("../results/girlseqrects-wcrt.npy", rects_wcrt)

# visualize the tracking results @ frame 1, 20, 40, 60, 80 
frames = [1, 20, 40, 60, 80]
fig, ax = plt.subplots(1, len(frames), figsize=(20, 5))
fig.suptitle("Girl Sequence Tracking Results") 
plt.subplots_adjust(wspace=0.1, hspace=0.1)

for i, frame in enumerate(frames):
    ax[i].imshow(seq[:, :, frame], cmap="gray")
    ax[i].axis("off")

    # without drift correction
    ax[i].add_patch(
        patches.Rectangle(
            (rects_nc[frame][0], rects_nc[frame][1]),
            rects_nc[frame][2] - rects_nc[frame][0],
            rects_nc[frame][3] - rects_nc[frame][1],
            linewidth=1,
            edgecolor="b",
            facecolor="none",
        )
    )

    # with drift correction
    ax[i].add_patch(
        patches.Rectangle(
            (rects_wcrt[frame][0], rects_wcrt[frame][1]),
            rects_wcrt[frame][2] - rects_wcrt[frame][0],
            rects_wcrt[frame][3] - rects_wcrt[frame][1],
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
    )
    ax[i].set_title(f"Frame {frame}", fontsize=12)
    ax[i].legend(["No Drift Correction", "With Drift Correction"])

plt.savefig("../results/Q1.4_girl.png")
plt.show()