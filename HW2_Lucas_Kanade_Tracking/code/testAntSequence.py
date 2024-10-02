import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from SubtractDominantMotion import SubtractDominantMotion, SubtractDominantMotionInverse

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument(
    '--num_iters', type=int, default=1e10, help='number of iterations of Lucas-Kanade'
)
parser.add_argument(
    '--threshold',
    type=float,
    default=5,
    help='dp threshold of Lucas-Kanade for terminating optimization',
)
parser.add_argument(
    '--tolerance',
    type=float,
    default=0.016,
    help='binary threshold of intensity difference when computing the mask',
)
parser.add_argument(
    '--seq_file',
    default='../data/antseq.npy',
)
parser.add_argument(
    '--inverseMotion', action='store_true', help='use SubtractDominantMotionInverse if set'
)

args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
tolerance = args.tolerance
seq_file = args.seq_file
inverse = args.inverseMotion

seq = np.load(seq_file)

'''
HINT:
1. Create an empty array 'masks' to store the motion masks for each frame.
2. Set the initial mask for the first frame to False.
3. Use the SubtractDominantMotion function to compute the motion mask between consecutive frames.
4. Use the motion 'masks; array for visualization.
'''
# intialize the masks
masks = []

# set the initial mask for the first frame to False
mask = np.zeros(seq.shape[1:], dtype=bool)

# total number of frames
num_frames = seq.shape[2]

# iterate over the frames
for i in range(seq.shape[2] - 1):
    print(f"Tracking frame: {i + 1}/{num_frames - 1}")
    if inverse:
        mask = SubtractDominantMotionInverse(seq[:,:,i], seq[:,:,i+1], threshold, num_iters, tolerance)
    else:
        # compute the motion mask between consecutive frames
        mask = SubtractDominantMotion(seq[:,:,i], seq[:,:,i+1], threshold, num_iters, tolerance)
    
    masks.append(mask)

# visualize the motion masks @ frames 30, 60, 90, 120
frames = [30, 60, 90, 120]
fig, ax = plt.subplots(1, len(frames), figsize=(20, 5))
fig.suptitle("Ant Sequence Tracking Results") 
plt.subplots_adjust(wspace=0.1, hspace=0.1)

for i, frame in enumerate(frames):
    ax[i].imshow(seq[:,:,frame], cmap='gray')
    ax[i].imshow(masks[frame], alpha=0.6, cmap='Blues')
    ax[i].axis('off')
    ax[i].set_title('Frame {}'.format(frame))

if inverse:
    output_file = "../results/Q3.1_ant.png"
else:
    output_file = "../results/Q2.3_ant.png"

plt.savefig(output_file)
plt.show()

# command to run the script
# python testAntSequence.py --num_iters 100000000000 --threshold 5 --tolerance 0.016 --seq_file ../data/antseq.npy
# python testAntSequence.py --num_iters 100000000000 --threshold 5 --tolerance 0.016 --seq_file ../data/antseq.npy --inverse
