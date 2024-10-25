import numpy as np
import matplotlib.pyplot as plt

import os

from helper import visualize_keypoints, plot_3d_keypoint, connections_3d, colors
from q3_2_triangulate import triangulate

# Insert your package here

"""
Q6.1 Multi-View Reconstruction of keypoints.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx3 matrix with the 2D image coordinates and confidence per row
            C2, the 3x4 camera matrix
            pts2, the Nx3 matrix with the 2D image coordinates and confidence per row
            C3, the 3x4 camera matrix
            pts3, the Nx3 matrix with the 2D image coordinates and confidence per row
    Output: P, the Nx3 matrix with the corresponding 3D points for each keypoint per row
            err, the reprojection error.

Modified by Vineet Tambe, 2023.
"""


def MultiviewReconstruction(C1, pts1, C2, pts2, C3, pts3, Thres=100, filename="q6_1.npz"):
    # Extract the confidence values
    conf1 = pts1[:, 2]  
    conf2 = pts2[:, 2] 
    conf3 = pts3[:, 2] 

    # Mask for points that meet the threshold
    mask = (conf1 > Thres) & (conf2 > Thres) & (conf3 > Thres)

    # Apply mask to the points
    u1, v1 = pts1[mask][:, 0], pts1[mask][:, 1]
    u2, v2 = pts2[mask][:, 0], pts2[mask][:, 1]
    u3, v3 = pts3[mask][:, 0], pts3[mask][:, 1]

    # Filter the points
    pts1_filtered = np.column_stack((u1, v1))
    pts2_filtered = np.column_stack((u2, v2))
    pts3_filtered = np.column_stack((u3, v3))

    # Initialize the 3D points and error
    total_err = 0
    valid_point_count = np.sum(mask) 

    # Triangulate the 3D points
    P, total_err = triangulate(C1, pts1_filtered, C2, pts2_filtered, C3, pts3_filtered)

    # Calculate the average error
    err = total_err / valid_point_count

    # Save P & Print the results
    np.savez(filename, P=P)
    print("Total Points: " + str(valid_point_count))
    print("Projection Error: " + str(err))

    return P, err


"""
Q6.2 Plot Spatio-temporal (3D) keypoints
    :param car_points: np.array points * 3
"""


def plot_3d_keypoint_video(pts_3d_video):
    # Initialize the figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")
    ax.set_title("Spatio-temporal 3D Reconstruction")

    # Loop through the points
    for i in range(len(pts_3d_video)):
        # Extract the points
        pts_3d = pts_3d_video[i]

        # Plot
        for j in range(len(connections_3d)):
            index0, index1 = connections_3d[j]
            xline = [pts_3d[index0, 0], pts_3d[index1, 0]]
            yline = [pts_3d[index0, 1], pts_3d[index1, 1]]
            zline = [pts_3d[index0, 2], pts_3d[index1, 2]]
            ax.plot(xline, yline, zline, color=colors[j])

    plt.show()
    

# Extra Credit
if __name__ == "__main__":
    pts_3d_video = []
    for loop in range(10):
        print(f"processing time frame - {loop}")

        data_path = os.path.join("data/q6/", "time" + str(loop) + ".npz")
        image1_path = os.path.join("data/q6/", "cam1_time" + str(loop) + ".jpg")
        image2_path = os.path.join("data/q6/", "cam2_time" + str(loop) + ".jpg")
        image3_path = os.path.join("data/q6/", "cam3_time" + str(loop) + ".jpg")

        im1 = plt.imread(image1_path)
        im2 = plt.imread(image2_path)
        im3 = plt.imread(image3_path)

        data = np.load(data_path)
        pts1 = data["pts1"]
        pts2 = data["pts2"]
        pts3 = data["pts3"]

        K1 = data["K1"]
        K2 = data["K2"]
        K3 = data["K3"]

        M1 = data["M1"]
        M2 = data["M2"]
        M3 = data["M3"]

        # Note - Press 'Escape' key to exit img preview and loop further
        img = visualize_keypoints(im2, pts2)

        # TODO: YOUR CODE HERE

        # Q6.1
        # Calcaulte Camera Matrix
        C1 = np.dot(K1, M1)
        C2 = np.dot(K2, M2)
        C3 = np.dot(K3, M3)

        # Multiview Reconstruction
        P, err = MultiviewReconstruction(C1, pts1, C2, pts2, C3, pts3, Thres=10) 

        # Append the 3D keypoints
        pts_3d_video.append(P)

        # Visualize resutls
        plot_3d_keypoint(P)

        # Q6.2
        # Visualize resutls
        plot_3d_keypoint_video(pts_3d_video)
        