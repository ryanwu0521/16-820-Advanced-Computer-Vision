import numpy as np
import matplotlib.pyplot as plt

from helper import displayEpipolarF, calc_epi_error, toHomogenous
from q2_1_eightpoint import eightpoint
from q2_2_sevenpoint import sevenpoint
from q3_2_triangulate import findM2

import scipy

# Insert your package here


# Helper functions for this assignment. DO NOT MODIFY!!!
"""
Helper functions.

Written by Chen Kong, 2018.
Modified by Zhengyi (Zen) Luo, 2021
"""


def plot_3D_dual(P_before, P_after):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Blue: before; red: after")
    ax.scatter(P_before[:, 0], P_before[:, 1], P_before[:, 2], c="blue")
    ax.scatter(P_after[:, 0], P_after[:, 1], P_after[:, 2], c="red")
    while True:
        x, y = plt.ginput(1, mouse_stop=2)[0]
        plt.draw()


"""
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
            nIters, Number of iterations of the Ransac
            tol, tolerence for inliers
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers

    Hints:
    (1) You can use the calc_epi_error from q1 with threshold to calcualte inliers. Tune the threshold based on 
        the results/expected number of inliners. You can also define your own metric. 
    (2) Use the seven point alogrithm to estimate the fundamental matrix as done in q1
    (3) Choose the resulting F that has the most number of inliers
    (4) You can increase the nIters to bigger/smaller values
 
"""


def ransacF(pts1, pts2, M, nIters=1000, tol=10):
    # Initialize variables
    num = pts1.shape[0]  # number of points
    # handle special case when number of points is less than 8
    if num < 8:
        raise ValueError("Number of points is less than 8")
    
    # Initialize the best F, best inliers, max_inliers
    best_F = None
    best_inliers = np.zeros(pts1.shape[0], dtype=bool)
    max_inliers = 0

    # Convert to homogeneous coordinates
    pts1_homo = toHomogenous(pts1)
    pts2_homo = toHomogenous(pts2)

    for i in range(nIters):
        # Randomly choose points (7 or 8)
        # idx = np.random.choice(num, 7, replace=False)
        idx = np.random.choice(num, 8, replace=False)
        pts1_sample = pts1[idx]
        pts2_sample = pts2[idx]

        # Compute the fundamental matrix (7 or 8 point algorithm)
        # F = sevenpoint(pts1_sample, pts2_sample, M)
        F = eightpoint(pts1_sample, pts2_sample, M)

        # Calculate the error
        epi_error = calc_epi_error(pts1_homo, pts2_homo, F)

        # Determine inliers
        inliers = epi_error < tol
        inlier_count = np.sum(inliers)

        # Update the best F and inliers
        if inlier_count > max_inliers:
            max_inliers = inlier_count
            best_F = F
            best_inliers = inliers

    return best_F, best_inliers


"""
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
"""


def rodrigues(r):
    theta = np.linalg.norm(r)
    
    # No rotation, return identity
    if theta == 0: 
        R = np.eye(3)
        return R
    
    # Unit vector
    u = r / theta
    
    # Skew-symmetric matric u_x
    u_x = np.array([[0, -u[2], u[1]],
                  [u[2], 0, -u[0]],
                  [-u[1], u[0], 0]])
    
    # Reshape for matrix multiplication
    u = u.reshape(-1, 1)

    # Rodrigues rotation formula
    I = np.eye(3)
    R = I * np.cos(theta) + (1 - np.cos(theta)) * (u @ u.T) + np.sin(theta) * u_x
        
    return R


"""
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
"""


def invRodrigues(R):
    # Compute and construct varibles: A, rho, s, c, theta
    A = (R - R.T) / 2
    rho = np.array([A[2, 1], A[0, 2], A[1, 0]])
    s = np.linalg.norm(rho)
    c = (np.trace(R) - 1) / 2
    theta = np.arctan2(s, c)

    # Special Case
    if s == 0 and c == 1:
        r = np.zeros(3)  # no rotation
        return r
    
    elif s == 0 and c == -1:
        # 180 degree rotation
        R_plus_I = R + np.eye(3)
        v = R_plus_I[:, np.argmax(np.diag(R_plus_I))]
        u = v / np.linalg.norm(v)
        r = u * np.pi

        # Sign flip condition
        if np.linalg.norm(r) == np.pi and \
           ((r[0] == 0 and r[1] == 0 and r[2] < 0) or
            (r[0] == 0 and r[1] < 0) or
            (r[0] < 0)):
            r = -r
        return r
        
    # General case 
    elif np.sin(theta) != 0:   
        u = rho / s
        r = u * theta
        return r


"""
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
"""


def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # TODO: Replace pass by your implementation
    pass


"""
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
            o1, the starting objective function value with the initial input
            o2, the ending objective function value after bundle adjustment

    Hints:
    (1) Use the scipy.optimize.minimize function to minimize the objective function, rodriguesResidual. 
        You can try different (method='..') in scipy.optimize.minimize for best results. 
"""


def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    obj_start = obj_end = 0
    # ----- TODO -----
    # YOUR CODE HERE
    raise NotImplementedError()
    return M2, P, obj_start, obj_end


if __name__ == "__main__":
    np.random.seed(1)  # Added for testing, can be commented out

    some_corresp_noisy = np.load("data/some_corresp_noisy.npz")  # Loading correspondences
    intrinsics = np.load("data/intrinsics.npz")  # Loading the intrinscis of the camera
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    noisy_pts1, noisy_pts2 = some_corresp_noisy["pts1"], some_corresp_noisy["pts2"]
    im1 = plt.imread("data/im1.png")
    im2 = plt.imread("data/im2.png")
            
    
    "Test the ransacF function with various parameters" 
    # (tol)
    # F, inliers = ransacF(noisy_pts1, noisy_pts2, M=np.max([*im1.shape, *im2.shape]), nIters=1000, tol=1)
    # print(f"RANSAC with nIters = 1000, tol = 1")

    # F, inliers = ransacF(noisy_pts1, noisy_pts2, M=np.max([*im1.shape, *im2.shape]), nIters=1000, tol=5)
    # print(f"RANSAC with nIters = 1000, tol = 5")

    F, inliers = ransacF(noisy_pts1, noisy_pts2, M=np.max([*im1.shape, *im2.shape]), nIters=1000, tol=10)
    print(f"RANSAC with nIters = 1000, tol = 10")

    # F, inliers = ransacF(noisy_pts1, noisy_pts2, M=np.max([*im1.shape, *im2.shape]), nIters=1000, tol=15)
    # print(f"RANSAC with nIters = 1000, tol = 15")

    # # (nIters)
    # F, inliers = ransacF(noisy_pts1, noisy_pts2, M=np.max([*im1.shape, *im2.shape]), nIters=100, tol=10)
    # print(f"RANSAC with nIters = 100, tol = 10")

    # F, inliers = ransacF(noisy_pts1, noisy_pts2, M=np.max([*im1.shape, *im2.shape]), nIters=500, tol=10)
    # print(f"RANSAC with nIters = 500, tol = 10")

    # F, inliers = ransacF(noisy_pts1, noisy_pts2, M=np.max([*im1.shape, *im2.shape]), nIters=1000, tol=10)
    # print(f"RANSAC with nIters = 1000, tol = 10")

    # F, inliers = ransacF(noisy_pts1, noisy_pts2, M=np.max([*im1.shape, *im2.shape]), nIters=1500, tol=10)
    # print(f"RANSAC with nIters = 1500, tol = 10")
    
    
    print(f"Inlier count: {np.sum(inliers)}")

    # Calculate the projection error
    epi_error = calc_epi_error(toHomogenous(noisy_pts1), toHomogenous(noisy_pts2), F)
    avg_epi_error = np.mean(epi_error[inliers])
    print(f"Average Epipolar error: {avg_epi_error}")

    # Normalize the F matrix (per Piazza post @ 117)
    F = F / F[2, 2]

    # displayEpipolarF(im1, im2, F)

    # Simple Tests to verify your implementation:
    pts1_homogenous, pts2_homogenous = toHomogenous(noisy_pts1), toHomogenous(noisy_pts2)

    assert F.shape == (3, 3)
    assert F[2, 2] == 1
    assert np.linalg.matrix_rank(F) == 2

    # Simple Tests to verify your implementation:
    from scipy.spatial.transform import Rotation as sRot

    rotVec = sRot.random()
    mat = rodrigues(rotVec.as_rotvec())

    assert np.linalg.norm(rotVec.as_rotvec() - invRodrigues(mat)) < 1e-3
    assert np.linalg.norm(rotVec.as_matrix() - mat) < 1e-3

    # Visualization:
    np.random.seed(1)
    correspondence = np.load("data/some_corresp_noisy.npz")  # Loading noisy correspondences
    intrinsics = np.load("data/intrinsics.npz")  # Loading the intrinscis of the camera
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    pts1, pts2 = correspondence["pts1"], correspondence["pts2"]
    im1 = plt.imread("data/im1.png")
    im2 = plt.imread("data/im2.png")
    M = np.max([*im1.shape, *im2.shape])

    # TODO: YOUR CODE HERE
    """
    Call the ransacF function to find the fundamental matrix
    Call the findM2 function to find the extrinsics of the second camera
    Call the bundleAdjustment function to optimize the extrinsics and 3D points
    Plot the 3D points before and after bundle adjustment using the plot_3D_dual function
    """
