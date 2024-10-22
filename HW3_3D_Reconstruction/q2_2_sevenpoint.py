import numpy as np
import matplotlib.pyplot as plt

from helper import displayEpipolarF, calc_epi_error, toHomogenous, _singularize, refineF

# Insert your package here
import cv2

"""
Q2.2: Seven Point Algorithm for calculating the fundamental matrix
    Input:  pts1, 7x2 Matrix containing the corresponding points from image1
            pts2, 7x2 Matrix containing the corresponding points from image2
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated 3x3 fundamental matrixes.
    
    HINTS:
    (1) Normalize the input pts1 and pts2 scale paramter M.
    (2) Setup the seven point algorithm's equation.
    (3) Solve for the least square solution using SVD. 
    (4) Pick the last two colum vector of vT.T (the two null space solution f1 and f2)
    (5) Use the singularity constraint to solve for the cubic polynomial equation of  F = a*f1 + (1-a)*f2 that leads to 
        det(F) = 0. Solving this polynomial will give you one or three real solutions of the fundamental matrix. 
        Use np.polynomial.polynomial.polyroots to solve for the roots
    (6) Unscale the fundamental matrixes and return as Farray
"""


def sevenpoint(pts1, pts2, M):
    Farray = []
    # Normalize the input points
    T = np.array([[1 / M, 0, 0], [0, 1 / M, 0], [0, 0, 1]])
    pts1_norm = pts1 / M
    pts2_norm = pts2 / M

    # Constructing the A matrix
    A = np.zeros((7, 9))
    for i in range(7):
        x1, y1 = pts1_norm[i]
        x2, y2 = pts2_norm[i]
        # A[i] = np.array([x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1])
        A[i] = np.array([x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1])

    # Solve for least square solution using SVD
    _, _, V = np.linalg.svd(A)
    
    # Pick the last two colum vector of vT.T (the two null space solution f1 and f2)
    f1 = V[-1].reshape(3, 3)
    f2 = V[-2].reshape(3, 3)

    # Coefficients of the polynomial equation
    # Set up the polynomial equation 
    def cubic_poly(a):
        F = a * f1 + (1 - a) * f2
        return np.linalg.det(F) 

    # Coefficients of the polynomial equation
    coeff = [cubic_poly(0), cubic_poly(1/3), cubic_poly(2/3), cubic_poly(1)]

    # Solve the polynomial equation (np.polynomial.polynomial.polyroots)
    roots = np.polynomial.polynomial.polyroots(coeff).real
    print("Roots of the polynomial:\n", roots)

    # Unscale the fundamental matrixes
    for root in roots:
        F = root * f1 + (1 - root) * f2
        F = _singularize(F)
        F = np.dot(np.dot(T.T, F), T)  # Unnormlize the fundamental matrix
        Farray.append(F)
    
    return Farray


if __name__ == "__main__":
    correspondence = np.load("data/some_corresp.npz")  # Loading correspondences
    intrinsics = np.load("data/intrinsics.npz")  # Loading the intrinscis of the camera
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    pts1, pts2 = correspondence["pts1"], correspondence["pts2"]
    im1 = plt.imread("data/im1.png")
    im2 = plt.imread("data/im2.png")

    # indices = np.arange(pts1.shape[0])
    # indices = np.random.choice(indices, 7, False)
    indices = np.array([82, 19, 56, 84, 54, 24, 18])

    M = np.max([*im1.shape, *im2.shape])

    Farray = sevenpoint(pts1[indices, :], pts2[indices, :], M)
    
    # for i, F in enumerate(Farray):
    #     print(f"Recovered Fundamental Matrix {i+1}:\n{F}\n")
    
    F = Farray[0]

    # Print the recovered fundamental matrix F
    print("Recovered Fundamental Matrix:\n" + str(F))

    # save the recovered F
    np.savez("q2_2.npz", F, M)

    # fundamental matrix must have rank 2!
    assert(np.linalg.matrix_rank(F) == 2)
    # displayEpipolarF(im1, im2, F)

    # Simple Tests to verify your implementation:
    # Test out the seven-point algorithm by randomly sampling 7 points and finding the best solution.
    np.random.seed(1)  # Added for testing, can be commented out

    pts1_homogenous, pts2_homogenous = toHomogenous(pts1), toHomogenous(pts2)

    max_iter = 500
    pts1_homo = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    pts2_homo = np.hstack((pts2, np.ones((pts2.shape[0], 1))))

    ress = []
    F_res = []
    choices = []
    M = np.max([*im1.shape, *im2.shape])
    for i in range(max_iter):
        choice = np.random.choice(range(pts1.shape[0]), 7)
        pts1_choice = pts1[choice, :]
        pts2_choice = pts2[choice, :]
        Fs = sevenpoint(pts1_choice, pts2_choice, M)
        for F in Fs:
            choices.append(choice)
            res = calc_epi_error(pts1_homo, pts2_homo, F)
            F_res.append(F)
            ress.append(np.mean(res))

    min_idx = np.argmin(np.abs(np.array(ress)))
    F = F_res[min_idx]
    print("Best Fundamental Matrix:\n", F)
    print("Error:", ress[min_idx])

    displayEpipolarF(im1, im2, F)

    # save the best F
    # np.savez("q2_2.npz", F, M)

    assert F.shape == (3, 3)
    assert F[2, 2] == 1
    assert np.linalg.matrix_rank(F) == 2
    assert np.mean(calc_epi_error(pts1_homogenous, pts2_homogenous, F)) < 1