import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import affine_transform
import cv2
from LucasKanadeAffine import affine_warp

def InverseCompositionAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array]
    """


    ################### TODO Implement Inverse Composition Affine ###################
    
    # intialize p & M
    p = np.zeros(6)
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    # image size
    H, W = It.shape

    # meshgrid for the template image
    x, y = np.meshgrid(np.arange(W), np.arange(H))

    # interpolate the template and current image
    It_interp = RectBivariateSpline(np.arange(H), np.arange(W), It)
    It1_interp = RectBivariateSpline(np.arange(H), np.arange(W), It1)

    # precompute the template image gradient
    It_template = It_interp.ev(y, x).reshape(H, W)
    x_grad = cv2.Sobel(It_template, cv2.CV_64F, 1, 0, ksize=3)
    y_grad = cv2.Sobel(It_template, cv2.CV_64F, 0, 1, ksize=3)

    # precomputer the Jacobian matrix A
    A = np.zeros((H * W, 6))
    A[:, 0] = x.flatten() * x_grad.flatten()
    A[:, 1] = x.flatten() * y_grad.flatten()
    A[:, 2] = x_grad.flatten()
    A[:, 3] = y.flatten() * x_grad.flatten()
    A[:, 4] = y.flatten() * y_grad.flatten()
    A[:, 5] = y_grad.flatten()

    for _ in range(int(num_iters)):

        # affine warp
        x_warped, y_warped = affine_warp(x.flatten(), y.flatten(), p) 

        # interpolate the warped image
        It1_warped = It1_interp.ev(y_warped, x_warped).reshape(H, W)

        # compute the error image
        b = (It_template - It1_warped).flatten()

        # compute the delta p (steepest descent)
        dp = np.linalg.lstsq(A, b, rcond=None)[0]

        # termination condition
        if np.linalg.norm(dp) < threshold:
            break

        # update the movement vector (inverse composition)
        deltaM = np.array([[1 + dp[0], dp[1], dp[2]], 
                           [dp[3], 1 + dp[4], dp[5]],
                           [0, 0, 1]])
        
        deltaM[:2, :2] = deltaM 
        
        # inverse of the deltaM
        deltaM_inv = np.linalg.inv(deltaM)

        # update the affine warp matrix
        M = np.dot(deltaM_inv[:2, :], M)

    return M
