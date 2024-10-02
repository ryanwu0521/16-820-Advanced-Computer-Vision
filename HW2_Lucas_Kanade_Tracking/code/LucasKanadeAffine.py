import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import affine_transform
import cv2


def affine_warp(x, y, p):
    # warp the coordinates using the affine warp matrix
    x_warped = (1 + p[0]) * x + p[1] * y + p[2]
    y_warped = p[3] * x + (1 + p[4]) * y + p[5]
    return x_warped, y_warped
    

def LucasKanadeAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array] put your implementation here
    """

    
    ################### TODO Implement Lucas Kanade Affine ###################
    
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


    for _ in range(int(num_iters)):

        # interpolate the current template image
        It_template = It_interp.ev(y, x).reshape(H, W)

        # affine warp
        x_warped, y_warped = affine_warp(x.flatten(), y.flatten(), p) 

        # interpolate the warped image
        It1_warped = It1_interp.ev(y_warped, x_warped).reshape(H, W)

        # compute the error image
        b = (It_template - It1_warped).flatten()

        # compute the gradient of warped image
        x_grad = cv2.Sobel(It1_warped, cv2.CV_64F, 1, 0, ksize=3)
        y_grad = cv2.Sobel(It1_warped, cv2.CV_64F, 0, 1, ksize=3)

        # compute the Jacobian
        A = np.zeros((H * W, 6))
        A[:, 0] = x.flatten() * x_grad.flatten()
        A[:, 1] = x.flatten() * y_grad.flatten()
        A[:, 2] = x_grad.flatten()
        A[:, 3] = y.flatten() * x_grad.flatten()
        A[:, 4] = y.flatten() * y_grad.flatten()
        A[:, 5] = y_grad.flatten()

        # compute the delta p
        dp = np.linalg.lstsq(A, b, rcond=None)[0]

        # update the movement vector
        p += dp

        # termination condition
        if np.linalg.norm(dp) < threshold:
            break

    # update the affine warp matrix
    M = np.array([[1 + p[0], p[1], p[2]], 
                  [p[3], 1 + p[4], p[5]]])
    
    # debug print
    # print(f"M: {M}")
    
    return M
