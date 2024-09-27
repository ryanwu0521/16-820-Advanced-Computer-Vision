import numpy as np
from scipy.interpolate import RectBivariateSpline
import cv2

def LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2)):
    """
    :param It: template image (previous frame)
    :param It1: Current image (current frame)
    :param rect: Current position of the car (top left, bot right coordinates) [rectangle region for tracking]
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """
	
    # Put your implementation here
    # set up the threshold
    ################### TODO Implement Lucas Kanade ###################
    
    # rectange coordinates/region for tracking
    x1, y1, x2, y2 = map(int, rect)

    # meshgrid for the template image
    x, y = np.meshgrid(np.arange(x1, x2 + 1), np.arange(y1, y2 + 1))

    # initial movement vector
    p = p0.copy()

    # interpolate the template and current image
    It_interp = RectBivariateSpline(np.arange(It.shape[0]), np.arange(It.shape[1]), It)
    It1_interp = RectBivariateSpline(np.arange(It1.shape[0]), np.arange(It1.shape[1]), It1)

    # termination condition
    dp = np.array([1, 1])
    i = 0

    while np.linalg.norm(dp) >= threshold and i < num_iters:
        i += 1

        # interpolate the current template image
        It_template = It_interp.ev(y, x).flatten()

        # warp the current image using the movement vector
        It1_warp = It1_interp.ev(y + p[1], x + p[0]).flatten()
        
        # compute the Jacobian
        A = np.array([It1_interp.ev(y + p[1], x + p[0], dy=1).flatten(), It1_interp.ev(y + p[1], x + p[0], dx=1).flatten()]).T

        # compute the error image
        b = (It_template - It1_warp).flatten()

        # compute the delta p
        dp = np.linalg.lstsq(A, b, rcond=None)[0]

        # update the movement vector
        p += dp

    return p
