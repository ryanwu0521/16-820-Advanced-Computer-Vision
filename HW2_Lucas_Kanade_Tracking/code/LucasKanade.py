import numpy as np
from scipy.interpolate import RectBivariateSpline
import cv2

def warp_image(x ,y, p):
    # warp the image
    x_warped = x + p[0]
    y_warped = y + p[1]
    
    return x_warped, y_warped


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

    # iterate through the number of iterations
    for _ in range(int(num_iters)):

        # interpolate the current template image
        It_template = It_interp.ev(y, x).flatten()

        # warp the current image using the movement vector
        x_warped, y_warped = warp_image(x, y, p)

        # interpolate the warped image
        It1_warped = It1_interp.ev(y_warped, x_warped).flatten()

        # compute the error image
        b = (It_template - It1_warped).flatten()

        # compute the Jacobian
        A = np.array([It1_interp.ev(y_warped, x_warped, dy=1).flatten(), It1_interp.ev(y_warped, x_warped, dx=1).flatten()]).T

        # compute the delta p
        dp = np.linalg.lstsq(A, b, rcond=None)[0]

        # update the movement vector
        p += dp

        # terminate condition
        if np.linalg.norm(dp) < threshold:
            break

    return p