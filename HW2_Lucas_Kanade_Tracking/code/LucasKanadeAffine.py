import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import affine_transform
import cv2

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
  



    return M
