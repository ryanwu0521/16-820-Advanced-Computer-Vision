import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation
from scipy.ndimage import affine_transform
from LucasKanadeAffine import LucasKanadeAffine
from InverseCompositionAffine import InverseCompositionAffine

def SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance):
    """
    :param image1: Images at time t
    :param image2: Images at time t+1
    :param threshold: used for LucasKanadeAffine
    :param num_iters: used for LucasKanadeAffine
    :param tolerance: binary threshold of intensity difference when computing the mask
    :return: mask: [nxm]
    """
    ################### TODO Implement Substract Dominent Motion ###################
    
    # compute the affine warp matrix
    M = LucasKanadeAffine(image1, image2, threshold, num_iters)

    # warp the image using the affine warp matrix
    image1_warped = affine_transform(image1, M[:2, :2], offset=M[:2, 2])

    # compute the mask with binary threshold of intensity difference
    mask = np.abs(image1_warped - image2) > tolerance

    # apply morphological operations
    mask = binary_erosion(mask, iterations=1)
    mask = binary_dilation(mask, iterations=5)

    return mask.astype(bool)
