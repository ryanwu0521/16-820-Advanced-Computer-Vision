# ##################################################################### #
# 16820: Computer Vision Homework 5
# Carnegie Mellon University
# 
# Nov, 2023
# ##################################################################### #

import numpy as np
import matplotlib.pyplot as plt
from q1 import (
    loadData,
    estimateAlbedosNormals,
    displayAlbedosNormals,
    estimateShape,
)
from q1 import estimateShape
from utils import enforceIntegrability, plotSurface

def estimatePseudonormalsUncalibrated(I):
    """
    Question 2 (b)

    Estimate pseudonormals without the help of light source directions.

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P matrix of loaded images

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pseudonormals

    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    """
    
    # Perfrom SVD on I
    U, S, Vt = np.linalg.svd(I, full_matrices=False)

    # Truncate the top 3 singular values
    S3 = np.diag(S[:3])     # first 3 singular values
    U3 = U[:, :3]           # first 3 columns of U
    Vt3 = Vt[:3, :]         # first 3 rows of Vt

    # Calculate lighting directions (L) and pseudonormals (B)
    L = U3 @ np.sqrt(S3)
    B = np.sqrt(S3) @ Vt3  # B: 3 x P
   
    return B, L


def plotBasRelief(B, mu, nu, lam):
    """
    Question 2 (f)

    Make a 3D plot of of a bas-relief transformation with the given parameters.

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of pseudonormals

    mu : float
        bas-relief parameter

    nu : float
        bas-relief parameter

    lambda : float
        bas-relief parameter

    Returns
    -------
        None

    """

    # Your code here

if __name__ == "__main__":
    # Part 2 (b)
    # Load the image data
    I, L0, s = loadData("../data/")

    # Estimate the albedos and normals
    B, L = estimatePseudonormalsUncalibrated(I)
    albedos, normals = estimateAlbedosNormals(B)
    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)

    # Save the images
    plt.imsave("../results/2b-a.png", albedoIm, cmap="gray")
    plt.imsave("../results/2b-b.png", normalIm, cmap="rainbow")

    # Part 2 (c)
    print('Ground truth lighting directions:\n', L0)
    print('Estimated lighting directions:\n', L)

    # Part 2 (d)
    surface = estimateShape(normals, s)
    plotSurface(surface)

    # Part 2 (e)
    # Your code here

    # Part 2 (f)
    # Your code here
