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

    ''' This implementation does not work for Q2.e for some reason.'''
    # # Perfrom SVD on I
    # U, S, Vt = np.linalg.svd(I, full_matrices=False)

    # # Truncate the top 3 singular values
    # S3 = np.diag(S[:3])     # first 3 singular values
    # U3 = U[:, :3]           # first 3 columns of U
    # Vt3 = Vt[:3, :]         # first 3 rows of Vt

    # # Calculate lighting directions (L) and pseudonormals (B)
    # L = U3 @ np.sqrt(S3)
    # B = np.sqrt(S3) @ Vt3  # B: 3 x P

    
    ''' This implementation works for Q2.e '''
    # Perfrom SVD on I
    U, S, Vt = np.linalg.svd(I, full_matrices=False)

    # Truncate to rank 3 and calculate B and L
    L = U[:, :3]            # first 3 columns of U
    B = Vt[:3, :]           # first 3 rows of Vt
   
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

    # Construct the bas-relief transformation matrix
    # G = np.array([[1, 0, mu], [0, 1, nu], [0, 0, lam]])     # Generalized Bas-Relief matrix
    G = np.array([[1, 0, 0], [0, 1, 0], [mu, nu, lam]])   # Generalized Bas-Relief matrix

    # Apply the bas-relief transformation
    # B = G @ B
    B = G.T @ B
    

    # Compute the albedos and normals
    albedos, normals = estimateAlbedosNormals(B)
   
    # Enforce integrability
    Nt = enforceIntegrability(normals, s)

    # Estimate the shape
    surface = estimateShape(Nt, s)

    # Normalize the surface for saving as an image
    normalized_surface = (surface - np.min(surface)) / (np.max(surface) - np.min(surface))

    # Plot the surface
    plotSurface(surface, suffix=f"_basrelief_mu{mu}_nu{nu}_lam{lam}")
    # plotSurface(normalized_surface, suffix=f"_basrelief_mu{mu}_nu{nu}_lam{lam}")

if __name__ == "__main__":
    # Part 2 (b)
    # Load the image data
    I, L0, s = loadData("../data/")

    # Estimate the albedos and normals
    B, L = estimatePseudonormalsUncalibrated(I)
    albedos, normals = estimateAlbedosNormals(B)
    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)

    # Save the images
    # plt.imsave("../results/2b-a.png", albedoIm, cmap="gray")
    # plt.imsave("../results/2b-b.png", normalIm, cmap="rainbow")

    # Part 2 (c)
    # print('Ground truth lighting directions:\n', L0)
    # print('Estimated lighting directions:\n', L)

    # Part 2 (d)
    # surface_unenforce = estimateShape(normals, s)
    # plotSurface(surface_unenforce, suffix="_unintegrated")

    # Part 2 (e)
    # Nt = enforceIntegrability(normals, s)
    # G = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])    # Generalized Bas-Relief matrix
    # Nt = G @ Nt
    # surface_enforce = estimateShape(Nt, s)
    # plotSurface(surface_enforce, suffix="_integrated")


# def plotBasRelief(B, mu, nu, lam):
    # Part 2 (f)

    # # Test varying mu
    # plotBasRelief(B, 1, 1, 1)
    # plotBasRelief(B, 5, 1, 1)
    # plotBasRelief(B, 10, 1, 1)

    # Test varying nu
    # plotBasRelief(B, 1, 1, 1)
    # plotBasRelief(B, 1, 5, 1)
    # plotBasRelief(B, 1, 10, 1)

    # # Test varying lambda
    # plotBasRelief(B, 0, 0, 1)
    # plotBasRelief(B, 0, 0, 5)
    # plotBasRelief(B, 0, 0, 10)

    

    # Test varying mu
    # plotBasRelief(B, -1, 0.5 ,1)
    # plotBasRelief(B, 0.5, 0.5 ,1)
    # plotBasRelief(B, 1, 0.5 ,1)

    # Test varying nu
    # plotBasRelief(B, 0.5, -1, 1)
    # plotBasRelief(B, 0.5, 0.5, 1)
    # plotBasRelief(B, 0.5, 1, 1)

    # Test varying lambda
    plotBasRelief(B, 0.5, 0.5, -1)
    plotBasRelief(B, 0.5, 0.5, 0.5)
    plotBasRelief(B, 0.5, 0.5, 10)