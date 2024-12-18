# ##################################################################### #
# 16820: Computer Vision Homework 5
# Carnegie Mellon University
# Nov, 2023
###################################################################### #

import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2xyz
from utils import plotSurface, integrateFrankot
from scipy.sparse import kron, csc_matrix
from scipy.sparse.linalg import lsqr


def renderNDotLSphere(center, rad, light, pxSize, res):
    """
    Question 1 (b)

    Render a hemispherical bowl with a given center and radius. Assume that
    the hollow end of the bowl faces in the positive z direction, and the
    camera looks towards the hollow end in the negative z direction. The
    camera's sensor axes are aligned with the x- and y-axes.

    Parameters
    ----------
    center : numpy.ndarray
        The center of the hemispherical bowl in an array of size (3,)

    rad : float
        The radius of the bowl

    light : numpy.ndarray
        The direction of incoming light

    pxSize : float
        Pixel size

    res : numpy.ndarray
        The resolution of the camera frame

    Returns
    -------
    image : numpy.ndarray
        The rendered image of the hemispherical bowl
    """
    # Meshgrid for camera frame
    [X, Y] = np.meshgrid(np.arange(res[0]), np.arange(res[1]))
    X = (X - res[0] / 2) * pxSize * 1.0e-4
    Y = (Y - res[1] / 2) * pxSize * 1.0e-4

    # Calculate Z using the equation of a sphere
    Z = np.sqrt(rad**2 + 0j - X**2 - Y**2)
    X[np.real(Z) == 0] = 0
    Y[np.real(Z) == 0] = 0
    Z = np.real(Z)

    # Initialize image array
    image = np.zeros((res[1], res[0]))

    # Normalize the light vector
    light = light / np.linalg.norm(light)

    # Loop over each pixel
    for i in range(res[1]):
        for j in range(res[0]):
            # Omit the pixels that are outside the sphere
            if Z[i, j] > 0:
                # Calculate the normal at the point
                normal = np.array([X[i, j], Y[i, j], Z[i, j]])
                normal = normal / np.linalg.norm(normal) 
                
                # Calculate the intensity of the pixel
                intensity = np.dot(normal, light)
                image[i, j] = max(0, intensity)
    
    return image


def loadData(path="../data/"):
    """
    Question 1 (c)

    Load data from the path given. The images are stored as input_n.tif
    for n = {1...7}. The source lighting directions are stored in
    sources.mat.

    Parameters
    ---------
    path: str
        Path of the data directory

    Returns
    -------
    I : numpy.ndarray
        The 7 x P matrix of vectorized images

    L : numpy.ndarray
        The 3 x 7 matrix of lighting directions

    s: tuple
        Image shape

    """

    # Initialize variables
    s = None
    luminance_channel = []
    
    # Load the lighting source from sources.npy (3 x 7)
    L = np.load(path + "sources.npy").T

    # Loop over each image
    for i in range(1, 8):
        # Load the image
        image = plt.imread(path + "input_" + str(i) + ".tif")
        image = image.astype(np.uint16)  # 16-bit image

        # Convert the image to XYZ color space to get the luminance
        image = rgb2xyz(image)
        luminance = image[:, :, 1]

        # Vectorize the luminance values and store them in I
        luminance_channel.append(luminance.flatten())

        # Get the shape of the image
        if s is None:
            s = luminance.shape
    
    # Convert the list to a numpy array (7 x P)
    I = np.vstack(luminance_channel)

    # Print the shapes 
    # print('Matrix I:', I.shape)
    # print('Matrix L:', L.shape)
    # print('Image shape:', s)

    return I, L, s


def estimatePseudonormalsCalibrated(I, L):
    """
    Question 1 (e)

    In calibrated photometric stereo, estimate pseudonormals from the
    light direction and image matrices

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P array of vectorized images

    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals
    """
    
    # Calculate the pseudonormals using the least squares method
    B = np.linalg.lstsq(L.T, I, rcond=None)[0]
    
    return B


def estimateAlbedosNormals(B):
    """
    Question 1 (f)

    From the estimated pseudonormals, estimate the albedos and normals

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of estimated pseudonormals

    Returns
    -------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals
    """

    # Calculate the albedos
    albedos = np.linalg.norm(B, axis=0)

    # Normalize the pseudonormals
    normals = B / albedos

    return albedos, normals


def displayAlbedosNormals(albedos, normals, s):
    """
    Question 1 (f)

    From the estimated pseudonormals, display the albedo and normal maps

    Please make sure to use the `coolwarm` colormap for the albedo image
    and the `rainbow` colormap for the normals.

    Parameters
    ----------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    -------
    albedoIm : numpy.ndarray
        Albedo image of shape s

    normalIm : numpy.ndarray
        Normals reshaped as an s x 3 image

    """

    # Reshape the albedos and normals
    albedoIm = albedos.reshape(s)
    normalIm = normals.T.reshape(s[0], s[1], 3)

    # Normalize the albedos and normals
    albedoIm = (albedoIm - np.min(albedoIm)) / (np.max(albedoIm) - np.min(albedoIm))
    normalIm = (normalIm - np.min(normalIm)) / (np.max(normalIm) - np.min(normalIm))

    return albedoIm, normalIm


def estimateShape(normals, s):
    """
    Question 1 (i)

    Integrate the estimated normals to get an estimate of the depth map
    of the surface.

    Parameters
    ----------
    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    ----------
    surface: numpy.ndarray
        The image, of size s, of estimated depths at each point

    """

    # Initialize the surface
    surface = np.zeros(s)

    # Calculate the gradients of the normals
    zx = np.reshape(normals[0, :] / -normals[2, :], s)
    zy = np.reshape(normals[1, :] / -normals[2, :], s)

    # Apply the Frankot-Chellappa algorithm
    surface = integrateFrankot(zx, zy)

    return surface


if __name__ == "__main__":
    # Part 1(b)
    radius = 0.75  # cm
    center = np.asarray([0, 0, 0])  # cm
    pxSize = 7  # um
    res = (3840, 2160)

    light = np.asarray([1, 1, 1]) / np.sqrt(3)
    image = renderNDotLSphere(center, radius, light, pxSize, res)
    plt.figure()
    plt.imshow(image, cmap="gray")
    plt.imsave("../results/1b-a.png", image, cmap="gray")

    light = np.asarray([1, -1, 1]) / np.sqrt(3)
    image = renderNDotLSphere(center, radius, light, pxSize, res)
    plt.figure()
    plt.imshow(image, cmap="gray")
    plt.imsave("../results/1b-b.png", image, cmap="gray")

    light = np.asarray([-1, -1, 1]) / np.sqrt(3)
    image = renderNDotLSphere(center, radius, light, pxSize, res)
    plt.figure()
    plt.imshow(image, cmap="gray")
    plt.imsave("../results/1b-c.png", image, cmap="gray")

    # # Part 1(c)
    I, L, s = loadData("../data/")

    # Part 1(d)
    U, S, Vt = np.linalg.svd(I, full_matrices=False)
    print('Singular Values of matrix I:', S)

    # Part 1(e)
    B = estimatePseudonormalsCalibrated(I, L)

    # Part 1(f)
    albedos, normals = estimateAlbedosNormals(B)
    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)
    plt.imsave("../results/1f-a.png", albedoIm, cmap="gray")
    plt.imsave("../results/1f-b.png", normalIm, cmap="rainbow")

    # Part 1(i)
    surface = estimateShape(normals, s)
    plotSurface(surface)
