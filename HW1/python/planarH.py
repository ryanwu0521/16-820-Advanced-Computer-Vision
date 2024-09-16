import numpy as np
import cv2


def computeH(x1, x2):
    #Q2.2.1
    # TODO: Compute the homography between two sets of points

    # Number of points
    N = x1.shape[0]

    # Initialize A matrix
    A = np.zeros((2*N, 9))

    # Construct A matrix
    for i in range(N):
        x, y = x1[i]
        X, Y = x2[i]
        A[2*i] = [-X, -Y, -1, 0, 0, 0, x*X, x*Y, x]
        A[2*i+1] = [0, 0, 0, -X, -Y, -1, y*X, y*Y, y]

    # Solve using SVD
    U, S, V = np.linalg.svd(A)

    # Homography
    H2to1 = V[-1].reshape(3, 3)
    
    return H2to1


def computeH_norm(x1, x2):
    #Q2.2.2
    # TODO: Compute the centroid of the points

    # Compute centroid of the points
    x1_centroid = np.mean(x1, axis=0)
    x2_centroid = np.mean(x2, axis=0)

    # TODO: Shift the origin of the points to the centroid
    x1_shifted = x1 - x1_centroid
    x2_shifted = x2 - x2_centroid

    # TODO: Normalize the points so that the largest distance from the origin is equal to sqrt(2)
    x1_scaleMax = np.sqrt(2) / np.max(np.linalg.norm(x1_shifted, axis=1))
    x2_scaleMax = np.sqrt(2) / np.max(np.linalg.norm(x2_shifted, axis=1))


    # TODO: Similarity transform 1
    T1 = np.array([[x1_scaleMax, 0, -x1_scaleMax*x1_centroid[0]], [0, x1_scaleMax, -x1_scaleMax*x1_centroid[1]], [0, 0, 1]])
    
    # TODO: Similarity transform 2
    T2 = np.array([[x2_scaleMax, 0, -x2_scaleMax*x2_centroid[0]], [0, x2_scaleMax, -x2_scaleMax*x2_centroid[1]], [0, 0, 1]])

    # TODO: Compute homography
    # Normalize points
    x1_normalized = (T1 @ np.hstack((x1, np.ones((x1.shape[0], 1)))).T).T
    x2_normalized = (T2 @ np.hstack((x2, np.ones((x2.shape[0], 1)))).T).T

    # Compute homography between normalized points
    H2to1_normalized = computeH(x1_normalized[:, :2], x2_normalized[:, :2])

    # TODO: Denormalization
    H2to1 = np.linalg.inv(T1) @ H2to1_normalized @ T2
        

    return H2to1




def computeH_ransac(locs1, locs2, opts):
    #Q2.2.3
    #Compute the best fitting homography given a list of matching points
    max_iters = opts.max_iters  # the number of iterations to run RANSAC for
    inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier

    # Initialize variables
    N = locs1.shape[0]
    bestH2to1 = np.eye(3)
    inliers = np.zeros(N)

    # Initialize inlier count
    max_inliers = 0

    for i in range(max_iters):
        # sample 4 points
        idx = np.random.choice(N, 4, replace=False)
        x1_sample = locs1[idx]
        x2_sample = locs2[idx]

        # Compute homography
        H2to1 = computeH(x1_sample, x2_sample)

        # Transform points 
        locs1_homogeneous = np.hstack((locs1, np.ones((N, 1))))
        locs2_project = (H2to1 @ locs1_homogeneous.T).T
        locs2_project /= locs2_project[:, [2]] # normalization

        # Compute the error and find inliers
        d = np.linalg.norm(locs2_project[:, :2] - locs2, axis=1)
        inliers = d < inlier_tol
        num_inliers = np.sum(inliers)

        # Update best homography
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            bestH2to1 = H2to1
        
    # Compute inliers
    if np.sum(inliers) > 4:
        bestH2to1 = computeH_norm(locs1[inliers], locs2[inliers])

    return bestH2to1, inliers



def compositeH(H2to1, template, img):
    
    #Create a composite image after warping the template image on top
    #of the image using the homography

    #Note that the homography we compute is from the image to the template;
    #x_template = H2to1*x_photo
    #For warping the template to the image, we need to invert it.

    # invert homography
    H2to1_invert = np.linalg.inv(H2to1)

    # TODO: Create mask of same size as template
    mask = np.ones((template.shape[0], template.shape[1]), dtype=np.uint8)

    # TODO: Warp mask by appropriate homography
    warped_mask = cv2.warpPerspective(mask, H2to1_invert, (img.shape[1], img.shape[0]))

    # TODO: Warp template by appropriate homography
    warped_template = cv2.warpPerspective(template, H2to1_invert, (img.shape[1], img.shape[0]))

    # TODO: Use mask to combine the warped template and the image
    composite_img = img.copy()
    composite_img[warped_mask == 1] = warped_template[warped_mask == 1]
    
    return composite_img


