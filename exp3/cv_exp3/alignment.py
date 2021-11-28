import math
import random

import cv2
import numpy as np
from scipy.spatial.distance import cdist

eTranslate = 0
eHomography = 1


def computeHomography(f1, f2, matches, A_out=None):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        A_out -- ignore this parameter. If computeHomography is needed
                 in other TODOs, call computeHomography(f1,f2,matches)
    Output:
        H -- 2D homography (3x3 matrix)
        Takes two lists of features, f1 and f2, and a list of feature
        matches, and estimates a homography from image 1 to image 2 from the matches.
    '''
    num_matches = len(matches)

    # Dimensions of the A matrix in the homogenous linear
    # equation Ah = 0
    num_rows = 2 * num_matches
    num_cols = 9
    A_matrix_shape = (num_rows, num_cols)
    A = np.zeros(A_matrix_shape)

    for i in range(len(matches)):
        m = matches[i]
        (a_x, a_y) = f1[m.queryIdx].pt
        (b_x, b_y) = f2[m.trainIdx].pt

        # BEGIN TODO 2
        # Fill in the matrix A in this loop.
        # Access elements using square brackets. e.g. A[0,0]
        # TODO-BLOCK-BEGIN
        row1 = np.array([a_x, a_y, 1, 0, 0, 0, -a_x * b_x, -a_y * b_x, -b_x])
        row2 = np.array([0, 0, 0, a_x, a_y, 1, -a_x * b_y, -a_y * b_y, -b_y])
        A[2 * i] = row1
        A[2 * i + 1] = row2
        # TODO-BLOCK-END
        # END TODO

    U, s, Vt = np.linalg.svd(A)

    if A_out is not None:
        A_out[:] = A

    # s is a 1-D array of singular values sorted in descending order
    # U, Vt are unitary matrices
    # Rows of Vt are the eigenvectors of A^TA.
    # Columns of U are the eigenvectors of AA^T.

    # Homography to be calculated
    H = np.eye(3)

    # BEGIN TODO 3
    # Fill the homography H with the appropriate elements of the SVD
    # TODO-BLOCK-BEGIN
    ATA = np.dot(A.T, A)
    featureValue, featureVector = np.linalg.eig(ATA)
    index = np.argmin(featureValue)
    H = featureVector[:, index]
    H = np.reshape(H, (3, 3))
    # TODO-BLOCK-END
    # END TODO

    return H


def alignPair(f1, f2, matches, m, nRANSAC, RANSACthresh):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        m -- MotionModel (eTranslate, eHomography)
        nRANSAC -- number of RANSAC iterations
        RANSACthresh -- RANSAC distance threshold

    Output:
        M -- inter-image transformation matrix
        Repeat for nRANSAC iterations:
            Choose a minimal set of feature matches.
            Estimate the transformation implied by these matches
            count the number of inliers.
        For the transformation with the maximum number of inliers,
        compute the least squares motion estimate using the inliers,
        and return as a transformation matrix M.
    '''

    # BEGIN TODO 4
    # Write this entire method.  You need to handle two types of
    # motion models, pure translations (m == eTranslation) and
    # full homographies (m == eHomography).  However, you should
    # only have one outer loop to perform the RANSAC code, as
    # the use of RANSAC is almost identical for both cases.

    # Your homography handling code should call compute_homography.
    # This function should also call get_inliers and, at the end,
    # least_squares_fit.
    # TODO-BLOCK-BEGIN
    max_inlier_indices = []
    matches_index = list(np.arange(len(matches)))  # matches下标列表，用于随机抽取匹配对

    for time in range(nRANSAC):
        M = np.eye(3)
        if (m == eTranslate):
            index = random.sample(matches_index, 1)[0]
            x1, y1 = f1[matches[index].queryIdx].pt
            x2, y2 = f2[matches[index].trainIdx].pt
            M[0, 2] = x2 - x1
            M[1, 2] = y2 - y1
        elif (m == eHomography):
            matchesHomo = []
            index = random.sample(matches_index, 4)
            for t in range(4):
                matchesHomo.append(matches[index[t]])
            M = computeHomography(f1, f2, matchesHomo)
        else:
            raise Exception("Error: Invalid motion model.")

        inlier_indices = getInliers(f1, f2, matches, M, RANSACthresh)
        if (len(inlier_indices) > len(max_inlier_indices)): max_inlier_indices = inlier_indices
    M = leastSquaresFit(f1, f2, matches, m, max_inlier_indices)
    # TODO-BLOCK-END
    # END TODO
    return M


def getInliers(f1, f2, matches, M, RANSACthresh):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        M -- inter-image transformation matrix
        RANSACthresh -- RANSAC distance threshold

    Output:
        inlier_indices -- inlier match indices (indexes into 'matches')

        Transform the matched features in f1 by M.
        Store the match index of features in f1 for which the transformed
        feature is within Euclidean distance RANSACthresh of its match
        in f2.
        Return the array of the match indices of these features.
    '''

    inlier_indices = []

    for i in range(len(matches)):
        # BEGIN TODO 5
        # Determine if the ith matched feature f1[id1], when transformed
        # by M, is within RANSACthresh of its match in f2.
        # If so, append i to inliers
        # TODO-BLOCK-BEGIN
        featurepoint1 = f1[matches[i].queryIdx]
        featurepoint2 = f2[matches[i].trainIdx]
        x, y = featurepoint1.pt
        col1 = np.array([x, y, 1])
        col1_trans = np.dot(M, col1)
        col1_trans /= col1_trans[2]
        x2, y2 = featurepoint2.pt
        col2 = np.array([x2, y2, 1])
        distance = np.linalg.norm(col2 - col1_trans)
        if (distance <= RANSACthresh): inlier_indices.append(i)
        # TODO-BLOCK-END
        # END TODO

    return inlier_indices


def leastSquaresFit(f1, f2, matches, m, inlier_indices):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        m -- MotionModel (eTranslate, eHomography)
        inlier_indices -- inlier match indices (indexes into 'matches')

    Output:
        M - transformation matrix

        Compute the transformation matrix from f1 to f2 using only the
        inliers and return it.
    '''

    # This function needs to handle two possible motion models,
    # pure translations (eTranslate)
    # and full homographies (eHomography).

    M = np.eye(3)

    if m == 0:
        # For spherically warped images, the transformation is a
        # translation and only has two degrees of freedom.
        # Therefore, we simply compute the average translation vector
        # between the feature in f1 and its match in f2 for all inliers.

        u = 0.0
        v = 0.0

        for i in range(len(inlier_indices)):
            # BEGIN TODO 6
            # Use this loop to compute the average translation vector
            # over all inliers.
            # TODO-BLOCK-BEGIN
            index1 = matches[i].queryIdx
            index2 = matches[i].trainIdx
            x1, y1 = f1[index1].pt
            x2, y2 = f2[index2].pt
            u += x2 - x1
            v += y2 - y1
            # TODO-BLOCK-END
            # END TODO

        u /= len(inlier_indices)
        v /= len(inlier_indices)

        M[0, 2] = u
        M[1, 2] = v

    elif m == eHomography:
        # BEGIN TODO 7
        # Compute a homography M using all inliers.
        # This should call computeHomography.
        # TODO-BLOCK-BEGIN
        matches_inliers = []
        for j in inlier_indices:
            matches_inliers.append(matches[j])
        M = computeHomography(f1, f2, matches_inliers)
        # TODO-BLOCK-END
        # END TODO

    else:
        raise Exception("Error: Invalid motion model.")

    return M
