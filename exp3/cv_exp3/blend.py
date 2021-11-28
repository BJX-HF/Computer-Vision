# 复杂
import math
import sys

import cv2
import numpy as np
from scipy import ndimage


class ImageInfo:
    def __init__(self, name, img, position):
        self.name = name
        self.img = img
        self.position = position


def imageBoundingBox(img, M):
    """
       This is a useful helper function that you might choose to implement
       that takes an image, and a transform, and computes the bounding box
       of the transformed image.

       INPUT:
         img: image to get the bounding box of
         M: the transformation to apply to the img
       OUTPUT:
         minX: int for the minimum X value of a corner
         minY: int for the minimum Y value of a corner
         maxX: int for the maximum X value of a corner
         maxY: int for the maximum Y value of a corner
    """
    # TODO 8
    # TODO-BLOCK-BEGIN
    corner = []
    corner.append(np.array([0, 0, 1]))
    corner.append(np.array([img.shape[1] - 1, 0, 1]))
    corner.append(np.array([img.shape[1] - 1, img.shape[0] - 1, 1]))
    corner.append(np.array([0, img.shape[0] - 1, 1]))

    for i in range(4):
        corner[i] = np.dot(M, corner[i].T)
        corner[i] /= corner[i][2]
    minX = min(corner[0][0], corner[1][0], corner[2][0], corner[3][0])
    maxX = max(corner[0][0], corner[1][0], corner[2][0], corner[3][0])
    minY = min(corner[0][1], corner[1][1], corner[2][1], corner[3][1])
    maxY = max(corner[0][1], corner[1][1], corner[2][1], corner[3][1])

    # TODO-BLOCK-END
    return int(minX), int(minY), int(maxX), int(maxY)


def accumulateBlend(img, acc, M, blendWidth):
    """
       INPUT:
         img: image to add to the accumulator
         acc: portion of the accumulated image where img should be added
         M: the transformation mapping the input image to the accumulator
         blendWidth: width of blending function. horizontal hat function
       OUTPUT:
         modify acc with weighted copy of img added where the first
         three channels of acc record the weighted sum of the pixel colors
         and the fourth channel of acc records a sum of the weights
    """
    # BEGIN TODO 10
    # Fill in this routine
    # TODO-BLOCK-BEGIN
    M /= M[2, 2]
    img1 = np.zeros((acc.shape[0], acc.shape[1], 4))
    for path in range(3):
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                temp = np.array([x, y, 1])
                vector = np.dot(M, temp)
                vector /= vector[2]
                vector = vector.astype(int)
                img1[vector[1], vector[0], path] = img[y, x, path].astype(np.uint8)
    if (np.where(acc != 0)[0].shape[0] == 0): return img1
    weight = np.zeros((img1.shape[0], img1.shape[1]))
    # 计算权重矩阵
    for h in range(img1.shape[0]):
        line = img1[h, :, 0] > 0
        line = line.astype(float)
        k = 1
        while (line[k] == 0 and k < img1.shape[1] - blendWidth - 1):
            k += 1
        if k + blendWidth < img1.shape[1] - 1:
            interm = list(np.arange(0, 1, 1 / (blendWidth)))  # 从0渐变到1的列表
            line[k - 1:k - 1 + len(interm)] = interm
            weight[h] = line
        h += 1

    res = np.zeros(img1.shape)
    for channel in range(3):
        res[:, :, channel] = acc[:, :, channel] * (1 - weight) + img1[:, :, channel] * weight
    return res.astype(np.uint8)
    # TODO-BLOCK-END
    # END TODO


def normalizeBlend(acc):
    """
       INPUT:
         acc: input image whose alpha channel (4th channel) contains
         normalizing weight values
       OUTPUT:
         img: image with r,g,b values of acc normalized
    """
    # BEGIN TODO 11
    # fill in this routine..
    # TODO-BLOCK-BEGIN
    # img = np.delete(acc,3,axis=2)
    # TODO-BLOCK-END
    # END TODO
    # todo10中已经将权重之和当作1计算，此处只需删除权重通道
    acc = np.delete(acc, 3, axis=2)
    # #解决HOMO的裂缝问题
    # for channel in range(3):
    #     for h in range(0, acc.shape[0]):
    #         line = acc[h,:,channel]
    #         line[-1] = 1
    #         k = 0
    #         while(k<acc.shape[1]-1):
    #             if(line[k]==0):
    #                 left = k
    #                 while(line[k]==0): k += 1
    #                 right = k
    #                 if(left>0): line[left:right] = (line[left-1]+line[right])/2
    #             k += 1
    return acc


def getAccSize(ipv):
    """
       This function takes a list of ImageInfo objects consisting of images and
       corresponding transforms and Returns useful information about the accumulated
       image.

       INPUT:
         ipv: list of ImageInfo objects consisting of image (ImageInfo.img) and transform(image (ImageInfo.position))
       OUTPUT:
         accWidth: Width of accumulator image(minimum width such that all tranformed images lie within acc)
         accHeight: Height of accumulator image(minimum height such that all tranformed images lie within acc)

         channels: Number of channels in the accumulator image
         width: Width of each image(assumption: all input images have same width)
         translation: transformation matrix so that top-left corner of accumulator image is origin
    """

    # Compute bounding box for the mosaic
    minX = sys.maxsize
    minY = sys.maxsize
    maxX = 0
    maxY = 0
    channels = -1
    width = -1  # Assumes all images are the same width
    M = np.identity(3)
    for i in ipv:
        M = i.position
        img = i.img
        _, w, c = img.shape
        if channels == -1:
            channels = c
            width = w

        # BEGIN TODO 9
        # add some code here to update minX, ..., maxY
        # TODO-BLOCK-BEGIN
        m = M[2, 2]
        M = M / m
        x1, y1, x2, y2 = imageBoundingBox(img, M)
        minX = min(x1, minX)
        minY = min(y1, minY)
        maxX = max(x2, maxX) + 1
        maxY = max(y2, maxY) + 1
        # TODO-BLOCK-END
        # END TODO

    # Create an accumulator image
    accWidth = int(math.ceil(maxX) - math.floor(minX))
    accHeight = int(math.ceil(maxY) - math.floor(minY))
    print('accWidth, accHeight:', (accWidth, accHeight))
    translation = np.array([[1, 0, -minX], [0, 1, -minY], [0, 0, 1]])

    return accWidth, accHeight, channels, width, translation


def pasteImages(ipv, translation, blendWidth, accWidth, accHeight, channels):
    acc = np.zeros((accHeight, accWidth, channels + 1))
    # Add in all the images
    M = np.identity(3)
    for count, i in enumerate(ipv):
        M = i.position
        img = i.img

        M_trans = translation.dot(M)
        acc = accumulateBlend(img, acc, M_trans, blendWidth)

    return acc


def getDriftParams(ipv, translation, width):
    # Add in all the images
    M = np.identity(3)
    for count, i in enumerate(ipv):
        if count != 0 and count != (len(ipv) - 1):
            continue

        M = i.position

        M_trans = translation.dot(M)

        p = np.array([0.5 * width, 0, 1])
        p = M_trans.dot(p)

        # First image
        if count == 0:
            x_init, y_init = p[:2] / p[2]
        # Last image
        if count == (len(ipv) - 1):
            x_final, y_final = p[:2] / p[2]

    return x_init, y_init, x_final, y_final


def computeDrift(x_init, y_init, x_final, y_final, width):
    A = np.identity(3)
    drift = (float)(y_final - y_init)
    # We implicitly multiply by -1 if the order of the images is swapped...
    length = (float)(x_final - x_init)
    A[0, 2] = -0.5 * width
    # Negative because positive y points downwards
    A[1, 0] = -drift / length

    return A


def blendImages(ipv, blendWidth, is360=False, A_out=None):
    """
       INPUT:
         ipv: list of input images and their relative positions in the mosaic
         blendWidth: width of the blending function
       OUTPUT:
         croppedImage: final mosaic created by blending all images and
         correcting for any vertical drift
    """
    accWidth, accHeight, channels, width, translation = getAccSize(ipv)
    acc = pasteImages(
        ipv, translation, blendWidth, accWidth, accHeight, channels
    )
    compImage = normalizeBlend(acc)
    cv2.imshow("1", compImage)
    cv2.waitKey(0)

    # Determine the final image width
    outputWidth = (accWidth - width) if is360 else accWidth
    x_init, y_init, x_final, y_final = getDriftParams(ipv, translation, width)
    # Compute the affine transform
    A = np.identity(3)
    # BEGIN TODO 12
    # fill in appropriate entries in A to trim the left edge and
    # to take out the vertical drift if this is a 360 panorama
    # (i.e. is360 is true)
    # Shift it left by the correct amount
    # Then handle the vertical drift
    # Note: warpPerspective does forward mapping which means A is an affine
    # transform that maps accumulator coordinates to final panorama coordinates
    # TODO-BLOCK-BEGIN
    # 非360度全景图下修正垂直漂移
    # if(is360==False):
    #     for count, i in enumerate(ipv):
    #         if count != 0 and count != (len(ipv) - 1):
    #             continue
    #         M = i.position
    #         M_trans = translation.dot(M)
    #
    #         # First image
    #         if count == 0:
    #             p = np.array([0, 0, 1])
    #             p = M_trans.dot(p)
    #             x_init, y_init = p[:2] / p[2]
    #         # Last image
    #         if count == (len(ipv) - 1):
    #             p = np.array([width, 0, 1])
    #             p = M_trans.dot(p)
    #             x_final, y_final = p[:2] / p[2]
    if is360 == True: A = computeDrift(x_init, y_init, x_final, y_final, width)  # 计算修正矩阵
    # if(is360==False): A[0,2] = 0
    # TODO-BLOCK-END
    # END TODO

    if A_out is not None:
        A_out[:] = A

    # Warp and crop the composite
    croppedImage = cv2.warpPerspective(
        compImage, A, (outputWidth, accHeight), flags=cv2.INTER_LINEAR
    )

    return croppedImage
