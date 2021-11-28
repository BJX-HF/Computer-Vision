import math
import numpy as np
import cv2

def cross_correlation_2d(img,kernel):
    if (np.array(img).ndim == 3):
        temp = np.ones(((3,img.shape[0],img.shape[1])))
        res = img.copy()
        for p in range(3):
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    temp[p][i][j] = img[i][j][p]
            temp[p]=cross_correlation_2d(temp[p],kernel)
        for p in range(3):
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    res[i][j][p] = temp[p][i][j]
    else:
        img_extend0 = img.copy()
        res = img.copy()
        kernel_h = len(kernel)
        kernel_w = len(kernel[0])
        raw = np.zeros(img.shape[1])
        for i in range(0,kernel_h//2):
            img_extend0 = np.insert(img_extend0,0,values=raw,axis=0)
            img_extend0 = np.insert(img_extend0,img_extend0.shape[0],values=raw,axis=0)
        col = np.zeros(img_extend0.shape[0])
        for i in range(0,kernel_w//2):
            img_extend0 = np.insert(img_extend0,0,values=col,axis=1)
            img_extend0 = np.insert(img_extend0,img_extend0.shape[1],values=col,axis=1)
        H = img.shape[0]
        W = img.shape[1]
        for i in range(0,H):
            for j in range(0,W):
                temp = img_extend0[i:i+kernel_h,j:j+kernel_w]
                temp = np.multiply(temp,kernel)
                res[i][j] = temp.sum()
    return res

def convolve_2d(img,kernel):
    kernel_h = len(kernel)
    kernel_w = len(kernel[0])
    kernel_sys = kernel.copy()
    for i in range(kernel_h):
        for j in range(kernel_w):
            kernel_sys[i][j] = kernel[kernel_h-1-i][kernel_w-1-j]
    return cross_correlation_2d(img,kernel_sys)

def gaussian_blur_kernel_2d(sigma, height, width):
    core_x = width//2
    core_y = height//2
    kernel = np.zeros((height,width))
    for i in range(height):
        for j in range(width):
            x = abs(j-core_x)
            y = abs(i-core_y)
            kernel[i][j] = math.exp(-(x**2+y**2)/(2*sigma*sigma))/(2*math.pi*sigma*sigma)
    kernel_sum = kernel.sum()
    for i in range(height):
        for j in range(width):
            kernel[i][j] = kernel[i][j]/kernel_sum
    return kernel
def low_pass(img, sigma, size):
    kernel_low_pass = gaussian_blur_kernel_2d(sigma,size,size)
    return convolve_2d(img,kernel_low_pass)

def high_pass(img, sigma, size):
    return img - low_pass(img,sigma,size)

