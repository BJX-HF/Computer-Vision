import math

import cv2
import numpy as np

def get_rot_mx( angle_z):
    '''
    Input:
        angle_x -- Rotation around the x axis in radians
        angle_y -- Rotation around the y axis in radians
        angle_z -- Rotation around the z axis in radians
    Output:
        A 4x4 numpy array representing 3D rotations. The order of the rotation
        axes from first to last is x, y, z, if you multiply with the resulting
        rotation matrix from left.
    '''
    # Note: For MOPS, you need to use angle_z only, since we are in 2D

    rot_z_mx = np.array([[math.cos(angle_z), -math.sin(angle_z), 0],
                         [math.sin(angle_z), math.cos(angle_z), 0],
                         [0, 0, 1],
                         ])

    return rot_z_mx


def get_trans_mx(trans_vec):
    '''
    Input:
        trans_vec -- Translation vector represented by an 1D numpy array with 3
        elements
    Output:
        A 4x4 numpy array representing 3D translation.
    '''
    assert trans_vec.ndim == 1
    assert trans_vec.shape[0] == 2

    trans_mx = np.eye(3)
    trans_mx[:2, 2] = trans_vec

    return trans_mx


def get_scale_mx(s_x, s_y):
    '''
    Input:
        s_x -- Scaling along the x axis
        s_y -- Scaling along the y axis
        s_z -- Scaling along the z axis
    Output:
        A 4x4 numpy array representing 3D scaling.
    '''
    # Note: For MOPS, you need to use s_x and s_y only, since we are in 2D
    scale_mx = np.eye(3)

    for i, s in enumerate([s_x, s_y]):
        scale_mx[i, i] = s

    return scale_mx

# print(get_rot_mx(np.pi/6))
# print(cv2.getRotationMatrix2D((0,0),-30,1))
# img = cv2.imread("resources/triangle1.jpg")
# mat1 = get_scale_mx(2,2)
# mat1 = np.delete(mat1,2,axis=0)
# mat2 = get_rot_mx(np.pi/6)
# mat2 = np.delete(mat2,2,axis=0)
# desc = cv2.warpAffine(img,mat2,(img.shape[0],img.shape[1]))
# cv2.imshow("0",img)
# cv2.imshow("1",desc)
# cv2.waitKey(0)