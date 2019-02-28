# -*- coding: utf-8 -*-
"""
Create on 2019/2/28 15:40
Create by ring
Function Description:
"""
import numpy as np
import cv2
import scipy.sparse
from scipy.sparse.linalg import spsolve
import sys
import os
sys.path.append('../../')

def laplacian_matrix(n, m):
    """Generate the Poisson matrix.

    Refer to:
    https://en.wikipedia.org/wiki/Discrete_Poisson_equation

    Note: it's the transpose of the wiki's matrix
    """
    mat_D = scipy.sparse.lil_matrix((m, m))  # 构建稀疏矩阵
    mat_D.setdiag(-1, -1)                    # 构建拉普拉斯矩阵
    mat_D.setdiag(4)
    mat_D.setdiag(-1, 1)

    mat_A = scipy.sparse.block_diag([mat_D] * n).tolil()  # 构建稀疏矩阵块

    mat_A.setdiag(-1, 1*m)
    mat_A.setdiag(-1, -1*m)

    return mat_A


def poisson_core(source, target, mask, offset):
    """
    The poisson blending function.
    Refer to:
    Perez et. al., "Poisson Image Editing", 2003.
    """
    # Assume:
    # target is not smaller than source.
    # shape of mask is same as shape of target.
    source, target, mask = source.copy(), target.copy(), mask.copy()
    # print(source.shape[2])
    # cv2.waitKey(0)
    # print(target.shape[:-1])
    y_max, x_max = target.shape[:-1]
    y_min, x_min = 0, 0

    x_range = x_max - x_min
    y_range = y_max - y_min

    M = np.float32([[1, 0, offset[0]], [0, 1, offset[1]]])
    source = cv2.warpAffine(source, M, (x_range, y_range))

    mask = mask[y_min:y_max, x_min:x_max]
    mask[mask != 0] = 1
    # mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

    mat_A = laplacian_matrix(y_range, x_range)

    # for \Delta g
    laplacian = mat_A.tocsc()

    # set the region outside the mask to identity
    for y in range(1, y_range - 1):
        for x in range(1, x_range - 1):
            if mask[y, x] == 0:
                k = x + y * x_range
                mat_A[k, k] = 1
                mat_A[k, k + 1] = 0
                mat_A[k, k - 1] = 0
                mat_A[k, k + x_range] = 0
                mat_A[k, k - x_range] = 0

    # corners
    # mask[0, 0]
    # mask[0, y_range-1]
    # mask[x_range-1, 0]
    # mask[x_range-1, y_range-1]

    mat_A = mat_A.tocsc()

    mask_flat = mask.flatten()
    for channel in range(source.shape[2]):
        source_flat = source[y_min:y_max, x_min:x_max, channel].flatten()
        target_flat = target[y_min:y_max, x_min:x_max, channel].flatten()

        #concat = source_flat*mask_flat + target_flat*(1-mask_flat)

        # inside the mask:
        # \Delta f = div v = \Delta g
        alpha = 1
        mat_b = laplacian.dot(source_flat)*alpha

        # outside the mask:
        # f = t
        mat_b[mask_flat==0] = target_flat[mask_flat==0]

        x = spsolve(mat_A, mat_b)
        #print(x.shape)
        x = x.reshape((y_range, x_range))
        #print(x.shape)
        x[x > 255] = 255
        x[x < 0] = 0
        x = x.astype('uint8')
        # x = cv2.normalize(x, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        # print(x.shape)

        target[y_min:y_max, x_min:x_max, channel] = x

    return target


def poisson_edit(source, mask, target):
    h, w = target.shape[0], target.shape[1]
    source = cv2.resize(source, (w, h))
    mask = cv2.resize(mask, (w, h))
    _, mask = cv2.threshold(mask, 127, 255, 0)
    offset = (0, 0)
    result = poisson_core(source, target, mask, offset)
    return result

if __name__ == '__main__':
    pass

