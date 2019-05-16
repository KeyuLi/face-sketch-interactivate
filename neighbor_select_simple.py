import time
import numpy as np
from sklearn.neighbors import KDTree
import cv2


def neighbor_select_simple(photo, photo_list, sketch_list, y1, y2, x1, x2):
    candidates = []
    photo_base = photo.reshape(1, -1)

    p_neighbor = np.zeros((photo_list.shape[0], 3 * (y2 - y1) * (x2 - x1)))
    s_neighbor = np.zeros((photo_list.shape[0], 3 * (y2 - y1) * (x2 - x1)))

    for img_idx in np.arange(photo_list.shape[0]):
        p_patch = photo_list[img_idx, :, :, :][y1:y2, x1:x2].clip(0, 255)

        p_patch = p_patch.flatten()

        s_patch = sketch_list[img_idx, :, :, :][y1:y2, x1:x2]
        s_patch = s_patch.flatten()

        p_neighbor[img_idx, :] = p_patch
        s_neighbor[img_idx, :] = s_patch

    kdtree = KDTree(p_neighbor, metric='l2')
    dist, idx = kdtree.query(photo_base, k=1)
    photo_neighbor = p_neighbor[idx[0, :], :].reshape(((y2 - y1), (x2 - x1), 3))
    sketch_neighbor = s_neighbor[idx[0, :], :].reshape(((y2 - y1),  (x2 - x1), 3))
    # photo_neighbor_pre = np.zeros((y2 - y1, x2 - x1, 3))
    # sketch_neighbor_pre = np.zeros((y2 - y1, x2 - x1, 3))
    # for c in np.arange(3):
    #     photo_neighbor_pre[:, :, c] = photo_neighbor[:, c].reshape((y2 - y1), (x2 - x1))

    # cv2.imshow('img1', photo_neighbor.astype('uint8'))
    # cv2.waitKey()
    candidates.append(sketch_neighbor.astype('uint8'))
    candidates.append(photo_neighbor.astype('uint8'))

    return candidates
