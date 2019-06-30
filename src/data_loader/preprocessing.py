import numpy as np
import scipy
import os
import cv2
from sklearn.preprocessing import MinMaxScaler

def get_labels(paths):
    labels = [[int(os.path.basename(path)[0])] for path in paths]
    for l in labels:
        if l[0] != 0 and l[0] != 1:
            print("Error in training set label is not binary, has value of: ", l[0])
            exit()
    return labels



def paths_to_images(paths, target_dims):
    imgs = []
    for path in paths:
        img = scipy.misc.imread(path, mode='RGB').astype(np.float)
        h, w, _ = img.shape
        img1 = img[:, :int(w/2), :]
        img2 = img[:, int(w/2):, :]

        img1_range = img1.max() - img1.min()
        img2_range = img2.max() - img2.min()

        img1 = (img1 - img1.min()) / (img1_range + 1e-5)
        img2 = (img2 - img2.min()) / (img2_range + 1e-5)

        if img1.min() < -1 or img1.max() > 1:
            print(" error in dataset normalization")
            exit()

        if h != target_dims[0] or w/2 != target_dims[1]:
            img1 = scipy.misc.imresize(img1, (target_dims[0], target_dims[1]))
            img2 = scipy.misc.imresize(img2, (target_dims[0], target_dims[1]))


        imgs.append((img1, img2))

    imgs = np.asanyarray(imgs)
    return imgs


def rgb_to_bgr(img):
    """Converts RGB to BGR

    Args:
        img: input image of color bytes arrangement R->G->B.

    Returns:
        Same image with color bytes arrangement B->G->R.
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
