"""After running `check_raw_data.py`, use this for plotting.

I normally save images and then arrange then in Google Drawings, but leave this script as an option
in case it's easier (and perhaps saves on file sizes) to programmatically arrange the camera (or
depth!) images to form 'tiled' images for the paper.
"""
import cv2, os, pickle, sys
import numpy as np
np.set_printoptions(suppress=True, linewidth=200)

ROLLOUTS = '/nfs/diskstation/seita/bed-make/rollouts/'
IMAGE_PATH = '/nfs/diskstation/seita/bed-make/images/'
