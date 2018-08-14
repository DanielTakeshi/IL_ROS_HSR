import os
import Pyro4
import time
import cPickle as pickle
import IPython
import cv2
from fast_grasp_detect.labelers.online_labeler import QueryLabeler
CANVAS_DIM = 420.0


class Python_Labeler:
    """Use this in bed-making code for the pop-up labeler that we use."""

    def __init__(self, cam=None):
        self.cam = cam

    def label_image(self, image=None):
        print("Python_Labeler.label_image(), created `fast_frasp_detect.labelers.online_labeler.QueryLabeler()`")
        if self.cam is not None:
            ql = QueryLabeler()
            ql.run(self.cam)
        else:
            ql = QueryLabeler()
            ql.run(self.cam, image=image)
        data = ql.label_data
        print("  ql.label_data: {}".format(data))
        del ql
        return data
