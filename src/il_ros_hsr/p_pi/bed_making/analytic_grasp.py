import cv2
import numpy as np
import cPickle as pickle
import IPython
from il_ros_hsr.p_pi.bed_making.analytic_supp import Analytic_Supp

from data_aug.draw_cross_hair import DrawPrediction

class Analytic_Grasp:
    def __init__(self):
        self.supp = Analytic_Supp()
        self.dp = DrawPrediction()

    def get_grasp(self, img, scale):
        side = self.supp.get_side(img)

        # #heuristic size found empirically to detect strips of white on side
        if side == -1:
            largest = self.supp.get_blue_below(img)
            prediction = min(largest, key = lambda p: p[0][0])[0]
        else:
            largest, size = self.supp.get_blob(img, self.supp.is_white)
            prediction = min(largest, key = lambda p: p[0][0] * side)[0]

        self.supp.draw_both(np.copy(img), prediction, largest)

        return prediction
