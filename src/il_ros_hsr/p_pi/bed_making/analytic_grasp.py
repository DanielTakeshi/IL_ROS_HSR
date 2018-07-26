import cv2, IPython, sys
import numpy as np
import cPickle as pickle
from il_ros_hsr.p_pi.bed_making.analytic_supp import Analytic_Supp
from fast_grasp_detect.visualizers.draw_cross_hair import DrawPrediction


class Analytic_Grasp:

    def __init__(self):
        self.supp = Analytic_Supp()
        self.dp = DrawPrediction()


    def get_grasp(self, img, scale, fname=None, color=None):
        """Called during external bed deployment code to get the analytic grasp.

        Args:
            img: Copy of the image, in standard numpy form, but scaled by `scale`.
                For example, a (640,480) image is usually shrunk by `scale=3`.
            scale: Factor we use to shrink image dimensionality.
            fname: File name of where we'd save image with the grasps overlaid.
                This way we can save 'intermediate' images for debugging various
                cv2 functions.
            color: TODO
        """
        # TODOs: need to check compatibility with newer, updated data.
        side = self.supp.get_side(img, fname)

        # heuristic size found empirically to detect strips of white on side
        # side == -1 means more blue on right side, else more on left side.
        # Somewhat confusingly, `get_blue_below` will call `get_blob`, but with
        # the condition of `supp.is_blue`, not `supp.is_white`?
        if side == -1:
            largest = self.supp.get_blue_below(img)
            prediction = min(largest, key = lambda p: p[0][0])[0]
        else:
            largest, size = self.supp.get_blob(img, self.supp.is_white)
            prediction = min(largest, key = lambda p: p[0][0] * side)[0]

        self.supp.draw_both(np.copy(img), prediction, largest)
        return prediction
