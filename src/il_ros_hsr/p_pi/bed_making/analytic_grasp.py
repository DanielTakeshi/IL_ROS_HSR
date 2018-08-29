import cv2, IPython, sys
import numpy as np
import cPickle as pickle
from il_ros_hsr.p_pi.bed_making.analytic_supp import Analytic_Supp
from fast_grasp_detect.data_aug.draw_cross_hair import DrawPrediction


class Analytic_Grasp:

    def __init__(self, sheet='blue'):
        self.supp = Analytic_Supp()
        self.dp = DrawPrediction()
        self.sheet = sheet


    def predict(self, img):
        """Should be the newer analytic way with blanket height.
        """
        raise NotImplementedError()


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
        side = self.supp.get_side(img, fname=fname)

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


    def get_grasp_white_sheet(self, img, scale, fname=None, color=None):
        """Called during external bed deployment code to get the analytic grasp.

        This is for newer data based on Ron's white sheets.

        TODO: will need to test more thoroughly about detecting the blue 'sheet' that's
        fixed to the frame; not sure if the way we do blue is OK.

        Args:
            img: Copy of the image, in standard numpy form, but scaled by `scale`.
                For example, a (640,480) image is usually shrunk by `scale=3`.
            scale: Factor we use to shrink image dimensionality.
            fname: File name of where we'd save image with the grasps overlaid.
                This way we can save 'intermediate' images for debugging various
                cv2 functions.
            color: TODO
        """
        side_w = self.supp.get_side(img,
                color='white',
                fname=fname.replace('.png','_w.png'),
                textcolor=(0,255,0)
        )
        side_b = self.supp.get_side(img,
                color='blue',
                fname=fname.replace('.png','_b.png'),
                textcolor=(0,255,0)
        )

        largest_w, size_w = self.supp.get_blob(img, self.supp.is_white)
        largest_b, size_b = self.supp.get_blob(img, self.supp.is_blue)

        def draw_save_cnt(image, c, fname, y_med):
            """
            Assumes `c` is from `cv2.findContours(...) and then taking one of them.
            So, it's a 3-axis numpy array, where each item is itself a (1,2)-sized numpy
            array, so it's an [[x,y]] coordinate. Also a median point to keep us informed
            of an upper bound on where we should be grasping.
            """
            cc = (0,255,0)
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            cv2.drawContours(image, [approx], contourIdx=-1, color=cc, thickness=1)
            image = cv2.resize(image, (640,480))
            cv2.line(image, (0,y_med), (640,y_med), color=cc, thickness=1)
            cv2.imwrite(fname, image)

        # Find north and south points, respectively, for extreme y coordinate values.
        # https://www.pyimagesearch.com/2016/04/11/finding-extreme-points-in-contours-with-opencv/
        c = largest_w
        y_top = tuple(c[c[:, :, 1].argmin()][0])
        y_bot = tuple(c[c[:, :, 1].argmax()][0])
        _, y_top = y_top
        _, y_bot = y_bot
        y_med = int((y_top+y_bot) / 2)

        # AH! We have to multiply by the scale, right? I _think_ OK as it's along 1 axis...
        y_med *= scale

        draw_save_cnt(img.copy(), largest_w, fname.replace('.png','_w_c.png'), y_med)
        draw_save_cnt(img.copy(), largest_b, fname.replace('.png','_b_c.png'), y_med)

        # As an alternative to these, we could just use image cutoff based on a
        # percentage of the image height. Sorry about this, it's really annoying
        # due to the scaling factor since contours assume we have the _scaled_ img...
        if side_w == -1:
            # More white on the RIGHT side of blanket (i.e., 'top' view).
            # Find leftmost white point on contour with ycoord > y_med.
            stuff = np.array([x for x in largest_w if x[0][1] > (y_med/scale)])
            prediction = min(stuff, key=lambda p: p[0][0])[0]
        else:
            # More white on the LEFT side of blanket (i.e, 'bottom' view).
            # Find rightmost white point on contour with ycoord > y_med.
            stuff = np.array([x for x in largest_w if x[0][1] > (y_med/scale)])
            prediction = max(stuff, key=lambda p: p[0][0])[0]

        return prediction
