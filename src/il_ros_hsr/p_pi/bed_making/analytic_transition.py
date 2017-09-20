import cv2
import numpy as np
import cPickle as pickle
from il_ros_hsr.p_pi.bed_making.analytic_supp import Analytic_Supp

class Analytic_Transition:
    def __init__(self):
        self.supp = Analytic_Supp()

    def predict(self, img, scale):
        largest, size = self.supp.get_blob(img, self.supp.is_blue)
        side = self.supp.get_side(img)

        self.supp.draw_both(img, None, largest)

        #lower bounds on success obtained from sampling a couple of rollouts blue blob sizes
        side1_thresh = 70000.0/(scale**2)
        side2_thresh = 100000.0/(scale**2)

        return (side == 1 and size > side1_thresh) or (side == -1 and size > side2_thresh)

# Used to calculated threshholds
# if __name__ == "__main__":
#     sample = pickle.load(open("rollouts/rollout_0/rollout.p", "rb"))
#     imgs = [sample[2 + 10 * i]['c_img'] for i in range(4)]
#     supp = Analytic_Supp()
#     sizes = [supp.get_blob(img, supp.is_blue)[1] for img in imgs]
#     print(sizes)
#     #[48660.5, 79579.5, 77867.5, 133412.0]
