import cv2, sys
import numpy as np
import cPickle as pickle

class Analytic_Supp:
    """Helper for the analytic grasping and success policies."""

    def draw_points(self, img, points):
        img = np.copy(img)
        delta = 5
        for p in points:
            for i in range(p[0] - delta, p[0] + delta):
                for j in range(p[1] - delta, p[1] + delta):
                    if i >= 0 and j >= 0 and i < len(img) and j < len(img[0]):
                        img[i][j] = (0, 255, 255)
        return img


    def is_blue(self, p):
        b, g, r = p
        return (b > 150 and (r < b - 40) and (g < b - 40)) or (r < b - 50) or (g < b - 50)


    def is_white(self, p):
        b, g, r = p
        return b > 200 and r > 200 and g > 200


    def get_blob(self, img, condition):
        """Find largest blob (contour) of some color. Return it, and the area."""
        bools = np.apply_along_axis(condition, 2, img)
        mask = np.where(bools, 255, 0)
        mask = mask.astype(np.uint8)

        # Bleh this was the old version ...
        #(contours, _) = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # And I think newer version of cv2 has three items to return.
        (_, contours, _) = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        print("\n`Analytic_Supp.get_blob()`, len(contours): {}".format(len(contours)))
        largest = max(contours, key = lambda cnt: cv2.contourArea(cnt))
        return largest, cv2.contourArea(largest)


    def get_side(self, img, color='blue', fname=None, textcolor=(0,0,0)):
        """Counts whether majority of pixels of a color are on left or right side, assuming
        `img` is a 'typical' BGR 3-channel image.

        By default we use blue but we can generalize to other colors. This will also save
        images for debugging purposes if `fname` is not None. Doesn't tell us what to do,
        etc., it just tells us which side has more of the color, and we go from there.

        Returns: 1 for left (i.e., more of target color there), -1 for right.
        """
        if color is None or color == 'blue':
            bools = np.apply_along_axis(self.is_blue, 2, img)
        elif color == 'white':
            bools = np.apply_along_axis(self.is_white, 2, img)
        else:
            raise ValueError(color)

        # Where 'bools' is true (i.e., matching color), return 1. Else, return 0.
        mask = np.where(bools, 1, 0)
        mid = len(img[0])/2
        left_c = np.sum(mask[:,:mid])
        right_c = np.sum(mask[:,mid:])

        if fname is not None:
            # Sanity check to show exact left_c, right_c quantities.
            c_img = cv2.resize(np.copy(img), (640,480))
            cv2.putText(c_img,
                        text="left_c: {}, right_c: {}".format(left_c, right_c),
                        org=(0,440),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=textcolor,
                        thickness=2)
            fname = fname.replace('.png', '_side.png')
            cv2.imwrite(fname, c_img)

        return 1 if left_c > right_c else -1


    def get_blue_below(self, img, cutoff=0.4):
        largest, size = self.get_blob(img, self.is_blue)
        pix_cutoff = len(img) * cutoff
        return np.array([c for c in largest if c[0][1] > pix_cutoff])


    def draw_both(self, img, prediction, contour):
        cv2.drawContours(img, [contour], -1, (0,255,0), 3)
        if prediction is not None:
            rc_prediction = [prediction[1], prediction[0]]
            img = self.draw_points(img, [rc_prediction])
        write = True
        if write:
            cv2.imwrite("sample.png", img)
