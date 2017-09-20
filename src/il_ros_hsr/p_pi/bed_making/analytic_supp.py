import cv2
import numpy as np
import cPickle as pickle

class Analytic_Supp:
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
        bools = np.apply_along_axis(condition, 2, img)
        mask = np.where(bools, 255, 0)
        mask = mask.astype(np.uint8)
        results = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = results[0]
        largest = max(contours, key = lambda cnt: cv2.contourArea(cnt))

        return largest, cv2.contourArea(largest)

    #-1 for side 2, 1 for side 1
    def get_side(self, img):
        #count whether majority of blue pixels are on the left or right
        bools = np.apply_along_axis(self.is_blue, 2, img)
        mask = np.where(bools, 1, 0)
        mid = len(img[0])/2
        left_c = np.sum(mask[:,:mid])
        right_c = np.sum(mask[:,mid:])

        return 1 if left_c > right_c else -1

    def get_blue_below(self, img, cutoff=0.4):
        largest, size = self.get_blob(img, self.is_blue)
        pix_cutoff = len(img) * cutoff
        return np.array([c for c in largest if c[0][1] > pix_cutoff])

    def draw_both(self, img, prediction, contour):
        cv2.drawContours(img, [contour], -1, (0,255,0), 3)

        if prediction != None:
            rc_prediction = [prediction[1], prediction[0]]
            img = self.draw_points(img, [rc_prediction])

        write = True
        if write:
            cv2.imwrite("sample.png", img)