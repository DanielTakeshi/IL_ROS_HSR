import numpy as np
import cv2
import IPython
from connected_components import singulate

def threshold_binarize(img, tolerance):
    #3 color channels
    dim = 3

    #create the image histogram
    split = cv2.split(img)
    hists = np.array([cv2.calcHist([split[i]], [0], None, [256], [0, 256]) for i in range(dim)])

    #find the threshold as the mode of the image
    modes = [np.argmax(hists[i]) for i in range(dim)]

    #add the tolerance to the bounds
    lb = np.array([modes[i] - tolerance for i in range(dim)])
    ub = np.array([modes[i] + tolerance for i in range(dim)])

    img = cv2.inRange(img, lb, ub)
    #switch black and white
    img[:,:,] = (255 - img[:,:,])
    return img


def run_connected_components(img, dist_tol=5, color_tol=45):
    img = np.copy(img)

    #crop image (change to detect automatically)
    lo_y, hi_y = 65, 165
    lo_x, hi_x = 220, 470
    crop = [slice(lo_y, hi_y), slice(lo_x, hi_x)]
    img = img[crop[0], crop[1]]

    #threshhold the background and reduce noise
    img = threshold_binarize(img, color_tol)
    img = cv2.medianBlur(img, 3)

    center_masses, directions = singulate(img, dist_tol)
    #transform centroids to uncropped image
    center_masses = [[c[0] + lo_y, c[1] + lo_x] for c in center_masses]
    return center_masses, directions

def draw(img, center_masses, directions):
    box_color = [255, 0, 0]
    box_size = 5
    line_color = box_color[::-1]
    line_size = 20

    for i in range(len(center_masses)):
        cm = center_masses[i]
        d = directions[i] #True if y orientation

        img[int(cm[0] - box_size):int(cm[0] + box_size),
            int(cm[1] - box_size):int(cm[1] + box_size)] = box_color
        if d:
            img[int(cm[0] - line_size):int(cm[0] + line_size),
            int(cm[1] - 1):int(cm[1] + 1)] = line_color
        else:
            img[int(cm[0] - 1):int(cm[0] + 1),
            int(cm[1] - line_size):int(cm[1] + line_size)] = line_color

    cv2.imshow('debug',img)
    cv2.waitKey(30)
    return img

if __name__ == "__main__":
    #number of pixels apart to be singulated
    dist_tol = 5
    #background range for threshholding the image
    color_tol = 45

    for img_num in range(15):
        name_pre = "images/frame_40_" + str(img_num)
        img = cv2.imread(name_pre + ".png")

        center_masses, directions = run_connected_components(img, dist_tol, color_tol)

        img_out = draw(img, center_masses, directions)

        cv2.imwrite(name_pre + "_out.png", img_out)
