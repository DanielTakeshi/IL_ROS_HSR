import numpy as np
import cv2
from scipy import signal

def im2tensor_binary(im,channels=3):

    shape = np.shape(im)
    h, w = shape[0], shape[1]
    zeros = np.zeros((h, w, channels))
    cutoff = 140
    for i in range(channels):
        zeros[:,:,i] = np.round(im[:,:,i]/(cutoff * 2.0)) * 255
    return zeros

def contourise(im):
    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    sobelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobelY = np.transpose(sobelX)

    sobConvX = signal.fftconvolve(im, sobelX, mode = "same")
    sobConvY = signal.fftconvolve(im, sobelY, mode = "same")

    return np.hypot(sobConvX, sobConvY)

def bins(im, bits=2):
    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    im = im/255.0
    im = np.ceil(im * 2**bits)/2**bits
    im = im * 255

    return im.astype(np.uint8)

if __name__ == "__main__":
    for i in range(11, 151):
        imgName = "images_in/frame_" + str(i) + ".png"
        img = cv2.imread(imgName)
        imgOut = bins(img,2)
        cv2.imwrite("images_test2/frame_" + str(i) + ".png", imgOut)
