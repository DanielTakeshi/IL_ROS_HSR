import cv2
import pickle
import matplotlib.pyplot as plt
import numpy as np
import time

dirPath = '/home/ron/Desktop/dataset/'
dstDirPath = '/home/ron/Desktop/dataset_zero/'

image_num=1
for i in range(2, 304, 1):
    data = pickle.load(open(dirPath + str(i) + '.pkl', 'rb'))
    RGBImage = data["RGBImage"]
    depthImage = data["depthImage"]
    w,h=depthImage.shape
    # NaNs=
    # NaNs=np.isnan(depthImage).astype(np.uint8)
    # num_of_NaNs=np.sum(NaNs.flat)
    # print 'NaNs % in the image:',num_of_NaNs/w/h*100
    # cv2.imshow('NaNs',NaNs*255)
    # cv2.waitKey(1000)
    # raw_input()
    # continue
    no_NaNs_image = cv2.patchNaNs(depthImage,0)
    # ones=np.ones((w,h)).astype(np.float32)
    # notNaNs=~np.isnan(depthImage)*ones
    # no_NaNs_image=notNaNs*depthImage
    outData = {
        "RGBImage": RGBImage,
        "depthImage": no_NaNs_image,
        "armState": data["armState"],
        "headState": data["headState"],
        "markerPos": data["markerPos"]
    }
    pickle.dump(outData, open(dstDirPath + str(image_num) + ".pkl", "wb"))
    image_num=image_num+1
    # time.sleep(0.1)
    print(str(image_num))
    continue
    # print(np.count_nonzero(np.isnan(depthImage)))

    print(no_NaNs_image[200,200]-depthImage[200,200])
    plt.figure()
    mng = plt.get_current_fig_manager()
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(RGBImage, cv2.COLOR_BGR2RGB))
    plt.subplot(2, 2, 2)
    plt.imshow(depthImage)
    plt.subplot(2, 2, 3)
    zer=np.zeros((w,h)).astype(np.uint8)
    plt.imshow(no_NaNs_image)

    mng.resize(*mng.window.maxsize())
    plt.show()
