"""
Example to show how to extract images from a webcam.
Use this for taking a top-down view of the workspace.

https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
(My OpenCV version is 3.3)
"""
import numpy as np
import cv2, sys

cap = cv2.VideoCapture(0)

# It will be ready right away, i.e., no 'startup' period.
ret, frame = cap.read()
print(frame, ret)
sys.exit()



while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = frame

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
