"""
Example to show how to extract images from a webcam.
Use this for taking a top-down view of the workspace.

https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
(My OpenCV version is 3.3)
"""
import numpy as np
import cv2, sys, time


def test0():
    """At least this streaming works.
    """
    cap = cv2.VideoCapture(0)
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


def test1():
    """This unbelievably doesn't do what I want ??

    With and without the imshow() shenanigans.
    """
    cap = cv2.VideoCapture(0)
    
    frame = None
    while frame is None:
        ret, frame = cap.read()
    print(frame, ret)
    cv2.imshow('frame',frame)
    cv2.waitKey(0)
    cv2.imwrite('img_start.png',frame)
    
    # move the blanket!
    print("now sleeping ... ... ... ... ... ... ... ... ... MOVE THE BLANKET ...")
    time.sleep(10)
    print("finished sleeping ...")
    
    frame = None
    while frame is None:
        ret, frame = cap.read()
    print(frame, ret)
    cv2.imshow('frame',frame)
    cv2.waitKey(0)
    cv2.imwrite('img_end.png',frame)


def test2():
    """Try two different video captures.

    THIS WORKS! The key is you need to have the first video capture release its
    source. THEN you can make the second one.
    """
    cap = cv2.VideoCapture(0)
    # Do NOT make `cap2 = cv2.VideoCapture(0)` until after we `cap.release()`.
    
    frame = None
    while frame is None:
        ret, frame = cap.read()
    print(frame, ret)
    #cv2.imshow('frame',frame)
    #cv2.waitKey(0)
    cv2.imwrite('img_start.png',frame)
    cap.release()
    
    # move the blanket!
    print("now sleeping ... MOVE THE BLANKET ...")
    time.sleep(10)
    print("finished sleeping ...")
    
    cap2 = cv2.VideoCapture(0)
    frame = None
    while frame is None:
        ret, frame = cap2.read()
    print(frame, ret)
    #cv2.imshow('frame',frame)
    #cv2.waitKey(0)
    cv2.imwrite('img_end.png',frame)
    cap2.release()


if __name__ == "__main__":
    test0()
    #test1()
    #test2()

    print("done")
