import cv2
import numpy as np
import IPython
import hsrb_interface

from il_ros_hsr.core.sensors import  RGBD, Gripper_Torque, Joint_Positions
from table_top import TableTop

from il_ros_hsr.p_pi.bed_making.com import Bed_COM as COM
from il_ros_hsr.core.python_labeler import Python_Labeler
import time



# #Used to find crop points
# i = 37
# fname = "rollout" + str(i) + "_tran_1.png"
# img = cv2.imread(fname)
# delta = 5
# for p in points:
# 	for i in range(p[0] - delta, p[0] + delta):
# 		for j in range(p[1] - delta, p[1] + delta):
# 			img[i][j] = (0, 255, 255)
# cv2.imwrite("testout.png", img)


def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
 
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
 
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
 
	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
 
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
 
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
 
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
 
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
 
	# return the warped image
	return warped

def is_blue(p):
	b, g, r = p
	return (b > 150 and (r < b - 40) and (g < b - 40)) or (r < b - 50) or (g < b - 50)

def is_white(p):
	b, g, r = p
	return b > 160 and r > 160 and g > 160

def get_points(img,wl):
	data = wl.label_image(img)

	datum = data['objects'][0]
	  

	x_min = float(datum['box'][0])
	y_min = float(datum['box'][1])
	x_max = float(datum['box'][2])
	y_max = float(datum['box'][3])

	p0 = (x_min,y_min)
	p1 = (x_max,y_min)
	p2 = (x_min,y_max)
	p3 = (x_max,y_max)

	points = np.array([p0, p1, p2, p3])

	return points





def get_success(img,wl):
	#(x, y) crop points of unstranslated bed image


	# p0 = (155, 25)
	# p1 = (540, 35)
	# p2 = (50, 205)
	# p3 = (639, 220)
	# points = np.array([p0, p1, p2, p3])

	points = get_points(img,wl)


	out = four_point_transform(img, points)

	w, h, c = np.shape(out)
	mask = np.zeros((w, h), dtype=np.uint8)

	for i, row in enumerate(out):
		for j, point in enumerate(row):
			if not is_blue(point) and not is_white(point):
				mask[i][j] = 255
			else:
				mask[i][j] = 0

	results = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	contours = results[0]
	largest = max(contours, key = lambda cnt: cv2.contourArea(cnt))
	cv2.drawContours(out, [largest], -1, (0,255,0), 3)


	percent = 100 - round(cv2.contourArea(largest)/(w*h), 2) * 100

	return percent, out


if __name__=='__main__':
	robot =  hsrb_interface.Robot()
	whole_body = robot.get('whole_body')
	omni_base = robot.get('omni_base')

	com = COM()

	cam = RGBD()
	# com.go_to_initial_state(whole_body)



	# tt = TableTop()
	# tt.find_table(robot)

	# tt.move_to_pose(omni_base,'lower_start')
	# whole_body.move_to_joint_positions({'head_tilt_joint':-0.8})
	time.sleep(2)
	wl = Python_Labeler(cam)

	c_img = cam.read_color_data()

	percent, out = get_success(c_img,wl)

	print "PERCENT ", percent
	cv2.imshow('out_image',out)

	cv2.waitKey(30)

	IPython.embed()