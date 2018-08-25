"""Use this script for inspecting COVERAGE.

For this, we need the human to laboriously write a contour for the 'exposed'
part of the table. If there is no contour, the human should make a contour that
does not intersect with the table top. Then the intersection is zero, etc. :-)

Note that it's important that the human contour be in order ... so be careful with
labeling of the points!
"""
import argparse, cv2, os, pickle, sys, matplotlib, utils
from os.path import join
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
import numpy as np
np.set_printoptions(suppress=True, linewidth=200, precision=4)
from fast_grasp_detect.data_aug.depth_preprocess import datum_to_net_dim
from collections import defaultdict

# ------------------------------------------------------------------------------
# ADJUST.
# ------------------------------------------------------------------------------
HEAD     = '/nfs/diskstation/seita/bed-make/results/'
RESULTS  = join(HEAD, 'deploy_network')
FIGURES  = join(HEAD, 'figures')
ESC_KEYS = [27, 1048603]
BLACK    = (0, 0, 0)
GREEN    = (0, 255, 0)
RED      = (0, 0, 255)
WHITE    = (255, 255, 255)
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


def call_wait_key(nothing=None):
    """Use like: call_wait_key(cv2.imshow(...))"""
    key = cv2.waitKey(0)
    if key in ESC_KEYS:
        print("Pressed ESC key. Terminating program ...")
        sys.exit()


def is_blue(p):
    b, g, r = p
    return (b > 150 and (r < b - 40) and (g < b - 40)) or (r < b - 50) or (g < b - 50)


def is_white(p):
    b, g, r = p
    return b > 200 and r > 200 and g > 200


def get_blob(img, condition):
    """
    Find largest blob (contour) of some color. Return it, and the area.
    Update: and the masking image.
    """
    bools = np.apply_along_axis(condition, 2, img)
    mask = np.where(bools, 255, 0)
    mask = mask.astype(np.uint8)

    # Bleh this was the old version ...
    #(contours, _) = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # And newer version of cv2 has three items to return.
    (_, contours, _) = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    print("len(contours): {}".format(len(contours)))
    largest = max(contours, key = lambda cnt: cv2.contourArea(cnt))
    return largest, cv2.contourArea(largest), mask


# For `click_and_crop`.
POINTS          = []
CENTER_OF_BOXES = []
# For `click_and_crop_v2`.
POINTS_2  = []
CENTERS_2 = []


def click_and_crop(event, x, y, flags, param):
    global POINTS, CENTER_OF_BOXES
             
    # If left mouse button clicked, record the starting (x,y) coordinates 
    # and indicate that cropping is being performed
    if event == cv2.EVENT_LBUTTONDOWN:
        POINTS.append((x,y))
                                                 
    # Check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # Record ending (x,y) coordinates and indicate that cropping is finished AND save center!
        POINTS.append((x,y))

        upper_left = POINTS[-2]
        lower_right = POINTS[-1]
        assert upper_left[0] < lower_right[0]
        assert upper_left[1] < lower_right[1]
        center_x = int(upper_left[0] + (lower_right[0]-upper_left[0])/2)
        center_y = int(upper_left[1] + (lower_right[1]-upper_left[1])/2)
        CENTER_OF_BOXES.append( (center_x,center_y) )
        
        # Draw a rectangle around the region of interest, w/center point. Blue=Before, Red=AfteR.
        cv2.rectangle(img=img_for_click, 
                      pt1=POINTS[-2], 
                      pt2=POINTS[-1], 
                      color=RED, 
                      thickness=1)
        cv2.circle(img=img_for_click, 
                   center=CENTER_OF_BOXES[-1], 
                   radius=3, 
                   color=RED, 
                   thickness=-1)
        cv2.imshow("This is in the click and crop method AFTER the rectangle. "+
                   "(Press any key, or ESC If I made a mistake.)", img_for_click)


def click_and_crop_v2(event, x, y, flags, param):
    global POINTS_2, CENTERS_2

    # If left mouse button clicked, record the starting (x,y) coordinates 
    # and indicate that cropping is being performed
    if event == cv2.EVENT_LBUTTONDOWN:
        POINTS_2.append((x,y))
                                                 
    # Check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # Record ending (x,y) coordinates and indicate that cropping is finished AND save center!
        POINTS_2.append((x,y))

        upper_left = POINTS_2[-2]
        lower_right = POINTS_2[-1]
        assert upper_left[0] < lower_right[0]
        assert upper_left[1] < lower_right[1]
        center_x = int(upper_left[0] + (lower_right[0]-upper_left[0])/2)
        center_y = int(upper_left[1] + (lower_right[1]-upper_left[1])/2)
        CENTERS_2.append( (center_x,center_y) )
        
        # Draw a rectangle around the region of interest, w/center point. Blue=Before, Red=AfteR.
        cv2.rectangle(img=img_for_click, 
                      pt1=POINTS_2[-2], 
                      pt2=POINTS_2[-1], 
                      color=GREEN,
                      thickness=1)
        cv2.circle(img=img_for_click, 
                   center=CENTERS_2[-1], 
                   radius=3,
                   color=GREEN,
                   thickness=-1)
        cv2.imshow("CLICK ESC WHEN DONE.", img_for_click)



if __name__ == "__main__":
    # --------------------------------------------------------------------------
    # For coverage I think it's easier to deal with global variables.
    # Easier way is to assume we have bounding box and can crop there.
    # But, we might as well support this 'slower way' where we manually hit the corners
    # There is no easy solution since the viewpoints will differ slightly anyway.
    # Thus let's manually go through the files. If we want to compute averages, we can
    # just do an `os.listdir()` and parse the file names (be CAREFUL with file names!).
    # --------------------------------------------------------------------------

    pp = argparse.ArgumentParser()
    pp.add_argument('path', type=str, help='Include FULL path, i.e., `/nfs/disk...` etc')
    args = pp.parse_args()
    assert '/nfs/diskstation/seita/bed-make/results/' in args.path

    # For saving later! Ignore the `deploy_network/results{...}.p` and go to the figures.
    HEAD = (args.path).split('/')[:-2] 
    HEAD = join('/'.join(HEAD), 'figures/') # a bit roundabout lol

    # `tail` should be: 'results_rollout_N_len_K.p' where `N` is the rollout index.
    tail = (args.path).split('/')[-1]
    assert 'results_rollout_' in tail and tail[-2:] == '.p'
    pth = tail.split('_')
    pth[-1] = pth[-1].replace('.p','') # dump '.p'
    assert pth[0] == 'results' and pth[1] == 'rollout' and pth[3] == 'len', pth
    rollout_index = int(pth[2])
    length = int(pth[4])

    # --------------------------------------------------------------------------
    # Extract relevant images and do a whole bunch of checks.
    # For now we take the start c_img and the final_c_img. For coverage, look at latter.
    # But later, we might report _relative_ coverage, hence why I use the first c_img.
    # Don't forget to do a whole bunch of copying before doing any cv2 operations ...!
    # --------------------------------------------------------------------------

    with open(args.path, 'r') as f:
        data = pickle.load(f)
    assert len(data) == length
    print("Loaded data. Here's the dictionary:")
    for d_idx,d_item in enumerate(data):
        print('  {} {}'.format(d_idx, d_item.keys()))

    print("\nFor now we focus on the first and last c_img\n")
    c_img_start = data[0]['c_img']
    c_img_end_t = data[-2]['c_img']
    c_img_end_s = data[-1]['final_c_img']
    assert c_img_start.shape == c_img_end_t.shape == c_img_end_s.shape == (480,640,3)

    # ----------------------------------------------------------------------
    # Coverage before. First, have user draw 4 bounding boxes by the bed edges.
    # Then press any key and we proceed. Double check length of points, etc.
    # DRAW CLOCKWISE STARTING FROM THE UPPER LEFT CORNER!!!
    # `img_for_click` contains the corners ... ignore image after this.
    # ----------------------------------------------------------------------

    num_targ = len(CENTER_OF_BOXES) + 4
    print("current points {}, target {}".format(len(CENTER_OF_BOXES), num_targ))

    while len(CENTER_OF_BOXES) < num_targ:
        img_for_click = c_img_start.copy()
        diff = num_targ - len(CENTER_OF_BOXES)
        wn = 'image at start, num points {}'.format(diff)
        cv2.namedWindow(wn, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(wn, click_and_crop)
        cv2.imshow(wn, img_for_click)
        key = cv2.waitKey(0)

    # After pressing, CENTER_OF_BOXES has four points.
    print("should now be four center points, let's close all windows and move forward ...")
    cv2.destroyAllWindows()
    print("here are the last four points btw: {}".format(CENTER_OF_BOXES[-4:]))
    last4 = CENTER_OF_BOXES[-4:]
    assert len(CENTER_OF_BOXES) % 4 == 0

    # ----------------------------------------------------------------------
    # We will have a user click manually as many times as is necessary to form
    # an accurate rendering of the contour. But please remember to hit ESC ...
    #
    # And visualize both human-made contour and the detected blue.
    # Abort here by clicking the ESC key if things are messy.
    # We have `largest` and `human_ctr` as two contours to double check.
    # API for draw contours: (1) image, (2) _list_ of contours, (3) index of the
    # contour to draw, or -1 for all of them, (4) color, (5) thickness. Whew!
    # Use `image_viz` for visualization purposes.
    # ----------------------------------------------------------------------

    while key not in ESC_KEYS:
        wn = 'image for blue region, num points so far {}'.format(len(CENTERS_2))
        cv2.namedWindow(wn, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(wn, click_and_crop_v2)
        cv2.imshow(wn, img_for_click)
        key = cv2.waitKey(0)

    print("Just hit ESC, have {} points for the blue contour".format(len(CENTERS_2)))

    # human_ctr is the table top, largest_c is the exposed blue region (or nothing).
    largest_c = np.array(CENTERS_2).reshape((-1,1,2)).astype(np.int32)
    human_ctr = np.array(last4).reshape((-1,1,2)).astype(np.int32)

    size           = cv2.contourArea(largest_c)
    human_ctr_area = cv2.contourArea(human_ctr)

    image_viz = c_img_start.copy()
    cv2.drawContours(image_viz, [human_ctr], 0, BLACK, 2)
    cv2.drawContours(image_viz, [largest_c], 0, GREEN, 2)
    largest_ctr_area = size
    print("\ncontour area of human_ctr: {}".format(human_ctr_area))
    print("contour area of largest: {}".format(largest_ctr_area))
    cc = 'ESC to abort, any other key to continue'
    call_wait_key(cv2.imshow(cc, image_viz))
    cv2.destroyAllWindows()

    # ----------------------------------------------------------------------
    # Some tricky stuff. We're going to find the intersection of the two
    # contours above. For this, make a blank image. Draw the two contours. Then
    # take their logical and. Must ensure we have thickness=-1 to fill in.
    # Don't forget that this results in 0s and 1s, and we want 0s and 255s.
    # ----------------------------------------------------------------------

    print("\nHere's the intersection of the above two")
    blank = np.zeros( (480,640) )
    img1 = cv2.drawContours( blank.copy(), [human_ctr], contourIdx=0, color=1, thickness=-1 )
    img2 = cv2.drawContours( blank.copy(), [largest_c], contourIdx=0, color=1, thickness=-1 )
    intersection = np.logical_and( img1, img2 ).astype(np.uint8) * 255.0
    intersection = intersection.astype(np.uint8)
    cc = 'Intersection between two previous contours'
    call_wait_key(cv2.imshow(cc, intersection))
    cv2.destroyAllWindows()

    # ----------------------------------------------------------------------
    # Find contour on the `intersection` image which should be easy! Visualize
    # using the original `image_viz` which had the original two contours. Then
    # take the area ratio and save images. Be careful to save both the original
    # one and another one which has the contours ...
    # ----------------------------------------------------------------------
 
    (_, contours_int, _) = cv2.findContours(intersection.copy(),
            cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print("\nChecked contours for 'intersection', length {}".format(len(contours_int)))
    largest_int = max(contours_int, key = lambda cnt: cv2.contourArea(cnt))

    AA = float(cv2.contourArea(human_ctr))
    BB = float(cv2.contourArea(largest_int))
    coverage = (1.0 - (BB / AA)) * 100.0
    print("area of full bed frame: {}".format(AA))
    print("area of exposed blue:   {}".format(BB))
    print("coverage of sheet:      {:.4f}".format(coverage))
    print("the original blue area was {} but includes other stuff in scene".format(size))

    cv2.drawContours(image_viz, [largest_int], 0, RED, thickness=2)
    cc = 'Red indicates intersection'
    call_wait_key(cv2.imshow(cc, image_viz))
    cv2.destroyAllWindows()

    # Original: c_img_start. Visualization: `image_viz`.
    path_1 = join(HEAD, 'res_{}_c_start_raw.png'.format(rollout_index))
    path_2 = join(HEAD, 'res_{}_c_start_{:.1f}.png'.format(rollout_index, coverage))
    cv2.imwrite(path_1, c_img_start)
    cv2.imwrite(path_2, image_viz)
    print("\nJust saved: {}".format(path_1))
    print("Just saved: {}\n".format(path_2))



    sys.exit()


    # ----------------------------------------------------------------------
    # Repeat the process for the final image!
    # ----------------------------------------------------------------------
    # Change c_img_start --> c_img_end_s (or c_img_end_t!)
    # ----------------------------------------------------------------------
    # Recommend to use c_img_end_s, though. For both the ending ones, lighting
    # poses a problem, unfortunately ...
    # ----------------------------------------------------------------------

    final_img = c_img_end_t
    #final_img = c_img_end_s

    # Coverage.

    num_targ = len(CENTER_OF_BOXES) + 4
    print("current points {}, target {}".format(len(CENTER_OF_BOXES), num_targ))

    while len(CENTER_OF_BOXES) < num_targ:
        img_for_click = final_img.copy()
        diff = num_targ - len(CENTER_OF_BOXES)
        wn = 'image at start, num points {}'.format(diff)
        cv2.namedWindow(wn, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(wn, click_and_crop)
        cv2.imshow(wn, img_for_click)
        key = cv2.waitKey(0)

    # After pressing anNow should be four points.
    print("should now be four center points, let's close all windows and move forward ...")
    cv2.destroyAllWindows()
    print("here are the last four points btw: {}".format(CENTER_OF_BOXES[-4:]))
    last4 = CENTER_OF_BOXES[-4:]
    assert len(CENTER_OF_BOXES) % 4 == 0

    # blobs

    image_viz = final_img.copy()
    #largest_c, size, mask = get_blob( image_viz.copy(), is_white )
    largest_c, size, mask = get_blob( image_viz.copy(), is_blue )

    # visualize

    human_ctr = np.array(last4).reshape((-1,1,2)).astype(np.int32)
    cv2.drawContours(image_viz, [human_ctr], 0, BLACK, 1)
    cv2.drawContours(image_viz, [largest_c], 0, GREEN, 1)
    human_ctr_area = cv2.contourArea(human_ctr)
    largest_ctr_area = size
    print("\ncontour area of human_ctr: {}".format(human_ctr_area))
    print("contour area of largest: {}".format(largest_ctr_area))
    cc = 'ESC to abort, any other key to continue'
    call_wait_key(cv2.imshow(cc, image_viz))
    cv2.destroyAllWindows()

    # intersections
    # note: if this is empty, usually perfect coverage or
    # likely a problem we'd have to hand-tune anyway.

    print("\nHere's the intersection of the above two")
    blank = np.zeros( (480,640) )
    img1 = cv2.drawContours( blank.copy(), [human_ctr], contourIdx=0, color=1, thickness=-1 )
    img2 = cv2.drawContours( blank.copy(), [largest_c], contourIdx=0, color=1, thickness=-1 )
    intersection = np.logical_and( img1, img2 ).astype(np.uint8) * 255.0
    intersection = intersection.astype(np.uint8)
    cc = 'Intersection between two previous contours'
    call_wait_key(cv2.imshow(cc, intersection))
    cv2.destroyAllWindows()

    # contour on intersection
 
    (_, contours_int, _) = cv2.findContours(intersection.copy(),
            cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print("\nChecked contours for 'intersection', length {}".format(len(contours_int)))

    if len(contours_int) == 0:
        print("LENGTH 0, automatically assuming 100% coverage.")
        AA = float(cv2.contourArea(human_ctr))
        BB = 0.0
        coverage = 100.0
    else:
        largest_int = max(contours_int, key = lambda cnt: cv2.contourArea(cnt))
        AA = float(cv2.contourArea(human_ctr))
        BB = float(cv2.contourArea(largest_int))
        coverage = (1.0 - (BB / AA)) * 100.0

    print("area of full bed frame: {}".format(AA))
    print("area of exposed blue:   {}".format(BB))
    print("coverage of sheet:      {:.4f}".format(coverage))
    print("the original blue area was {} but includes other stuff in scene".format(size))

    if len(contours_int) > 0:
        cv2.drawContours(image_viz, [largest_int], 0, RED, thickness=1)
    cc = 'Red indicates intersection'
    call_wait_key(cv2.imshow(cc, image_viz))
    cv2.destroyAllWindows()

    # Original: c_img_end_s (final_img). Visualization: `image_viz`.

    path_1 = join(HEAD, 'res_{}_c_end_raw.png'.format(rollout_index))
    path_2 = join(HEAD, 'res_{}_c_end_{:.1f}.png'.format(rollout_index, coverage))
    cv2.imwrite(path_1, final_img)
    cv2.imwrite(path_2, image_viz)
    print("\nJust saved: {}".format(path_1))
    print("Just saved: {}\n".format(path_2))
