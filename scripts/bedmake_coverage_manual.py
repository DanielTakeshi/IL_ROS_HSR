"""Use this script for inspecting COVERAGE.

For this, we need the human to laboriously write a contour for the 'exposed'
part of the table. If there is no contour, the human should make a contour that
does not intersect with the table top. Then the intersection is zero, etc. :-)

It's important that the human contour be in order ... so be careful with labeling
of the points! If something is wrong kill the program and delete any files created.

THIS WILL SAVE A BUNCH OF IMAGES WITH CONTOURS BUT NOT MAKE PLOTS ... for that we
need a separate script that goes through the directory and makes stats about it.

We go one by one because due to lighting differences, nothing will be perfect and
I think this is going to be the most accurate way to do this.

To call, this, pass in the FULL path to the pickle file, for one rollout. Yes, I
know, sorry. Like this:

python scripts/bedmake_coveage_manual.py \
    /nfs/diskstation/seita/bed-make/results/deploy_network/results_rollout_0_len_7.p

And then that will handle reults for this rollout. The `deploy_network` needs to be
adjusted or calibrated based on WHICH network/data we used, BTW. So it will be like:
`deploy_network_combo_v01` or `deploy_human` for a human supervisor, or `deploy_analytic`
for an analytic baseline, etc. Then in the `figures` sub-directory, we will have these
file names as additional sub-directories, etc.
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
        cv2.imshow("Press ESC if mistake, space to continue", img_for_click)


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
        cv2.imshow("Click ESC when done filling exposed contour", img_for_click)



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
    # See later for prints to confirm ... a lot of manual work. :-(
    HEAD = (args.path).split('/')[:-2]
    HEAD = join('/'.join(HEAD), 'figures/') # a bit roundabout lol
    HEAD_last2 = (args.path).split('/')[-2:]
    ROLLOUT_TYPE = HEAD_last2[0]
    FIGDIR = join(HEAD, ROLLOUT_TYPE)

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

    # I included one extra item ...
    if 'honda' in args.path:
        assert len(data) == length+1
    else:
        assert len(data) == length

    print("Loaded rollout data. Here's the dictionary we recorded:")
    for d_idx,d_item in enumerate(data):
        print('  {} {}'.format(d_idx, d_item.keys()))
    print("\nHEAD:          {}".format(HEAD))
    print("HEAD_last2:    {}".format(HEAD_last2))
    print("ROLLOUT_TYPE:  {}".format(ROLLOUT_TYPE))
    print("tail:          {}".format(tail))
    print("FIGDIR:        {}".format(FIGDIR))
    print("(the FIGDIR is where we save)\n")
    if not os.path.exists(FIGDIR):
        os.makedirs(FIGDIR)
    #sys.exit()

    print("\nFor now we focus on the first and last c_img\n")
    image_start = data[-1]['image_start']  # or do image_start2
    image_final = data[-1]['image_final']  # or do image_final2
    #c_img_start = data[0]['c_img']
    #c_img_end_t = data[-2]['c_img']
    #c_img_end_s = data[-1]['final_c_img']
    assert image_start.shape == image_final.shape == (480,640,3)
    #assert image_start.shape == c_img_end_t.shape == c_img_end_s.shape == (480,640,3)

    # ------------------------- END OF PROCESSING ----------------------------

    # ----------------------------------------------------------------------
    # Coverage before. First, have user draw 4 bounding boxes by the bed edges.
    # Then press any key and we proceed. Double check length of points, etc.
    # DRAW CLOCKWISE STARTING FROM THE UPPER LEFT CORNER!!!
    # `img_for_click` contains the corners ... ignore image after this.
    # ----------------------------------------------------------------------

    num_targ = len(CENTER_OF_BOXES) + 4
    print("current points {}, target {}".format(len(CENTER_OF_BOXES), num_targ))

    while len(CENTER_OF_BOXES) < num_targ:
        img_for_click = image_start.copy()
        wn = 'Click for table frame border!'
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
        wn = 'Now click for blue/exposed region.'
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

    image_viz = image_start.copy()
    cv2.drawContours(image_viz, [human_ctr], 0, BLACK, 2)
    cv2.drawContours(image_viz, [largest_c], 0, GREEN, 2)
    largest_ctr_area = size
    print("\ncontour area of human_ctr: {}".format(human_ctr_area))
    print("contour area of largest: {}".format(largest_ctr_area))
    cc = 'ESC to abort, SPACE to continue'
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
    cc = 'Intersection between two contours (press SPACE)'
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
    cc = 'Red indicates intersection (press SPACE)'
    call_wait_key(cv2.imshow(cc, image_viz))
    cv2.destroyAllWindows()

    # Original: image_start. Visualization: `image_viz`.
    path_1 = join(FIGDIR, 'res_{}_c_start_raw.png'.format(rollout_index))
    path_2 = join(FIGDIR, 'res_{}_c_start_{:.2f}.png'.format(rollout_index, coverage))
    cv2.imwrite(path_1, image_start)
    cv2.imwrite(path_2, image_viz)
    print("\nJust saved: {}".format(path_1))
    print("Just saved: {}\n".format(path_2))

    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    # Repeat the process for the final image!
    # ----------------------------------------------------------------------
    # Change image_start --> image_final
    # ----------------------------------------------------------------------
    # Recommend to use c_img_end_s, though. For both the ending ones, lighting
    # poses a problem, unfortunately ... which is why we hand-draw the contour!
    # UPDATE: never mind, we have a top-down view now. :-)
    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------

    print("\nMOVING ON TO THE SECOND IMAGE\n")
    final_img = image_final

    num_targ = len(CENTER_OF_BOXES) + 4
    print("current points {}, target {}".format(len(CENTER_OF_BOXES), num_targ))

    while len(CENTER_OF_BOXES) < num_targ:
        img_for_click = final_img.copy()
        wn = 'Click for table frame border!'
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

    # We will have a user click manually as many times as is necessary to form
    # an accurate rendering of the contour. But please remember to hit ESC ...

    num_old_pts = len(CENTERS_2)
    while key not in ESC_KEYS:
        wn = 'Now click for blue/exposed region.'
        cv2.namedWindow(wn, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(wn, click_and_crop_v2)
        cv2.imshow(wn, img_for_click)
        key = cv2.waitKey(0)

    print("Just hit ESC, have {} points for the blue contour".format(len(CENTERS_2) - num_old_pts))

    # human_ctr is the table top, largest_c is the exposed blue region (or nothing).
    largest_c = np.array( CENTERS_2[num_old_pts:] ).reshape((-1,1,2)).astype(np.int32)
    human_ctr = np.array( last4 ).reshape((-1,1,2)).astype(np.int32)

    size           = cv2.contourArea(largest_c)
    human_ctr_area = cv2.contourArea(human_ctr)

    image_viz = final_img.copy()
    cv2.drawContours(image_viz, [human_ctr], 0, BLACK, 2)
    cv2.drawContours(image_viz, [largest_c], 0, GREEN, 2)
    largest_ctr_area = size
    print("\ncontour area of human_ctr: {}".format(human_ctr_area))
    print("contour area of largest: {}".format(largest_ctr_area))
    cc = 'ESC to abort, SPACE to continue'
    call_wait_key(cv2.imshow(cc, image_viz))
    cv2.destroyAllWindows()

    # Some tricky stuff. We're going to find the intersection

    print("\nHere's the intersection of the above two")
    blank = np.zeros( (480,640) )
    img1 = cv2.drawContours( blank.copy(), [human_ctr], contourIdx=0, color=1, thickness=-1 )
    img2 = cv2.drawContours( blank.copy(), [largest_c], contourIdx=0, color=1, thickness=-1 )
    intersection = np.logical_and( img1, img2 ).astype(np.uint8) * 255.0
    intersection = intersection.astype(np.uint8)
    cc = 'Intersection between two contours (press SPACE)'
    call_wait_key(cv2.imshow(cc, intersection))
    cv2.destroyAllWindows()

    # Find contour on the `intersection` image which should be easy! Visualize.

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
    cc = 'Red indicates intersection (press SPACE)'
    call_wait_key(cv2.imshow(cc, image_viz))
    cv2.destroyAllWindows()

    # Original: c_img_end_s (final_img). Visualization: `image_viz`.

    path_1 = join(FIGDIR, 'res_{}_c_end_raw.png'.format(rollout_index))
    path_2 = join(FIGDIR, 'res_{}_c_end_{:.2f}.png'.format(rollout_index, coverage))
    cv2.imwrite(path_1, final_img)
    cv2.imwrite(path_2, image_viz)
    print("\nJust saved: {}".format(path_1))
    print("Just saved: {}\n".format(path_2))
