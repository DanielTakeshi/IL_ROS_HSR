"""Use this script for inspecting results after doing bed-making deployment.

See the bed-making deployment code for how we saved things.
There are lots of things we can do for inspection.
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
# ADJUST. HH is directory like: 'grasp_1_img_depth_opt_adam_lr_0.0001_{etc...}'
# ------------------------------------------------------------------------------
HEAD = '/nfs/diskstation/seita/bed-make/results/'
RESULTS = join(HEAD, 'deploy_network')
FIGURES = join(HEAD, 'figures')

# For the plot(s). There are a few plot-specific parameters, though.
tsize = 30
xsize = 25
ysize = 25
tick_size = 25
legend_size = 25
alpha = 0.5
error_alpha = 0.3
error_fc = 'blue'

ESC_KEYS = [27, 1048603]
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

def call_wait_key(nothing=None):
    """Use like: call_wait_key(cv2.imshow(...))"""
    key = cv2.waitKey(0)
    if key in ESC_KEYS:
        print("Pressed ESC key. Terminating program ...")
        sys.exit()


def analyze_time(stats):
    """For analyzing time. These should be in seconds unless otherwise stated.
    """

    # Robot motion to another side. Very slow. Should include both sides.
    move_t = []
    for stats_l in stats['move_times']:
        for tt in stats_l:
            move_t.append(tt)
    print("\nTimes for moving to other side, length: {}".format(len(move_t)))
    print("{:.3f}\pm {:.1f}".format(np.mean(move_t), np.std(move_t)))

    # Robot grasp execution. Also slow, can be highly variable.
    grasp_t = []
    for stats_l in stats['grasp_times']:
        for tt in stats_l:
            grasp_t.append(tt)
    print("\nTimes for executing grasps, length: {}".format(len(grasp_t)))
    print("{:.3f}\pm {:.1f}".format(np.mean(grasp_t), np.std(grasp_t)))

    # Now times for grasping and success net. These are quick.
    grasp_net_t = []
    success_net_t = []
    lengths = []
    idx = 0
    key = 'result_{}'.format(idx)

    while key in stats:
        # ----------------------------------------------------------------------
        # Analyze one rollout (i.e., `stats[key]`) at a time.
        # This is a list, where at each index, result[i] is a dict w/relevant
        # info. Also, I ignore the last `final_dict` for this purpose.
        # ----------------------------------------------------------------------
        result = (stats[key])[:-1]

        for i,info in enumerate(result):
            if result[i]['type'] == 'grasp':
                grasp_net_t.append( result[i]['g_net_time'] )
            else:
                assert result[i]['type'] == 'success'
                success_net_t.append( result[i]['s_net_time'] )
        idx += 1
        key = 'result_{}'.format(idx)
        lengths.append(len(result))
    assert len(grasp_net_t) == len(grasp_net_t)

    # For the grasp/success nets, if they're the same architecture, prob combine them
    grasp_net_t = np.array(grasp_net_t)
    success_net_t = np.array(success_net_t)
    print("\ngrasp_net_t.shape: {}".format(grasp_net_t.shape))
    print("{:.3f}\pm {:.1f}".format(np.mean(grasp_net_t), np.std(grasp_net_t)))
    print("\nsuccess_net_t.shape: {}".format(success_net_t.shape))
    print("{:.3f}\pm {:.1f}".format(np.mean(success_net_t), np.std(success_net_t)))
    all_net_t = np.concatenate((grasp_net_t,success_net_t))
    print("\nboth networks, data shape: {}".format(all_net_t.shape))
    print("{:.3f}\pm {:.1f}".format(np.mean(all_net_t), np.std(all_net_t)))

    # Another thing, trajectory _lengths_.
    print("\nlengths.mean(): {}".format(np.mean(lengths)))
    print("lengths.std():  {}".format(np.std(lengths)))
    print("lengths.min():  {}".format(np.min(lengths)))
    print("lengths.max():  {}".format(np.max(lengths)))


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

BLACK  = (0,0,0)
YELLOW = (0,255,0) # well, green ...
RED    = (255,0,0)
WHITE  = (255,255,255)

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
                      color=(0,0,255), 
                      thickness=2)
        cv2.circle(img=img_for_click, 
                   center=CENTER_OF_BOXES[-1], 
                   radius=6, 
                   color=(0,0,255), 
                   thickness=-1)
        #cv2.putText(img=img_for_click, 
        #            text="{}".format(CENTER_OF_BOXES[-1]), 
        #            org=CENTER_OF_BOXES[-1],  
        #            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
        #            fontScale=1, 
        #            color=(0,0,255), 
        #            thickness=2)
        cv2.imshow("This is in the click and crop method AFTER the rectangle. "+
                   "(Press any key, or ESC If I made a mistake.)", img_for_click)


if __name__ == "__main__":
    p_files = sorted([join(RESULTS,x) for x in os.listdir(RESULTS) if 'results_rollout' in x])

    # stuff info here for plotting, etc.
    stats = defaultdict(list)

    for p_idx, p_file in enumerate(p_files):
        with open(p_file, 'r') as f:
            data = pickle.load(f)
        print("\n==============================================================")
        print("loaded file #{} at {}".format(p_idx, p_file))

        # All items except last one should reflect some grasp or success nets.
        # Actually let's pass in the full data, and ignore the last one elsewhere.
        key = 'result_{}'.format(p_idx)
        stats[key] = data
       
        # We know the final dict has some useful stuff in it
        final_dict = data[-1]
        assert 'move_times' in final_dict and 'grasp_times' in final_dict \
                and 'final_c_img' in final_dict
        stats['move_times'].append( final_dict['move_times'] )
        stats['grasp_times'].append( final_dict['grasp_times'] )

    analyze_time(stats)

    # --------------------------------------------------------------------------
    # For this I think it's easier to deal with global variables.
    # Haven't been able to figure out why, honestly...
    # Easier way is to assume we have bounding box and can crop there.
    # But, we might as well support this 'slower way' where we manually hit the corners
    # There is no easy solution since the viewpoints will differ slightly anyway.
    # --------------------------------------------------------------------------

    print("\nNow analyzing _coverage_ ...")
    idx = 0
    key = 'result_{}'.format(idx)
    beginning_coverage = []
    ending_coverage = []

    while key in stats:
        # This is ONE trajectory, so evaluate and add a data point here.
        result = stats[key]
        print("Just loaded: {}, with len {}".format(key, len(result)))
        
        # Extract relevant images
        c_img_start = result[0]['c_img']
        c_img_end_t = result[-2]['c_img']
        c_img_end_s = result[-1]['final_c_img']
        assert c_img_start.shape == c_img_end_t.shape == c_img_end_s.shape

        # Analyze coverage.
        start = np.copy(c_img_start)
        end   = np.copy(c_img_end_s)

        # ----------------------------------------------------------------------
        # Coverage before
        # ----------------------------------------------------------------------
        num_targ = len(CENTER_OF_BOXES) + 4
        print("current points {}, target {}".format(len(CENTER_OF_BOXES), num_targ))
        print("DRAW CLOCKWISE FROM UPPER LEFT CORNER")
        typ = 'start'

        while len(CENTER_OF_BOXES) < num_targ:
            img_for_click = np.copy(start)
            diff = num_targ - len(CENTER_OF_BOXES)
            wn = 'idx {}, type: {}, num_p: {}'.format(idx, typ, diff)
            cv2.namedWindow(wn, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(wn, click_and_crop)
            cv2.imshow(wn, img_for_click)
            key = cv2.waitKey(0)
        
        # Now should be four points.
        print("should now be four center points, let's close all windows and draw....")
        cv2.destroyAllWindows()
        print("and also do the cv contour detection ...")
        print("last four points: {}".format(CENTER_OF_BOXES[-4:]))
        image = np.copy(img_for_click)
        last4 = CENTER_OF_BOXES[-4:]
        assert len(CENTER_OF_BOXES) % 4 == 0

        # Experiment with detecting blobs. Note: mask.shape == (480,640), i.e., single channel.
        #largest, size, mask = get_blob( image, is_white )
        largest, size, mask = get_blob( image, is_blue )

        # Find how much of 'largest' is inside the contained points.

        # Draw stuff.
        # Visualize. Once user presses key, go ahead and save. :-)
        # You can make a line of the points here but better to just make it another contour.
        # So we have `largest` and `human_ctr` as two contours.
        # ----------------------------------------------------------------------
        #cv2.line(image, last4[0], last4[1], YELLOW, 1)
        #cv2.line(image, last4[1], last4[2], YELLOW, 1)
        #cv2.line(image, last4[2], last4[3], YELLOW, 1)
        #cv2.line(image, last4[3], last4[0], YELLOW, 1)
        cv2.drawContours(image, [largest], -1, YELLOW, 1)
        human_ctr = np.array(last4).reshape((-1,1,2)).astype(np.int32)
        cv2.drawContours(image, [human_ctr], 0, RED , 1)
        print("contour area of human_ctr: {}".format(cv2.contourArea(human_ctr)))
        print("contour area of largest: {}".format(size))
        cc = 'ESC abort, other key to save. Rollout: {}, at {}'.format(key, typ)
        call_wait_key(cv2.imshow(cc, image))


        # draw contours, for 3rd argument its the index of the contour to draw (or -1 tod raw all).
        # Remaining argument is color.
        blank = np.zeros( (480,640) )
        img1 = cv2.drawContours( blank.copy(), [human_ctr], contourIdx=0, color=1, thickness=-1 )
        img2 = cv2.drawContours( blank.copy(), [largest], contourIdx=0, color=1, thickness=-1 )
        # now AND the two together and save the intersection image if we scale 1 -> 255.
        intersection = np.logical_and( img1, img2 ).astype(np.uint8) * 255.0
        intersection = intersection.astype(np.uint8)
        cv2.imwrite('intersection.png', intersection)

        # find contour on that image, should be easy because it's intersection ....
        (_, contours_intersection, _) = cv2.findContours(intersection.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print("len(contours_intersection): {}".format(len(contours_intersection)))
        largest_intersection = max(contours_intersection, key = lambda cnt: cv2.contourArea(cnt))
        cv2.drawContours(intersection, [largest_intersection], 0, RED, -1)
        cv2.imwrite('largest_intersection.png', intersection)


        # finally, take the area ratio!
        AA = float(cv2.contourArea(human_ctr))
        BB = float(cv2.contourArea(largest_intersection))
        print("area of full bed frame: {}".format(AA))
        print("area of exposed blue:   {}".format(BB))
        print("ratio (i.e., coverage): {}".format(BB / AA))
        print("the original blue area was {} but includes other stuff in scene".format(size))
        sys.exit()


        # ----------------------------------------------------------------------
        # Save images, record stats, etc. Some shenanigans to saving to ensuring
        # that we don't waste time and effort on this...
        # ----------------------------------------------------------------------
        path_start = join(FIGURES, 'res_{}_c_start.png'.format(idx))
        path_end_t = join(FIGURES, 'res_{}_c_end_t.png'.format(idx))
        path_end_s = join(FIGURES, 'res_{}_c_end_s.png'.format(idx))
        cv2.imwrite(path_start, c_img_start)
        cv2.imwrite(path_end_t, c_img_end_t)
        cv2.imwrite(path_end_s, c_img_end_s)
        idx += 1
        key = 'result_{}'.format(idx)
        # Count coverage TODO
        # start_coverage = ...
        # end_coverage = ...
        # beginning_coverage.append( start_coverage )
        # ending_coverage.append( end_coverage )
        # TODO: save intermediate runs due to high chance of error?

    print("Look at: {}".format(FIGURES))
    print("beginning_coverage: {:.1f}\pm {:.1f}, range ({:.1f},{:.1f})".format(
            np.mean(beginning_coverage),
            np.std(beginning_coverage),
            np.min(beginning_coverage),
            np.max(beginning_coverage))
    )
    print("ending_coverage: {:.1f}\pm {:.1f}, range ({:.1f},{:.1f})".format(
            np.mean(ending_coverage),
            np.std(ending_coverage),
            np.min(ending_coverage),
            np.max(ending_coverage))
    )
