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
from il_ros_hsr.p_pi.bed_making.analytic_supp import Analytic_Supp

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




# For `click_and_crop`.
POINTS          = []
CENTER_OF_BOXES = []

BLACK  = (0,0,0)
YELLOW = (0,255,0)
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
    supp = Analytic_Supp()

    print("\nNow analyzing _coverage_ ...")
    idx = 0
    key = 'result_{}'.format(idx)

    while key in stats:
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

        #  Coverage before
        num_targ = len(CENTER_OF_BOXES) + 4
        print("current points {}, target {}".format(len(CENTER_OF_BOXES), num_targ))
        print("DRAW CLOCKWISE FROM UPPER LEFT CORNER")

        while len(CENTER_OF_BOXES) < num_targ:
            typ = 'start'
            img_for_click = np.copy(start)
            diff = num_targ - len(CENTER_OF_BOXES)
            wn = 'idx {}, type: {}, num_p: {}'.format(idx, typ, diff)
            cv2.namedWindow(wn, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(wn, click_and_crop)
            cv2.imshow(wn, img_for_click)
            key = cv2.waitKey(0)
        
        # Now should be four points.
        print("should now be four center points, let's close all windows...")
        cv2.destroyAllWindows()

        print("now do the cv contour detection ...")
        print("last four points: {}".format(CENTER_OF_BOXES[-4:]))
        image = np.copy(img_for_click)
        largest, size = supp.get_blob( image, supp.is_white )
        cv2.drawContours(image, [largest], -1, YELLOW, 2)
        caption = 'contour. idx {}, type: {}'.format(idx, typ)
        call_wait_key(cv2.imshow(caption, image))


        # Save images, increment key, etc.
        path_start = join(FIGURES, 'res_{}_c_start.png'.format(idx))
        path_end_t = join(FIGURES, 'res_{}_c_end_t.png'.format(idx))
        path_end_s = join(FIGURES, 'res_{}_c_end_s.png'.format(idx))
        cv2.imwrite(path_start, c_img_start)
        cv2.imwrite(path_end_t, c_img_end_t)
        cv2.imwrite(path_end_s, c_img_end_s)
        idx += 1
        key = 'result_{}'.format(idx)

    print("Look at: {}".format(FIGURES))

