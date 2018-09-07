"""Check rollouts.

Actually, I might just use this script for investigating rollouts ...
So, AFTER we deploy the bed-making robot, we can use this to analyze.
BTW, do this AFTER I have converted Honda's data over to my format, but I won't
have time to check all the keys so some of this is dependent on the data source.
"""
import pickle, os, sys, cv2
import numpy as np
from os.path import join
from fast_grasp_detect.data_aug.depth_preprocess import depth_to_net_dim
from fast_grasp_detect.data_aug.draw_cross_hair import DrawPrediction

# The 'FIGHEAD' is really this (plus the rollout type), but we replace 'results'
# with 'results_all_figs' for simplicity.
HEAD = '/nfs/diskstation/seita/bed-make/results'
DP = DrawPrediction()

path_theirs = sorted([join(HEAD,x) for x in os.listdir(HEAD) if 'honda' in x])
path_mine   = sorted([join(HEAD,x) for x in os.listdir(HEAD) if 'deploy' in x])


def process(pth):
    print("\n===================\nOn pth:        {}".format(pth))
    fig_pth = pth.replace('/results/', '/results_all_figs/')
    print("with fig_path: {}".format(fig_pth))
    if not os.path.exists(fig_pth):
        os.makedirs(fig_pth)
    rollouts = sorted([join(pth,x) for x in os.listdir(pth) if 'results_rollout' in x])
    return fig_pth, rollouts


def make_fig_path(pkl_file):
    fig_for_rollout = pkl_file.replace('/results/','/results_all_figs/')
    assert fig_for_rollout[-2:] == '.p', fig_for_rollout
    fig_for_rollout = fig_for_rollout[:-2]
    if not os.path.exists(fig_for_rollout):
        os.makedirs(fig_for_rollout)
    return fig_for_rollout


def iterate(dtype):
    """Remember, need to run `convert_h_rollouts_to_mine.py` before this.
    """
    if dtype == 'mine':
        for pth in path_mine:
            _, rollouts = process(pth)
            # Do some debugging of the first one 
            first = True # For debugging prints

            for pkl_file in rollouts:
                with open(pkl_file, 'r') as f:
                    data = pickle.load(f)
                print("loaded: {}, has length {}".format(pkl_file, len(data)))
                fig_path_rollout = make_fig_path(pkl_file)
        
                # Analyze _my_ data, with the HSR. Note the depth image processing.
                # For all grasp-related images we might as well overlay the poses.

                for t_idx,datum in enumerate(data[:-1]):
                    if first:
                        print(datum['side'], datum['type'])
                    c_img = datum['c_img']
                    d_img = datum['d_img']
                    assert c_img.shape == (480,640,3) and d_img.shape == (480,640)
                    d_img = depth_to_net_dim(d_img, robot='HSR')
                    t_str = str(t_idx).zfill(2)
                    pth1 = join(fig_path_rollout, 'c_img_{}.png'.format(t_str))
                    pth2 = join(fig_path_rollout, 'd_img_{}.png'.format(t_str))
                    if datum['type'] == 'grasp':
                        c_img = DP.draw_prediction(c_img, datum['net_pose'])
                        d_img = DP.draw_prediction(d_img, datum['net_pose'])
                    cv2.imwrite(pth1, c_img)
                    cv2.imwrite(pth2, d_img)

                # I saved a final dictionary.
                final_dict = data[-1]
                image_start = final_dict['image_start']
                image_final = final_dict['image_final']
                pth_s = join(fig_path_rollout, 'image_start.png')
                pth_f = join(fig_path_rollout, 'image_final.png')
                cv2.imwrite(pth_s, image_start)
                cv2.imwrite(pth_f, image_final)
                first = False


    elif dtype == 'theirs':
        for pth in path_theirs:
            _, rollouts = process(pth)
            first = True # For debugging prints

            for pkl_file in rollouts:
                with open(pkl_file, 'r') as f:
                    data = pickle.load(f)
                print("loaded: {}, has length {}".format(pkl_file, len(data)))
                fig_path_rollout = make_fig_path(pkl_file)
        
                # Analyze _my_ data, with the HSR. We don't need to process the
                # depth image because they saved the processed depth image.  See
                # Prakash's email for the keys in each `datum`.  It starts at
                # BOTTOM then goes to the top.

                for t_idx,datum in enumerate(data[:-1]):
                    if first:
                        print(datum['side'], datum['pose'], datum['class'])
                        print("({:.1f}, {:.2f}, {:.1f})".format(datum['t_grasp'], datum['t_fwrd_pass'], datum['t_transition']))
                    c_img = datum['c_img']
                    d_img = datum['d_img']
                    assert c_img.shape == (480,640,3) and d_img.shape == (480,640,3)
                    # Actually they already processed it.
                    #d_img = depth_to_net_dim(d_img, robot='Fetch')
                    o_img = datum['overhead_img']
                    t_str = str(t_idx).zfill(2)
                    pth1 = join(fig_path_rollout, 'c_img_{}.png'.format(t_str))
                    pth2 = join(fig_path_rollout, 'd_img_{}.png'.format(t_str))
                    pth3 = join(fig_path_rollout, 'o_img_{}.png'.format(t_str))
                    cv2.imwrite(pth1, c_img)
                    cv2.imwrite(pth2, d_img)
                    cv2.imwrite(pth3, o_img)

                # I saved a final dictionary to try and match it with my data.
                final_dict = data[-1]

                first = False


if __name__ == "__main__":
    iterate('theirs')
    iterate('mine')
