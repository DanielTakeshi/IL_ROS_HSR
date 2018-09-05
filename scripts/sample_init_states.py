"""
For seeing the initial state distriution, for plotting later.

USE FOR THE FIGURE IN THE PAPER, where we show the distribution of initial states,
with both depth and RGB. That way we showcase more diversity.
"""
import sys, os, cv2
import cPickle as pickle
import numpy as np
from os.path import join
from fast_grasp_detect.data_aug.depth_preprocess import depth_to_net_dim

HEAD = '/nfs/diskstation/seita/bed-make/results'
OUT = 'tmp/'
if not os.path.exists(OUT):
    os.makedirs(OUT)


def inspect(roll_path, fighead):

    for idx,roll in enumerate(roll_path):
        print("On: {}".format(roll))
        with open(roll, 'r') as f:
            data = pickle.load(f)

        rgb   = data[0]['c_img']
        depth = depth_to_net_dim( data[0]['d_img'], robot='HSR' )
        f_rgb   = '{}_rgb.png'.format(idx)
        f_depth = '{}_depth.png'.format(idx)
        out1 = join(fighead, f_rgb)
        out2 = join(fighead, f_depth)
        cv2.imwrite(out1, rgb)
        cv2.imwrite(out2, depth)

        # also do the next one ...
        rgb   = data[-2]['c_img']
        depth = depth_to_net_dim( data[-2]['d_img'], robot='HSR' )
        f_rgb   = '{}_next_rgb.png'.format(idx)
        f_depth = '{}_next_depth.png'.format(idx)
        out1 = join(fighead, f_rgb)
        out2 = join(fighead, f_depth)
        cv2.imwrite(out1, rgb)
        cv2.imwrite(out2, depth)

    print("")


if __name__ == "__main__":
    net_white = join(HEAD, 'deploy_network_white')
    net_teal  = join(HEAD, 'deploy_network_teal')
    net_cal   = join(HEAD, 'deploy_network_cal')

    net_white_rolls = sorted([join(net_white,x) for x in os.listdir(net_white)])
    net_teal_rolls  = sorted([join(net_teal,x) for x in os.listdir(net_teal)])
    net_cal_rolls   = sorted([join(net_cal,x) for x in os.listdir(net_cal)])

    fig_white = join(OUT,'white')
    fig_teal  = join(OUT,'teal')
    fig_cal   = join(OUT,'cal')

    if not os.path.exists(fig_white):
        os.makedirs(fig_white)
    if not os.path.exists(fig_teal):
        os.makedirs(fig_teal)
    if not os.path.exists(fig_cal):
        os.makedirs(fig_cal)

    inspect(net_white_rolls, fig_white)
    inspect(net_teal_rolls,  fig_teal)
    inspect(net_cal_rolls,   fig_cal)
