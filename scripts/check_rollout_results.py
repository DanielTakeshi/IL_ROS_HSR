"""Check rollouts.

Actually, I might just use this script for investigating rollouts ...
So, AFTER we deploy the bed-making robot, we can use this to analyze.

UPDATE: in progress!!
"""
import pickle, os, sys, cv2
import numpy as np
from os.path import join

HEAD = '/nfs/diskstation/seita/bed-make/results'
paths = sorted([x for x in os.listdir(HEAD) if 'honda' in x])
FIGHEAD = '/nfs/diskstation/seita/bed-make/results_rollouts_figs/'

for pth in paths:
    print("\nOn pth: {}".format(pth))
    rollouts = sorted([x for x in os.listdir(join(HEAD,pth)) if 'rollout' in x])

    for rollout_path in rollouts:
        pkl_file = join(HEAD,pth,rollout_path,'rollout.pkl')
        with open(pkl_file, 'r') as f:
            data = pickle.load(f)
        print("Just loaded: {}, has length {}".format(pkl_file, len(data)))
        print("data[0]['choice']: {}  (0=human, 1=analytic, 2=network)".format(data[0]['choice']))

        # Where we save everthing to, i.e. images of each time step.
        figpath = join(FIGHEAD,pth,rollout_path)
        if not os.path.exists(figpath):
            os.makedirs(figpath)

        for t_idx,datum in enumerate(data):
            c_img        = datum['c_img']
            d_img        = datum['d_img']
            overhead_img = datum['overhead_img']

            t_str = str(t_idx).zfill(2)
            pth1 = join(figpath,'c_img_{}.png'.format(t_str))
            #pth2 = join(figpath,'d_img_{}.png'.format(t_str))
            pth3 = join(figpath,'overhead_img_{}.png'.format(t_str))
            cv2.imwrite(pth1, c_img)
            #cv2.imwrite(pth2, d_img)
            cv2.imwrite(pth3, overhead_img)
