"""
Convert their rollout directory to my form, so that I can seamlessly integrate
my coverage results, etc. Take what they download and put it in the
`HEAD_THEIRS` directory, so that's in their format. Then I need to put it in my
results.

HEAD_THEIRS should be a list of rollout directories, each of which has a file
called `rollout.pkl` in it.

I also realize, I should probably change their dict so it has the same keys ...

DON'T RUN THIS MORE THAN ONCE! Or at least, be careful, because this will keep
adding more rollouts ...
"""
import pickle, os, sys, cv2
import numpy as np
from os.path import join

# Just paste where their results are in HEAD_THEIRS.
HEAD_THEIRS = '/nfs/diskstation/seita/bed-make/results_honda_v02'
HEAD_MINE   = '/nfs/diskstation/seita/bed-make/results'


paths = sorted([join(HEAD_THEIRS,x) for x in os.listdir(HEAD_THEIRS)])

for pth in paths:
    print("\nOn pth: {}".format(pth))
    pkl_file = join(pth,'rollout.pkl')
    with open(pkl_file, 'r') as f:
        data = pickle.load(f)
    print("Just loaded: {}, has length {}".format(pkl_file, len(data)))
    print("data[0]['choice']: {}  (0=human, 1=analytic, 2=network)".format(data[0]['choice']))
    choice = data[0]['choice']
    for idx,datum in enumerate(data):
        assert datum['choice'] == choice

    # Save it to appropriate directory.
    if choice == 0:
        target = join(HEAD_MINE,'honda_human')
    elif choice == 1:
        target = join(HEAD_MINE,'honda_analytic')
    elif choice == 2:
        target = join(HEAD_MINE,'honda_network_white')

    num_existing = len( [x for x in os.listdir(target) if 'results_rollout' in x] ) 
    K = len(data)
    file_name = 'results_rollout_{}_len_{}.p'.format(num_existing,K)
    target = join(target, file_name)

    # So, I think I will do this. Add a new item which is a dict and has info I
    # need. This will NOT be counted as part of the length, for now, because I
    # think they already do something similar in the last element of data ...

    last_dict = {}
    last_dict['image_start'] = data[0]['overhead_img']
    last_dict['image_final'] = data[-1]['overhead_img']
    data.append(last_dict)

    with open(target, 'w') as f:
        pickle.dump(data, f)
    print("just saved:\n{}".format(target))
