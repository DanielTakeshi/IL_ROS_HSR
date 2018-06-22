"""Quick script to investigate a rollout.
Helps me to understand how Michael encoded the data.
"""

import cv2, os, pickle, sys
import numpy as np
np.set_printoptions(suppress=True, linewidth=200)

# I just copied some of Michael's trajs and put them here.0
HEAD = '/Users/danielseita/bed-making'
indices = [i for i in range(20,29)]
indices.append(2)

for i in indices:
    IMG_PATH = os.path.join(HEAD, 'rollout_'+str(i))
    PATH     = os.path.join(HEAD, 'rollout_'+str(i), 'rollout.p')
    info     = pickle.load(open(PATH))
    print("\n=====================================================================")
    print("=====================================================================")
    print("IMG_PATH {}".format(IMG_PATH))
    for idx,item in enumerate(info):
        print("On item {}".format(idx))
        print(item)
        if idx > 0:
            cv2.imwrite(os.path.join(IMG_PATH, 'img_'+str(idx)+'.png'), item['c_img'])
    print("=====================================================================")
    print("=====================================================================")
