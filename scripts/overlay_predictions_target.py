"""Use this script for making visuals with overlaid predictions and targets.

Use for visualizing the grasp network's predictions. It's a bit clumsy and lots of copying/pasting
but it will work for now. Make sure you inspect the outcome of `python main/train_bed_grasp.py` to
see the exact labels.

        For Daniel Cal data:

data_manager.load_test_set(), held-out rollouts: [34, 7, 39, 37, 46] (cv index 0) from
/nfs/diskstation/seita/bed-make/rollouts/
/nfs/diskstation/seita/bed-make/rollouts/rollout_34,  len(grasp_rollout)=3,  w/len(rollout)=7 [TEST]
/nfs/diskstation/seita/bed-make/rollouts/rollout_7,  len(grasp_rollout)=2,  w/len(rollout)=5 [TEST]
/nfs/diskstation/seita/bed-make/rollouts/rollout_39,  len(grasp_rollout)=2,  w/len(rollout)=5 [TEST]
/nfs/diskstation/seita/bed-make/rollouts/rollout_37,  len(grasp_rollout)=2,  w/len(rollout)=5 [TEST]
/nfs/diskstation/seita/bed-make/rollouts/rollout_46,  len(grasp_rollout)=3,  w/len(rollout)=7 [TEST]
len(self.test_labels): 12
test_batch_images.shape: (12, 14, 14, 1024)
test_batch_labels.shape: (12, 2)
test_batch_labels:
[[-0.396094 -0.134375]
 [ 0.363281 -0.110417]
 [ 0.059375 -0.022917]
 [-0.229687 -0.176042]
 [ 0.239062 -0.054167]
 [-0.0625   -0.126042]
 [ 0.177344 -0.120833]
 [-0.204688 -0.132292]
 [-0.053906 -0.003125]
 [-0.135937 -0.15625 ]
 [ 0.369531 -0.144792]
 [ 0.097656 -0.054167]]


        Now for Michael's NYTimes, lol ...:

data_manager.load_test_set(), path: /nfs/diskstation/seita/bed-make/held_out_nytimes/
/nfs/diskstation/seita/bed-make/held_out_nytimes/rollout_23,  len(grasp_rollout)=4,  w/len(rollout)=8 [TEST]
/nfs/diskstation/seita/bed-make/held_out_nytimes/rollout_14,  len(grasp_rollout)=3,  w/len(rollout)=6 [TEST]
/nfs/diskstation/seita/bed-make/held_out_nytimes/rollout_10,  len(grasp_rollout)=1,  w/len(rollout)=2 [TEST]
/nfs/diskstation/seita/bed-make/held_out_nytimes/rollout_19,  len(grasp_rollout)=3,  w/len(rollout)=6 [TEST]
len(self.test_labels): 11
test_batch_images.shape: (11, 14, 14, 1024)
test_batch_labels.shape: (11, 2)
test_batch_labels:
[[ 0.292969 -0.094792]
 [-0.151562 -0.15625 ]
 [ 0.028906  0.057292]
 [ 0.223437 -0.01875 ]
 [-0.291406 -0.135417]
 [ 0.214844 -0.073958]
 [ 0.228125 -0.058333]
 [ 0.188281  0.036458]
 [ 0.232812 -0.138542]
 [-0.2625   -0.171875]
 [ 0.18125  -0.091667]]
(raw) test_batch_labels:
[[507.5 194.5]
 [223.  165. ]
 [338.5 267.5]
 [463.  231. ]
 [133.5 175. ]
 [457.5 204.5]
 [466.  212. ]
 [440.5 257.5]
 [469.  173.5]
 [152.  157.5]
 [436.  196. ]]
"""
import cv2, os, pickle, sys
import numpy as np
np.set_printoptions(suppress=True, linewidth=200)
from fast_grasp_detect.data_aug.depth_preprocess import datum_to_net_dim


DATA = 'DanielCal'
#DATA = 'MichaelNYTimes'

if DATA == 'DanielCal':
    # Daniel Cal data. Use this or Michael's data.
    ROLLOUT_HEAD = '/nfs/diskstation/seita/bed-make/rollouts/'
    ROLLOUTS = [34, 7, 39, 37, 46]
    IMG_PATH = '/nfs/diskstation/seita/bed-make/figures/caldata/'
    LABELS = np.array([
         [ 66.5, 175.5],
         [552.5, 187. ],
         [358. , 229. ],
         [173. , 155.5],
         [473. , 214. ],
         [280. , 179.5],
         [433.5, 182. ],
         [189. , 176.5],
         [285.5, 238.5],
         [233. , 165. ],
         [556.5, 170.5],
         [382.5, 214. ]
    ])
    # This one had: Test loss: 0.003662 (raw: 131.33)
    PREDS = np.array([
        [128.98283 , 170.324407],
        [515.57682 , 180.984392],
        [369.701996, 240.482483],
        [157.534618, 142.987833],
        [463.412628, 202.685609],
        [269.350872, 182.671037],
        [370.449677, 192.724371],
        [224.871044, 177.224379],
        [309.365063, 232.848058],
        [243.062553, 148.531981],
        [522.005806, 168.653669],
        [324.99279 , 206.485519]
    ])

elif DATA == 'MichaelNYTimes':
    # Michael's NYTimes data. Use this or Daniel's Cal data.
    ROLLOUT_HEAD = '/nfs/diskstation/seita/bed-make/held_out_nytimes/'
    ROLLOUTS = [23, 14, 10, 19]
    IMG_PATH = '/nfs/diskstation/seita/bed-make/figures/nytimes/'
    LABELS = np.array([
        [507.5, 194.5],
        [223. , 165. ],
        [338.5, 267.5],
        [463. , 231. ],
        [133.5, 175. ],
        [457.5, 204.5],
        [466. , 212. ],
        [440.5, 257.5],
        [469. , 173.5],
        [152. , 157.5],
        [436. , 196. ]
    ])
    # This one had: Test loss: 0.003685 (raw: 116.08)
    PREDS = np.array([
        [445.24292 , 178.167372],
        [259.14814 , 176.248198],
        [354.15493 , 247.549925],
        [450.059357, 218.486266],
        [149.539433, 161.027741],
        [453.298721, 214.537654],
        [443.569565, 216.162958],
        [435.328789, 216.05145 ],
        [431.146622, 182.257876],
        [187.953796, 170.060949],
        [418.046951, 224.496417],
    ])


idx = 0
for rnum in ROLLOUTS:
    print("\n=====================================================================")
    print("rollout {}".format(rnum))
    path = os.path.join(ROLLOUT_HEAD, 'rollout_{}/rollout.p'.format(rnum))
    if not os.path.exists(path):
        print("Error: {} does not exist".format(path))
        sys.exit()
    data = pickle.load(open(path,'rb'))
    g_in_rollout = 0

    for (d_idx,datum) in enumerate(data):
        # Ignore the first thing which is the 'starting' points.
        if type(datum) == list or datum['type'] == 'success':
            continue
        print("\ncurrently on item {} in this rollout, out of {}:".format(d_idx,len(data)))
        print('type:   {}'.format(datum['type']))
        print('side:   {}'.format(datum['side']))
        print('pose:   {}'.format(datum['pose']))

        # All this does is modify the datum['d_img'] key; it leaves datum['c_img'] alone.
        datum_to_net_dim(datum)

        # Paths, etc.
        c_path = os.path.join(IMG_PATH, 'rollout_{}_grasp_{}_rgb.png'.format(rnum,g_in_rollout))
        d_path = os.path.join(IMG_PATH, 'rollout_{}_grasp_{}_depth.png'.format(rnum,g_in_rollout))
        c_img = (datum['c_img']).copy()
        d_img = (datum['d_img']).copy()
        pose = datum['pose']
        print("LABELS[idx]: {} (should be same as pose)".format(LABELS[idx]))
        print("PREDS[idx]: {} (hopefully close...)".format(PREDS[idx]))
        assert pose[0] == LABELS[idx][0]
        assert pose[1] == LABELS[idx][1]

        # Overlay the pose to the image (red circle, black border).
        pose_int = (int(pose[0]), int(pose[1]))
        cv2.circle(img=c_img, center=pose_int, radius=8, color=(0,0,255), thickness=-1)
        cv2.circle(img=d_img, center=pose_int, radius=8, color=(0,0,255), thickness=-1)
        cv2.circle(img=c_img, center=pose_int, radius=10, color=(0,0,0), thickness=3)
        cv2.circle(img=d_img, center=pose_int, radius=10, color=(0,0,0), thickness=3)

        # The PREDICTION, though, will be a large blue circle (yellow border?).
        preds = (int(PREDS[idx][0]), int(PREDS[idx][1]))
        cv2.circle(img=c_img, center=preds, radius=8, color=(255,0,0), thickness=-1)
        cv2.circle(img=d_img, center=preds, radius=8, color=(255,0,0), thickness=-1)
        cv2.circle(img=c_img, center=preds, radius=10, color=(0,255,0), thickness=3)
        cv2.circle(img=d_img, center=preds, radius=10, color=(0,255,0), thickness=3)

        cv2.imwrite(c_path, c_img)
        cv2.imwrite(d_path, d_img)
        g_in_rollout += 1
        idx += 1

    print("=====================================================================")
