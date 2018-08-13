"""Use this script for making visuals with overlaid predictions and targets.
For Ron's data.

data_manager.load_test_set(), held-out rollouts: [0] (cv index 0) from
/nfs/diskstation/seita/bed-make/rollouts_ron_v02_h0/
/nfs/diskstation/seita/bed-make/rollouts_ron_v02_h0/rollout_0,  len(grasp_rollout)=16,
w/len(rollout)=16 [TEST]
len(self.test_labels): 16
test_batch_images.shape: (16, 14, 14, 1024)
test_batch_labels.shape: (16, 2)
"""
import cv2, os, pickle, sys
import numpy as np
np.set_printoptions(suppress=True, linewidth=200)
from fast_grasp_detect.data_aug.depth_preprocess import datum_to_net_dim


ROLLOUTS = [0]
ROLLOUT_HEAD = '/nfs/diskstation/seita/bed-make/rollouts_ron_v02_c1/'
IMG_PATH = '/nfs/diskstation/seita/bed-make/images_ron_v02/'



LABELS = np.array([
  [489., 252.],
  [186., 366.],
  [111., 342.],
  [403., 257.],
  [288., 299.],
  [284., 271.],
  [127., 250.],
  [163., 346.],
  [371., 197.],
  [356., 217.],
  [406., 216.],
  [109., 357.],
  [235., 351.],
  [416., 241.],
  [359., 221.],
  [451., 213.]
])

# This one had: Test loss: 0.018608 (raw: 66.94)
# (with the correctly averaged raw loss this time ...)
PREDS = np.array([
  [429.959946, 225.561819],
  [207.99921 , 318.51943 ],
  [141.281509, 265.763826],
  [354.308777, 256.633301],
  [252.299385, 295.782909],
  [254.552841, 266.395512],
  [158.184872, 263.582554],
  [146.219883, 384.646454],
  [287.892952, 213.186207],
  [332.984314, 224.20289 ],
  [370.606689, 241.812344],
  [171.789665, 325.821304],
  [263.573799, 315.135984],
  [328.350525, 237.755342],
  [312.800827, 217.859902],
  [298.176994, 186.096382]
])


# Processing, assertions, etc.
idx = 0
rnum = 0
inner = 5
outer = 7
print("\n=====================================================================")
print("rollout {}".format(rnum))
path = os.path.join(ROLLOUT_HEAD, 'rollout_{}/rollout.p'.format(rnum))
if not os.path.exists(path):
    print("Error: {} does not exist".format(path))
    sys.exit()
if not os.path.exists(IMG_PATH):
    print("Error: {} does not exist".format(IMG_PATH))
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
    cv2.circle(img=c_img, center=pose_int, radius=inner, color=(0,0,255), thickness=-1)
    cv2.circle(img=d_img, center=pose_int, radius=inner, color=(0,0,255), thickness=-1)
    cv2.circle(img=c_img, center=pose_int, radius=outer, color=(0,0,0), thickness=3)
    cv2.circle(img=d_img, center=pose_int, radius=outer, color=(0,0,0), thickness=3)

    # The PREDICTION, though, will be a large blue circle (yellow border?).
    preds = (int(PREDS[idx][0]), int(PREDS[idx][1]))
    cv2.circle(img=c_img, center=preds, radius=inner, color=(255,0,0), thickness=-1)
    cv2.circle(img=d_img, center=preds, radius=inner, color=(255,0,0), thickness=-1)
    cv2.circle(img=c_img, center=preds, radius=outer, color=(0,255,0), thickness=3)
    cv2.circle(img=d_img, center=preds, radius=outer, color=(0,255,0), thickness=3)

    cv2.imwrite(c_path, c_img)
    cv2.imwrite(d_path, d_img)
    g_in_rollout += 1
    idx += 1

print("=====================================================================")
