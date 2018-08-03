"""Use this script for making visuals with overlaid predictions and targets.

This automates the task, given some scripts that I ran, e.g. after:
    ./main/grasp.sh | tee logs/grasp.log
Better than copying/pasting, after all...

So, we need a bunch of those pickle files that we stored out of training.
They should have the best set of predictions, along with the actual labels.
From there, we have the information we need to do analysis and figures.
"""
import cv2, os, pickle, sys
import numpy as np
np.set_printoptions(suppress=True, linewidth=200)
from fast_grasp_detect.data_aug.depth_preprocess import datum_to_net_dim


ROLLOUT_HEAD = '/nfs/diskstation/seita/bed-make/rollouts_white_v01/'
IMG_PATH     = '/nfs/diskstation/seita/bed-make/figures/white_v01/'





if __name__ == "__main__":
    head1 = '/nfs/diskstation/seita/bed-make/grasp_output_for_plots/nytimes'
    head2 = '/nfs/diskstation/seita/bed-make/grasp_output_for_plots/danielcal'
    plot(head1, figname1)
    plot(head2, figname2)



def parse_name(name):
    """Given a file name, we parse the file to return relevant stuff for legends, etc."""
    name_split = name.split('_')
    assert name_split[0] == 'grasp' and name_split[1] == 'net', "name_split: {}".format(name_split)
    assert name_split[2] == 'depth' and name_split[4] == 'optim', "name_split: {}".format(name_split)
    assert name_split[6] == 'fixed' and name_split[8] == 'lrate', "name_split: {}".format(name_split)
    info = {'depth': name_split[3],
            'optim': (name_split[5]).lower(),
            'fixed': name_split[7],
            'lrate': (name_split[9]).rstrip('.p')}
    # Actually might as well do the legend here. Start with depth vs rgb, etc.
    if info['depth'] == 'True':
        legend = 'depth-'
    else:
        assert info['depth'] == 'False'
        legend = 'rgb-'
    if info['fixed'] == 'True':
        legend += 'fix26-'
    else:
        assert info['fixed'] == 'False'
        legend += 'train26-'
    legend += 'opt-{}-lr-{}'.format(info['optim'], info['lrate'])
    return info, legend


def plot(head, figname):
    # https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots.html
    nrows, ncols = 1, 2
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10*ncols, 8*nrows),
            squeeze=False, sharex=True, sharey=True)

    # Load and parse information.
    logs = sorted([x for x in os.listdir(head) if x[-2:]=='.p'])
    print("Loading these logs:")

    for pickle_file_name in logs:
        _, legend_l = parse_name(pickle_file_name)
        print("{}  (legend: {})".format(pickle_file_name, legend_l))

        # Should have keys: ['test', 'epoch', 'train', 'raw_test', 'name'].
        data = pickle.load( open(os.path.join(head,pickle_file_name)) )
        epochs = data['epoch']
        K = len(data['test'])
        assert K == len(data['raw_test'])
        factor = float(epochs) / K
        print("epochs: {}, len(data['test']): {}".format(epochs, K))
        xcoord = np.arange(K) * factor # I actually kept multiple values per epoch.

        # Add min value to legend label
        legend_l += '-min-{:.4f}'.format(np.min(data['test']))

        # For plotting test loss, I actually kept multiple values per epoch. Just take the min.
        cc = 0
        if 'depth-' in legend_l:
            cc = 1
        #axes[0,cc].plot(xcoord, data['raw_test'], lw=lw, label=legend_l) # raw pixels
        axes[0,cc].plot(xcoord, data['test'], lw=lw, label=legend_l) # Or scaled

    axes[0,0].set_title('Test Losses, RGB Images Only', size=title_size)
    axes[0,1].set_title('Test Losses, Depth Images Only', size=title_size)

    # Bells and whistles.
    for rr in range(nrows):
        for cc in range(ncols):
            axes[rr,cc].tick_params(axis='x', labelsize=tick_size)
            axes[rr,cc].tick_params(axis='y', labelsize=tick_size)
            axes[rr,cc].legend(loc="best", prop={'size':legend_size})
            axes[rr,cc].set_xlabel('Training Epochs', size=xsize)
            axes[rr,cc].set_ylabel('Test Loss (Raw, L2 Pixels)', size=ysize)
            axes[rr,cc].set_yscale('log')
            axes[rr,cc].set_ylim([0.003,0.1])
    fig.tight_layout()
    fig.savefig(figname)
    print("\nJust saved: {}\n".format(figname))




ROLLOUTS = [34, 7, 39, 37, 46]
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


# TODO: try cross rather than circle?
INNER = 4
OUTER = 5
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
        # But be careful that it's consistent with what we used, e.g., the distance cutoff.
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
        cv2.circle(img=c_img, center=pose_int, radius=INNER, color=(0,0,255), thickness=-1)
        cv2.circle(img=d_img, center=pose_int, radius=INNER, color=(0,0,255), thickness=-1)
        cv2.circle(img=c_img, center=pose_int, radius=OUTER, color=(0,0,0), thickness=2)
        cv2.circle(img=d_img, center=pose_int, radius=OUTER, color=(0,0,0), thickness=2)

        # The PREDICTION, though, will be a large blue circle (yellow border?).
        preds = (int(PREDS[idx][0]), int(PREDS[idx][1]))
        cv2.circle(img=c_img, center=preds, radius=INNER, color=(255,0,0), thickness=-1)
        cv2.circle(img=d_img, center=preds, radius=INNER, color=(255,0,0), thickness=-1)
        cv2.circle(img=c_img, center=preds, radius=OUTER, color=(0,255,0), thickness=2)
        cv2.circle(img=d_img, center=preds, radius=OUTER, color=(0,255,0), thickness=2)

        cv2.imwrite(c_path, c_img)
        cv2.imwrite(d_path, d_img)
        g_in_rollout += 1
        idx += 1

    print("=====================================================================")
