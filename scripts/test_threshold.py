"""
Test image thresholds, to see if we can add offsets.
Use this for inspecting saved data to determine the thresholds.

Easiest way is to compare the image the grasp net saw, with the image the
success net sees, since that way we compare what directly happens next.
"""
import os, cv2, sys, pickle, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
import numpy as np
from os.path import join
from fast_grasp_detect.data_aug.depth_preprocess import depth_to_net_dim
from skimage.measure import compare_ssim

HEAD = '/nfs/diskstation/seita/bed-make/results'
TEST_HEAD = 'test_imgs'

DEPLOY_A = join(HEAD, 'deploy_analytic')
DEPLOY_H = join(HEAD, 'deploy_human')
DEPLOY_N = join(HEAD, 'deploy_network_white')

RESULTS_A = sorted([join(DEPLOY_A,x) for x in os.listdir(DEPLOY_A) if 'results' in x])
RESULTS_H = sorted([join(DEPLOY_H,x) for x in os.listdir(DEPLOY_H) if 'results' in x])
RESULTS_N = sorted([join(DEPLOY_N,x) for x in os.listdir(DEPLOY_N) if 'results' in x])


def save_imgs(RES, pref):
    """I'm thinking we want to check the grasp img w/the subsequent success img.
    """
    diffs = []
    ssims = []

    for ridx,res in enumerate(RES):
        fidx = 0
        with open(res, 'r') as f:
            data = pickle.load(f)
        num_grasp = int( len(data) / 2.0 )
        print(res, num_grasp)
        previous_grasp_img = None

        for item in data:
            if 'type' not in item:
                continue
            dimg = item['d_img']
            if np.isnan(np.sum(dimg)):
                cv2.patchNaNs(dimg, 0.0)
            dimg = depth_to_net_dim(dimg, robot='HSR')

            # Record current grasp img so we can compare w/next success net img.
            which_net = item['type'].lower()
            if which_net == 'grasp':
                previous_grasp_img = dimg
            else:
                diff = np.linalg.norm( previous_grasp_img - dimg )
                diffs.append(diff)
                gray1 = previous_grasp_img[:,:,0]
                gray2 = dimg[:,:,0]
                assert gray1.shape == gray2.shape == (480,640)

                # we can optinally return other stuff
                # http://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.compare_ssim
                #(score, difference) = compare_ssim(gray1, gray2)
                score = compare_ssim(gray1, gray2)
                ssims.append(score)

            # Save image, put L2 in success net img name
            side = item['side'].lower()
            fname = '{}_roll_{}_side_{}_idx_{}_{}.png'.format(pref, ridx, side, fidx, which_net)
            fname = join(TEST_HEAD, fname)
            if which_net == 'success':
                fname = fname.replace('.png', '_{:.3f}_{:.3f}.png'.format(diff, score))
                fidx += 1
            cv2.imwrite(fname, dimg)
    return diffs, ssims


def collect_consecutives():
    """Try to get a list of consecutive images.
    """
    print("For length, subtract one, then divide by two.")
    print("That's how many images the grasp network saw.")
    print("The number will be in {2, 3, 4, 5, 6, 7, 8}.")
    print("\non analytic rollouts:")
    diffs1, ssims1 = save_imgs(RESULTS_A, 'ana')
    print("\non human rollouts:")
    diffs2, ssims2 = save_imgs(RESULTS_H, 'hum')
    print("\non network (w/white) rollouts:")
    diffs3, ssims3 = save_imgs(RESULTS_N, 'net_w')

    all_d = diffs1 + diffs2 + diffs3
    all_s = ssims1 + ssims2 + ssims3

    nrows, ncols = 1, 2
    fig, ax = plt.subplots(nrows, ncols, figsize=(10*ncols,8*nrows))
    ax[0].hist(all_d, bins=50)
    ax[1].hist(all_s, bins=50)
    plt.savefig('histogram.png')


if __name__ == "__main__":
    collect_consecutives()
