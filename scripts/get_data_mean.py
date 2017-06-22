#get mean in each channel
import os
from il_ros_hsr.p_pi.safe_corl.vgg_options import VGG_Options as options

Options = options()

f = []

for (dirpath, dirnames, filenames) in os.walk(Options.rollouts_dir):
    f.extend(dirnames)

means = np.array([0, 0, 0])
num_imgs = 0

for filename in f:
    rollout_data = pickle.load(open(Options.rollouts_dir+filename+'/rollout.p','r'))

    #compute a rolling average (don't have to save overall total)
    for img in rollout_data:
        img_mean = np.mean(img)
        means = means * 1.0 * (num_imgs)/(num_imgs + 1) + (img_mean) * 1.0/(num_imgs + 1)
        num_imgs += 1

print(means)
