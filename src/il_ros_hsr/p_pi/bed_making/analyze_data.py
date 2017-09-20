import il_ros_hsr.p_pi.bed_making.config_bed as cfg
import cv2
import numpy as np
import cPickle as pickle

def read_rollout(num):
    path = cfg.ROLLOUT_PATH + "rollout_" + str(num) + '/'
       

    data = pickle.load(open(path+'rollout.p','rb'))

    before = data[0]
    after = data[1]

    for dic, prefix in [(before, "before"), (after, "post")]:
        img = dic['c_img']
        warped = dic['out_img']
        res = dic['per']
        pre = "analyze/" + "rollout" + str(num) + "_" + prefix + "_"

        cv2.imwrite(pre + "full.png", img)
        cv2.imwrite(pre + "warped_" + str(res) + ".png", warped)

    final = data[1]["per"]
    former = data[0]["per"]

    return final, former 

avg_final = 0.0
avg_delta = 0.0

for n in range(34, 39):
    final, former = read_rollout(n)
    avg_final += final 
    avg_delta += (final - former)

avg_final = avg_final/5.0
avg_delta = avg_delta/5.0

print(avg_final, avg_delta)