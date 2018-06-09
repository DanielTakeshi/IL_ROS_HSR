# IL_ROS_HSR

A repo for performing imitation learning experiments on a ROS-compatible Toyota HSR

Daniel Seita's fork of this repository for finishing up the bed-making project,
but it's also flexible enough to deal with other projects.

# Bed-Making Instructions

Here are full yet concise instructions for the bed-making project.


## Preliminaries and Setup

1. Install this and the [fast_grasp_detect repository][2].
2. Adjust and double-check the [configuration file][1].
3. Get the bed setup by putting the bed towards where the rviz markers are
located. Just a coarse match is expected and sufficient.


## Data Collection

1. Run `python collect_data_bed.py`. For each setup, we sample a starting bed
configuration with a red line and then have to physically adjust the bed sheet
to match that image. This provides the variation in starting states.
2. TODO: DART? Faster ways to collect data?


## Neural Network Training

1. Collect the data in an appropriate manner.
2. Run `python train_bed_grasp.py` and `python train_bed_success.py` in the
`fast_grasp_detect` repository.


## Evaluation

1. Run `python deploy_network.py` for testing the method we propose with deep
imitation learning (with DART ideally).
2. Run `python deploy_analytic.py` for testing the baseline method.


[1]:https://github.com/DanielTakeshi/IL_ROS_HSR/blob/master/src/il_ros_hsr/p_pi/bed_making/config_bed.py
[2]:https://github.com/DanielTakeshi/fast_grasp_detect
