# IL_ROS_HSR

A repo for performing imitation learning experiments on a ROS-compatible Toyota
HSR

Daniel Seita's fork of this repository for finishing up the bed-making project,
but it's also flexible enough to deal with other projects.

Currently, it is working to do some of the bed making rollouts, but need to
double check that all the paths are referring to my stuff instead of Michael's
package.

# Bed-Making Instructions

Here are full instructions for the bed-making project.


## Preliminaries and Setup

1. Install this and the [fast_grasp_detect repository][2]:

    - Use `python setup.py develop` in the two repositories.
    - Install TensorFlow 1.4 or 1.5.
    - Install the TMC code library, which includes `hsrb_interface`.
    - TODO: need to get a requirements text somewhere.
    - Adjust and double-check the [configuration file][1] and other paths, to
      ensure that you're referring to correct workspaces. TODO: need to get this
      in the configuration rather than `sys.path.append(...)` calls in the code.
    - Also double check that the overall data directory (by default
      `/media/autolab/1tb/[something...]`) is mounted and accessible for data
      writing.
    - Make sure the HSR is charged, but that the charging tube is disconnected.

2. Make sure an AR maker is taped on the ground, [and that it is AR maker
11][3], and that the robot is in a good starting position by using the joystick,
so that it can see the AR marker. **For these steps, be sure you are in HSRB
mode (`export ROS_MASTER_URI ...`) and in the correct python virtual
environment!**

    - Run `python scripts/joystick_X.py` first and then move the robot to the
      designated starting position. (It should be marked with tape ... put tape
      on if it isn't!)
    - Turn off the joystick script.
    - In another tab, run `rosrun rviz rviz`.
    - Get the bed setup by putting the bed towards where the rviz markers are
      located. Just a coarse match is expected and sufficient. 

      To be clear:
      - Match the following frames: `head up`, `head down`, `bottom up`, and
        `bottom down`.
      - The centers should be fixed, but the orientation can be a bit confusing
        ... for now I'll keep it the way it is. At least the blue axis (z axis I
        think) should point *downwards*. 
      - In order to do this step, you need to run `python
        main/collect_data_bed.py` first to get the frames to appear,
        but hopefully after this, no changes to the bed positioning are needed.
      - If you run this the first time, it is likely you will need to reposition
        the robot so that the AR marker is visible. Use rviz for visualizing,
        but again, this requires the data collection script to be run for the AR
        maker/11 frame to appear.

    - The easiest way to do the last step above is by running `python
      main/collect_data_bed.py` --- which we have to do anyway for the data
      collection --- and adjusting at the beginning.

## Data Collection

1. As mentioned above, run `python main/collect_data_bed.py`. Make sure there
are no error messages. 

    - If there are no topics found initially, that likely means the AR marker is
      not visible. Please check rviz.
    - For each setup, we sample a starting bed configuration with a red line and
      then have to *physically adjust the bed sheet to match that image*. This
      provides the variation in starting states. Again, only coarse matches are
      needed.

2. After the sheet has been adjusted, the robot can move.

    - Press B on the joystick. Then the robot should move up to the bed, and
      pause.
    - An image loader/viewer pops up. Click "Load" (upper right corner) to load
      the current camera image of the robot.
    - Drag the bounding box where the robot should grasp. Click "send command"
      and then close the window. *The grasp will not execute until the window is
      closed*.

3. After the grasp, we need to check for transitioning vs re-grasping.

    - Load the image as usual by clicking "Load". 
    - Below the "Load" button, drag and drop either "Yes" or "No".
    - Draw a bounding box; **TODO: not sure if it matters where the box is if we
      know the robot should transition?**
    - Send the command, close the window.

    **TODO: I can't figure out, I click yes and no both times and it causes the
    robot to move to the other side, never re-grasping. Hmm ...**

4. Data Storage

    - After the HSR has transitioned to the start, it will save the data under
      the specified directory, as `rollout_X.p` where `X` is the index. Check
      the print logs for the location (if it doesn't match what you want, check
      the configuration files again).
    - The `rollout_X.p` is a list of length 5 assuming the robot just executed
      one grasp at the bottom then one grasp at the top. It contains:
        - A list of 3-D points; I think these are poses but not sure.
        - A dictionary of: `c_img`, `d_img`, `class`, `pose`, `side`, and
          `type`. So it tells us information about this particular grasp.
        - Another dictionary with similar keys, this time for success or
          failure of a grasp.
        - And then two more for the top version of these.

5. Other stuff

    - TODO: DART? Faster ways to collect data?
    - How to deal with the transition back to the start?


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
[3]:https://docs.hsr.io/manual_en/development/ar_marker.html
[4]:https://github.com/DanielTakeshi/fast_grasp_detect/commit/424463e12996b037c3f3539e58d1b5572f4ca835
