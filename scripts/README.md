# Scripts

For these, run from the top-level repository directory, i.e.: `python scripts/check_raw_data.py`,
rather than `python check_raw_data.py`.  You also may need appropriate python packages. I use a
virtual environment with `requirements.txt` outlined in this GitHub repository.

Data labeling:

- `label_bed_fast.py`: for labeling the data if it was collected using the 'faster' way as shown by
  Michael, where he simulates the data.

Visualize the data and do quick checks:

- `check_raw_data.py`: for checking the data after we collect rollouts.
- `check_raw_data_plot.py`: after running the prior script, use this to plot in case it's easier to
  code this together.
- `data_augmentation_example.py`: for inspecting data augmentation. Works for both RGB and depth
  images.
- `nan_check.py`: to check if NaNs exist in the depth image data.
- `analytic_viz.py`: check how analytic baselines work on images, e.g., using corner detection.
  Ideally, **run this right after data collection**, to instantly see how analytic and heuristic
  methods work and to see how much room we have for improvement.

Understanding grasping network performance:

- `overlay_predictions_target.py`: for visualizing the target and predicted grasp point on an image.
  Try to stick to these labeling characterizations. I also have one for Ron's data.
- `scripts/overlay_auto_preds_targ.py`: do this automatically, rather than copy/paste. **This also
  reports lots of other interesting statistics, and creates heat maps. Should see if there's ways we
  can make this better and easier to use.**

Understanding training:

- `plot_exploratory_training.py`: after trying a bunch of stuff, collect logs into plots to check
  which hyperparameters and settings work the best.

Ron's scripts for data extraction (I have a few for visualizing his as well):

- `extractData.py`
- `getBlanketImages.py`

To visualize these, use `check_ron_data.py`. To format these in a way that fits the training code
with minimal changes, use `format_ron_data.py`. Use `plot_training_ron.py` to investigate the
results from training on Ron's data using cross validation.
