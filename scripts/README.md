# Scripts

For these, run from the top-level repository directory, i.e.: `python scripts/check_raw_data.py`,
rather than `python check_raw_data.py`.

You also may need appropriate python packages. I use a virtual environment with `requirements.txt`
outlined in this GitHub repository.

- `check_raw_data.py`: for checking the data after we collect rollouts.
- `check_raw_data_plot.py`: after running the prior script, use this to plot in case it's easier to
  code this together.
- `data_augmentation_example.py`: for inspecting data augmentation. Works for both RGB and depth
  images.
- `label_bed_fast.py`: for labeling the data if it was collected using the 'faster' way as shown by
  Michael, where he simulates the data.
- `plot_exploratory_training.py`: after trying a bunch of stuff, collect logs into plots to check
  which hyperparameters and settings work the best.
- `nan_check.py`: to check if NaNs exist in the depth image data.
- `overlay_predictions_target.py`: for visualizing the target and predicted grasp point on an image.
  Try to stick to these labeling characterizations.
- `analytic_viz.py`: check how analytic baselines work on images, e.g., using corner detection.
  Ideally, **run this right after data collection**, to instantly see how analytic and heuristic
  methods work and to see how much room we have for improvement.


Ron's scripts for data extraction:

- `extractData.py`
- `getBlanketImages.py`

To visualize these, use `check_ron_data.py`. To format these in a way that fits the training code
with minimal changes, use `format_ron_data.py`.
