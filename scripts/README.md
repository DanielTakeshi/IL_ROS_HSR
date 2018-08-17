# Scripts

For these, run from the top-level repository directory, i.e.: `python scripts/check_raw_data.py`,
rather than `python check_raw_data.py`.  You also may need appropriate python packages. I use a
virtual environment with `requirements.txt` outlined in this GitHub repository.

Data labeling:

- `label_bed_fast.py`: for labeling the data if it was collected using the 'faster' way as shown by
  Michael, where he simulates the data.

Data formatting:

- `convert_to_list_cache.py`: for making the cache. **We should try and always cache our data**, to
  turn it from separate rollout directories into a series of 10 lists, for each CV fold.

Visualize the data and do quick checks:

- `check_raw_data.py`: for checking the data after we collect rollouts.
- `data_augmentation_example.py`: for inspecting data augmentation.
- `nan_check.py`: to check if NaNs exist in the depth image data. I process it out but H does not.
- `analytic_viz.py`: check how analytic baselines work on images, e.g., using corner detection.
  Ideally, **run this right after data collection**, to instantly see how analytic and heuristic
  methods work and to see how much room we have for improvement.

Understanding grasping network performance after training:

- **The main thing we use for evaluating grasping performance**, with self-explanatory names.
    - `inspect_results_cache.py`
    - `inspect_results_nocache.py` (edit: deprecated)
- Can also run `stitch_results.py` for stitching together different dictionaries together in an
  ad-hoc way, e.g., net 1 and net 3 comparisons.
- `scripts/overlay_auto_preds_targ.py`: do this automatically, rather than copy/paste. This also
  reports lots of other interesting statistics, and creates heat maps. It's deprecated by the
  `inspect_results.py` script.

## Other / Old

Ron's scripts for data extraction (I have a few for visualizing his as well):

- `extractData.py`
- `getBlanketImages.py`

To visualize these, use `check_ron_data.py`. To format these in a way that fits the training code
with minimal changes, use `format_ron_data.py`. Use `plot_training_ron.py` to investigate the
results from training on Ron's data using cross validation.
