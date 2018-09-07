# Scripts

For these, run from the top-level repository directory, i.e.: `python scripts/check_raw_data.py`,
rather than `python check_raw_data.py`.  You also may need appropriate python packages. I use a
virtual environment with `requirements.txt` outlined in this GitHub repository.

Data labeling:

- `label_bed_fast.py`: for labeling the data if it was collected using the 'faster' way as shown by
  Michael, where he simulates the data.

Data formatting:

- `convert_to_list_cache.py`: for making the cache. **We should try and always cache our data**, to
  turn it from separate rollout directories into a series of 10 lists, for each CV fold. This script
  is flexible to handle different datasets.
- Use `success_list_cache.py` for the equivalent version but for the success network's data.
- `convert_to_list_combo_cache.py`: for combining datasets formed from `convert_to_list_cache`.

Visualize the data and do quick checks:

- `check_raw_data.py`: for checking the data after we collect rollouts.
    - Also `check_other_data.py` for H's data, `check_daniel_data.py` for mine.
- `data_augmentation_example.py`: for inspecting data augmentation.
- `nan_check.py`: to check if NaNs exist in the depth image data. I process it out but H does not.
- `analytic_viz.py`: check how analytic baselines work on images, e.g., using corner detection.
  Ideally, **run this right after data collection**, to instantly see how analytic and heuristic
  methods work and to see how much room we have for improvement.

Understanding grasping network performance after training:

- **The main thing we use for evaluating grasping performance**: `inspect_results_cache.py`.
    - For success net: `success_inspect_results.py`.
- Can also run `stitch_results.py` for stitching together different dictionaries together in an
  ad-hoc way, e.g., net 1 and net 3 comparisons. This will probably be needed for a paper. In
  general, anything starting with `stitch_results` requires manual tuning but is better for a final
  figure to report.

Evaluate bed-making results from deployment:

- Use `bedmake_coverage_{auto,manual}.py`: for _coverage_.
    - Auto: will go through and detect blue automatically. Unfortunately I've found this isn't as
      good as I'd like. It's reasonable for the starting configuration but is vulnerable to lighting
      and thinking the floor is a contour.
    - Manual: we click points to form the contour, then compute area from that. I think that's very
      good, it's like we are tracing out our contours. Seems much better, and if it's 100% we
      can just say that.
    - Regardless, we have to click to form the table top. For saving, save the raw images AND the
      image with the contours for visualization.  Keep the coverage percentage in the file name so
      it's easier to inspect.
- Use `bedmake_results.py`: for anything _but_ coverage.
- Use `bedmake_box_plots.py` for the actual bar or box plots. (I use bar for now.)

In order to convert H's data so that the coverage script (and thus the box
plot script) works, please run `convert_h_rollouts_to_mine.py`.

To further investigate, e.g., to get images of the policy and actual
predictions, please run `check_rollout_results.py`. So, directory structure:

```
results/
    # contains all of the deployment scripts
    figures/
        # contains all COVERAGE images, so we use this to show contours
        # it is used for the box plot computations
results_all_figs/
    # use this for all OTHER images, e.g. to see the c_img and d_img in the
    # actual rollouts.
results_honda/
    # what they give me. I then convert to my format and put in `results`
```


## Other / Old

Ron's scripts for data extraction (I have a few for visualizing his as well):

- `extractData.py`
- `getBlanketImages.py`

To visualize these, use `check_ron_data.py`. To format these in a way that fits the training code
with minimal changes, use `format_ron_data.py`. Use `plot_training_ron.py` to investigate the
results from training on Ron's data using cross validation.
