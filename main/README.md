# Scripts

- Use `collect_data_bed_fast.py` for collecting data.

- Use `deploy.py` for deploying anything. We have arguments which can specify
  the use of the human supervisor, the grasping network, or analytic grasping
  policies.

Notes:

- Images should be saved via the top-down view from the webcam.

- When running the deployment script, first run with `phase` set to 1 so we can
  see the frames. Then, toss the sheet, and call the numpy random number
  generator to generate `u`, a uniform random variable, such that:

  ```
  human    if u in [0, 1/3)
  analytic if u in [1/3, 2/3)
  network  if u in [2/3, 1]
  ```

  that way we avoid an obvious source of bias.

  When we do the analytic one, we should also count how many attempts had the
  grasp point (i.e., highest point) on the opposite side of the bed, since we do
  _not_ use a similar cutoff for the neural networks, so we are strengthening
  the baseline.
