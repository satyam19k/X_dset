# Single_data_collect.py

- **record_simulation**
  - Records simulation actions and states.
  - Saves the simulation as a video.
  - Generates `states.txt` capturing the recorded states.

- **create_video_from_actions**
  - Reads the saved `action.pth` file and the initial state.
  - Generates a full rollout video based on recorded actions.
  - Produces a `states.txt` corresponding to the action-based video.

- **create_video_from_states**
  - Reads the saved `state.pth` file.
  - Generates a video using the recorded states.
  - Produces a `states.txt` corresponding to the state-based video.

- **Sanity Check Process**
  - Combine the videos from `record_simulation`, `create_video_from_actions`, and `create_video_from_states` into a single video for visual comparison.
  - Save and compare the `states.txt` files from each method using a diff tool to ensure exact state matching.


# Continuous_data_collect

    - collects the data in continuous fashion.

# X_dset
  - obses (recorded observations)
  - states.pth (combined states for all 50 states and padded accordingly by max of seq_len)
  - actions.pth (combined actions for all 50 states and padded accordingly by max of seq_len)
  - seq_len.pkl (seq length of each data_point)
  - shapes.pkl (shape of each data_point)
  - sanity_check.py (sanity check for obtained rollout videos vs recorded obs)