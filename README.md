# ConvLSTM Autoencoder for Face PAD Project

Author: Selim Firat Yilmaz

* Done for the CS579 Biometrics course.


## Running Scripts
* `bash run_baselines.sh` makes the baseline experiments.
* `bash run_convlstm.sh` makes hyperparameter search for ConvLSTM Autoencoder.
* `bash run_best_interdb.sh` runs best baselines for interdb settings.
* `bash run_iq` runs the image quality experiments.


# Python Scripts
* `main.py` baseline experiment runner. Runs iForest, OC-SVM, and Autoencoder.
* `convlstm_main` ConvLSTM experiment runner. Runs ConvLSTM Autoencoder.
* `convlstm_cell.py` ConvLSTM Cell Implementation.
* `convlstm_autoencoder` ConvLSTM Autoencoder implementation
* `do_evaluation` Creates some figures, however, I do not use them in final report. Some methods are used by print_best_models.py
* `extract_features.py` Extracts VGG16 and VGGFace features from Replay-Attack and Replay-Mobile databases and saves them as h5 file.
* `extract_openface.py` Extracts face alignment frames from databases via Openface.
* `faces_to_video.py` Converts Openface extracted frames to videos to extract features from.
* `generate_tables.py` Generates baseline tables.
* `normalized_model.py` Apply feature normalization before running iForest, OC-SVM. Used by `main.py`
* `num_frames_hist.py` Generates number of frames histograms in the final report.
* `plot_helpers.py` Provides convenience functions for plotting.
* `print_best_models.py` Chooses best models by Dev EER and prints their scores etc.
* `boxplots.py` creates the boxplots in the final report.
