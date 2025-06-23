Thesis Code Submission – README
===============================

Project Title:
Anticipating Haptic Greetings from Egocentric Video Using Deep Learning

Author:
Efe Kirisci – Cognitive Science and Artificial Intelligence, Tilburg University, Student Number: 2101524

Purpose:
---------
This folder contains the full implementation of the deep learning pipelines used for the bachelor thesis project. The goal of the project is to anticipate social haptic gestures (e.g., hugs, handshakes) from egocentric video using CNN + LSTM and ViT + LSTM models under different temporal constraints.

Folder Structure:
-----------------
- scripts/
    Contains all core Python scripts used for training, evaluation, data loading, and feature extraction. This is the main codebase.
    
- predictions_test/
    CSV files showing predicted vs. true labels for each test video. Useful for error analysis and per-class performance calculations.

- models/
    Contains `.pt` files of the six best-performing trained models. Each model corresponds to a unique architecture and input length.

- features/
    CNN-extracted features from video frames, used as input to the CNN + LSTM model. (Included)

- vit_features/
    ViT-extracted features from video frames, used as input to the ViT + LSTM model. (Included)

- norm_cmatrices_test/ & norm_cmatrices_val/
    Normalized confusion matrices (PNG) visualizing classification performance per class.

- raw_cmatrices_test/ & raw_cmatrices_val/
    Raw (unnormalized) confusion matrices (PNG) for detailed reference. Mentioned in appendix.

- extra_visualization/
    Includes the per-class F1 score bar chart and other optional illustrative figures.

- data/, processed_data/
    These folders are left empty due to size and NDA restrictions. They originally contained raw and processed video data provided by the thesis coordinator dr. Merel Jung

Notes:
------
- All essential code for training and evaluating models is found in the `scripts/` folder.
- Trained models are ready to use for inference or further fine-tuning.
- Frame extraction, padding, and collate logic are fully implemented.
- Requirements file and virtual environments are not included. The core dependencies are:
    - Python 3.10+
    - PyTorch 2.2.1
    - NumPy
    - matplotlib
    - scikit-learn

Contact:
--------
For further clarification, please contact the author via university channels.