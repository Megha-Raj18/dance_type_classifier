# dance_type_classifier

Main Idea:

use pose extracton (3D keypoints, motion sequences) info from AIST++ dataset https://aistdancedb.ongaaccel.jp/ from 10 genres of dances to train and build a model from scratch that predicts the genre of a dance from a given video input

Data Preprocessing:

- loads 3D keypoint info from pickle files
- extract genre from file name (file is of the form gXX_sXX_cXX_dXX_mXX_chXX where gXX represents the genre)
- normalize skeletons from motion sequences by centering the pelvis at the origin and scaling using maximum joint distance
- samples each sequence to a fixed length of 300 frames (first 300 frames are used for now)
- outputs training.pt/validation.pt which contains
  - data (shape - [N videos for training, T frames (first 300), J joints (17), C = 3 coordinates (3d)])
  - labels (numeric - {1,10})
  - genre_to_index (mapping from string genre to numeric labels)
 
Model training:

- used LSTM [for now] (since we're dealing with sequential data)
- dataset wrapper to read .pt files and extract data, also flattens the 3D joint coordinates to get a single vector per frame ([N, T, J, C] -> [N, T, J * C])
- Model (currently):
  - 2 level LSTM with hidden size = 256, dropout = 0.3
  - temporal pooling over frames
- runs for 20 epochs
- loss calculated using crossentropyloss

Initial run:

stored in training_outputs/temporal_pooling/train_output_first_300_frames.txt
after 20 epochs, we see around 60% for both training and validation accuracy

Epoch 19: 
Training Loss: 1.1361, Training Accuracy: 60.37%
Validation Loss: 1.1503, Validation Accuracy: 61.43%


