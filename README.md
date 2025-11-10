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

Run 2:

stored in training_outputs/temporal_pooling/train_output_first_300_frames_(2).txt
we see that increasing the number of epochs from 20 to 30 substantially improves the training and validation accuracy, more epochs allows the model to really fit into the data

Epoch 29: 
Training Loss: 0.4258, Training Accuracy: 85.60%
Validation Loss: 0.4503, Validation Accuracy: 87.14%

Run 3:

stored in training_outputs/temporal_pooling/train_output_every_3rd_frame.txt
instead of using the first 300 frames of the motion sequence, I decided to try using every 3rd (or so) frame from each sequence using linspace which improved the train/val accuracy by around 4 - 5%. This seems to be a better idea since the model gets enough info on the entire motion sequence without
  a) looking at all 720 frames
  b) capturing miniscule differences between consecutive frames

Epoch 29: 
Training Loss: 0.2972, Training Accuracy: 90.55%
Validation Loss: 0.3046, Validation Accuracy: 91.43%

TESTING:

first test run using every 3rd frame achieved around 73% test accuracy and we can see from the confusion matrix alot of classes do get mislabelled. This could be because of 
- two types of dances having very similar poses (middle hip hop [MH] and LA style hip hop [LH], JB and WA - similar loops formed by arms??) - main reason probably
- minor signature moves of each dance type being overshadowed by taking an average (temporal pooling)
- not enough data to capture signature steps/moves

  <img width="519" height="440" alt="image" src="https://github.com/user-attachments/assets/3d901c32-aa12-4f1b-a47c-cb149112a02b" />


Note: Attempt at adding attention

stored in training_outputs/attention/train_output_every_3rd_frame.txt
adding attention actually made the accuracy worse which could be due to the following reasons:
  a) adding attention adds more parameters for the model to learn, but with a relatively small dataset (868 motion sequences) it may lead to overfitting and just poor model performance in general
  b) attention usually focuses specific frames that the model believes are important to the classification, but working with dance data its better to use an overall look at all the frames rather than a certain subset of them

Epoch 29: 
Training Loss: 1.3194, Training Accuracy: 52.30%
Validation Loss: 1.2696, Validation Accuracy: 55.71%








