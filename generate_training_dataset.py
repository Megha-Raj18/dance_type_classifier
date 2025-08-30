import os
import pickle
import numpy as np
import torch

split_file_path = "aist_plusplus_final\\splits\\pose_train.txt"
keypoints_path = "aist_plusplus_final\\keypoints3d"
motions_path = "aist_plusplus_final\\motions"
num_frames = 300
output_path = "training.pt"


def load_split(split_file_path):
    with open(split_file_path, "r") as f:
        return [line.strip() for line in f]
    
def extract_genre_from_seqid(seq_id: str) -> str:
    # AIST++ seq_id format: gXX_sXX_cXX_dXX_mXX_chXX
    # after g is the genre code
    return seq_id.split("_")[0][1:]  
    
def process_sequence(seq_id, keypoints_path, motions_path, num_frames, use_optim = True):

    # get data from motion pkl files
    motion_file = os.path.join(motions_path, f"{seq_id}.pkl")

    # for later if i plan to train on mesh params as well
    # with open(motion_file, "rb") as f:
    #     motion_info = pickle.load(f)
    
    genre = extract_genre_from_seqid(seq_id) #label for training

    #get 3d keypoints
    key_file = os.path.join(keypoints_path, f"{seq_id}.pkl")
    with open(key_file, "rb") as f:
        pose_info = pickle.load(f)
    
    key = "keypoints3d_optim" if use_optim else "keypoints3d"
    pose_data = pose_info[key]

    #preprocessing the pose data

    #normalize the skeleton to make pelvis at origin
    pelvis = 0.5 * (pose_data[:, 11, :] + pose_data[:, 12, :])
    pelvis = pelvis[:, None, :]
    pose_data = pose_data - pelvis
    pose_data = pose_data / np.linalg.norm(pose_data, axis = -1, keepdims = True).max()

    #sequence length
    if pose_data.shape[0] > num_frames:
        pose_data = pose_data[ : num_frames]
    else:
        pad = num_frames - pose_data.shape[0]
        pose_data = np.pad(pose_data, ((0, pad), (0,0), (0,0)), mode = "constant")
    
    return pose_data, genre

def main():
    seq_ids = load_split(split_file_path)
    print(f"Loaded {len(seq_ids)} training sequences")

    data = []
    labels = []
    genres = []

    for i, seq_id in enumerate(seq_ids):
        pose_data, genre = process_sequence(seq_id, keypoints_path, motions_path, num_frames)
        data.append(pose_data)
        labels.append(genre)
        genres.append(genre)

        print(f"Processed {i + 1}/{len(seq_ids)} : {seq_id}, genre={genre}, shape={pose_data.shape}")
        unique_genres = sorted(set(genres))
        genre_to_index = {genre: i for i , genre in enumerate(unique_genres)}
        numeric_labels = [genre_to_index[genre] for genre in labels]

        data_tensor = torch.tensor(np.stack(data), dtype = torch.float32)
        labels_tensor = torch.tensor(numeric_labels, dtype = torch.long)

        torch.save({
            "data" : data_tensor,
            "labels" : labels_tensor,
            "genre_to_index" : genre_to_index
        }, output_path)

        print("Saved dataset")
        print(f"Genres: {genre_to_index}")
if __name__ == "__main__":
    main()