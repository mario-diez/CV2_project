import glob
from sklearn.model_selection import train_test_split
import os
import shutil

shot_path = './BARD/data/shot/'
no_shot_path = './BARD/data/no_shot/'

shot_list = glob.glob(shot_path + '*.mp4')
no_shot_list = glob.glob(no_shot_path + '*.mp4')

import random

def split_data(data_list, max_train=39, seed=42):
    random.seed(seed)

    data_list = data_list.copy()
    random.shuffle(data_list)

    train = data_list[:max_train]
    remaining = data_list[max_train:]

    mid = len(remaining) // 2
    val = remaining[:mid]
    test = remaining[mid:]

    return train, val, test


def copy_files(file_list, destination_folder):
    os.makedirs(destination_folder, exist_ok=True)

    for file_path in file_list:
        file_name = os.path.basename(file_path)
        dest_path = os.path.join(destination_folder, file_name)
        shutil.copy(file_path, dest_path)

    print(f"Successfully copied {len(file_list)} files to {destination_folder}")


shot_train, shot_val, shot_test = split_data(shot_list)
no_shot_train, no_shot_val, no_shot_test = split_data(no_shot_list)

print(
    f"Shot -> train:{len(shot_train)}, val:{len(shot_val)}, test:{len(shot_test)}"
)
print(
    f"No shot -> train:{len(no_shot_train)}, val:{len(no_shot_val)}, test:{len(no_shot_test)}"
)

copy_files(shot_train, './data/train/shot')
copy_files(shot_val, './data/val/shot')
copy_files(shot_test, './data/test/shot')

copy_files(no_shot_train, './data/train/no_shot')
copy_files(no_shot_val, './data/val/no_shot')
copy_files(no_shot_test, './data/test/no_shot')