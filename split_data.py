import os

import splitfolders

def split_data(input_folder, output_folder, training_split, validation_split, test_split):
    if not os.path.exists(input_folder):
        raise FileNotFoundError
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    splitfolders.ratio(
        input_folder,
        output_folder,
        seed=0,
        ratio=(training_split, validation_split, test_split),
        group_prefix=None,
        move=False,
    )