import splitfolders
import os

input_path = '../preprocessed_data'
output_path = '../split_data'

splitfolders.ratio(input_path, output=output_path, seed=18, ratio=(0.80, 0.10, 0.10))