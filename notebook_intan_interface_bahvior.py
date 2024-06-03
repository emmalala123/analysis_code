#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 13:04:21 2024

@author: emmabarash
"""

# Load the background Python functions that allow for data loading and plotting
import notebook_intan_bahavior as ut
import glob
import os

path = '/Users/emmabarash/Desktop/emma_behavioral_dat/eb18_behavior_2_26_240226_124852'
os.chdir(path)
print(os.getcwd())

# import many
def concatenate_rhd_files(output_file_path, input_pattern):
    # Find all .rhd files matching the input pattern
    file_paths = glob.glob(input_pattern)
    file_paths.sort()  # Ensure the files are in the correct order

    # Open the output file in binary write mode
    with open(output_file_path, 'wb') as outfile:
        for file_path in file_paths:
            # Open each .rhd file in binary read mode
            with open(file_path, 'rb') as infile:
                # Read the content of the file and write it to the output file
                outfile.write(infile.read())

# Example usage
concat_file = concatenate_rhd_files('concatenated_output.rhd', '*.rhd')

# import one
# filename = 'eb18_behavior_2_26_240226_131452.rhd' 
# Change this variable to load a different data file
result, data_present = ut.load_file(concat_file)

ut.print_all_channel_names(result)
print('result', result)
channel_name = 'DIGITAL-IN-08' # Change this variable and re-run cell to plot a different channel


if data_present:
    ut.plot_channel(channel_name, result)
    
else:
    print('Plotting not possible; no data in this file')