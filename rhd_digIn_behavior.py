#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 11:57:31 2024

@author: emmabarash

"""
import tables
import os
import numpy as np
import tqdm

def read_digins(hdf5_name, dig_in_int, dig_in_file_list): 
	atom = tables.IntAtom()
	hf5 = tables.open_file(hdf5_name, 'r+')
	# Read digital inputs, and append to the respective hdf5 arrays
	print('Reading dig-ins')
	for i, (dig_int, dig_in_filename) in \
			enumerate(zip(dig_in_int, dig_in_file_list)):
		dig_in_name = f'dig_in_{dig_int:02d}'
		print(f'Reading {dig_in_name}')
		inputs = np.fromfile(dig_in_filename,
					   dtype = np.dtype('uint16'))
		hf5_dig_array = hf5.create_earray('/digital_in', dig_in_name, atom, (0,))
		hf5_dig_array.append(inputs)
		hf5.flush()
	hf5.close()
    
# Use info file for port list calculation
# info_file = np.fromfile(dir_name + '/info.rhd', dtype=np.dtype('float32'))
# sampling_rate = int(info_file[2])