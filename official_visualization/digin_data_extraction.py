#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 12:20:05 2024

@author: emmabarash
"""

import seaborn as sns
import pandas as pd
import argparse
import glob
import os

import easygui
import numpy as np
import pylab as plt
from tqdm import tqdm

# Get name of directory with the data files
# Create argument parser
# parser = argparse.ArgumentParser(description = 'Plots DIG_INs and AMP files')
# parser.add_argument('dir_name',  help = 'Directory containing data files')
# args = parser.parse_args()

# if args.dir_name:
#     dir_path = args.dir_name
#     if dir_path[-1] != '/':
#         dir_path += '/'
# else:
#     dir_path = easygui.diropenbox(msg = 'Please select data directory')
dir_path = '/Users/emmabarash/Desktop/emma_behavioral_dat/eb20_behavior_4_30_240430_113352'
# Create plot dir
plot_dir = os.path.join(dir_path, "channel_profile_plots")
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# Get files to read

#HANNAH CHANGE: ADDED TEST OF ONE FILE PER SIGNAL TYPE
print("Testing File Type")
file_list = os.listdir(dir_path)
try:
	file_list.index('auxiliary.dat')
	file_type = ['one file per signal type']
except:
	file_type = ['one file per channel']
print("\tFile Type = " + file_type[0])

if file_type == ['one file per channel']:
	amp_files = glob.glob(os.path.join(dir_path, "amp*dat"))
	amp_files = sorted(amp_files)
	digin_files = sorted(glob.glob(os.path.join(dir_path, "board-DI*")))
elif file_type == ['one file per signal type']:
	amp_files = glob.glob(os.path.join(dir_path, "amp*dat"))
	digin_files = glob.glob(os.path.join(dir_path, "dig*dat"))
	#Use info file for port list calculation
	info_file = np.fromfile(dir_path + 'info.rhd', dtype = np.dtype('float32'))
	sampling_rate = int(info_file[2])
	# Read the time.dat file for use in separating out the one file per signal type data
	num_recorded_samples = len(np.fromfile(dir_path + '/' + 'time.dat', dtype = np.dtype('float32')))
	total_recording_time = num_recorded_samples/sampling_rate #In seconds


downsample = 100
	
print("Now plotting digital input signals")
#digin_files = digin_files[7:]
if file_type == ['one file per channel']:
	fig,ax = plt.subplots(len(digin_files),
	        sharex=True, sharey=True, figsize = (8,10))
	for this_file, this_ax in tqdm(zip(digin_files, ax.flatten())):
	    data = np.fromfile(this_file, dtype = np.dtype('uint16'))
	    this_ax.plot(data[::downsample])
	    this_ax.set_ylabel("_".join(os.path.basename(this_file)\
	            .split('.')[0].split('-')[1:]))
	plt.suptitle('DIGIN Data')
	fig.savefig(os.path.join(plot_dir, 'digin_data'))
	#plt.close(fig)
elif file_type == ['one file per signal type']:
	d_inputs = np.fromfile(digin_files[0], dtype=np.dtype('uint16'))
	d_inputs_str = d_inputs.astype('str')
	d_in_str_int = d_inputs_str.astype('int64')
	d_diff = np.diff(d_in_str_int)
	dig_in = list(np.unique(np.abs(d_diff)) - 1)
	dig_in.remove(-1)
	num_dig_ins = len(dig_in)
	dig_inputs = np.zeros((num_dig_ins,len(d_inputs)))
	for n_i in range(num_dig_ins):
		start_ind = np.where(d_diff == n_i + 1)[0]
		end_ind = np.where(d_diff == -1*(n_i + 1))[0]
		for s_i in range(len(start_ind)):
			dig_inputs[n_i,start_ind[s_i]:end_ind[s_i]] = 1
	fig,ax = plt.subplots(num_dig_ins,1,
	        sharex=True, sharey=True, figsize = (8,10))
	for d_i in tqdm(range(num_dig_ins)):
	    ax_i = plt.subplot(num_dig_ins,1,d_i+1)
	    ax_i.plot(dig_inputs[d_i,::downsample])
	    ax_i.set_ylabel('Dig_in_' + str(dig_in[d_i]))
	plt.suptitle('DIGIN Data')
	fig.savefig(os.path.join(plot_dir, 'digin_data'))
	#plt.close(fig)
    