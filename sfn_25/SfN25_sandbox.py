#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 12:28:07 2025

@author: emmabarash
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

# target the directory with the h5, .info, and trial_info files.
path = "/Users/emmabarash/Desktop/ephys_data_sandbox/"

# get the trial info frames into a file list
trail_info_files = glob.glob(os.path.join(path + "**/*.csv"))

# get the h5 files into a file list
h5_files = glob.glob(os.path.join(path + "**/*.h5"))

