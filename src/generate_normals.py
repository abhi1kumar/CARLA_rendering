"""
    Sample Run:
    python .py
"""
import os, sys
sys.path.append(os.getcwd())

import numpy as np
import torch
import torch.nn as nn

np.set_printoptions   (precision= 2, suppress= True)
torch.set_printoptions(precision= 2, sci_mode= False)

from plot.common_operations import *
import plot.plotting_params as params
import matplotlib

import logging
logging.basicConfig(filename="test/log.txt", level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))