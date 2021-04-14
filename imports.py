if 'get_ipython' in locals():
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')

import os, sys
import operator
import math

import numpy as np
from numpy.random import rand, randint, randn, normal
from numpy import zeros, ones, empty, array

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.distributions.categorical import Categorical
from torch.distributions.utils import probs_to_logits
from torch import tensor, as_tensor, from_numpy


from tqdm import tqdm
from copy import deepcopy
from datetime import datetime
from time import time, sleep
import pickle

from matplotlib import pyplot
import matplotlib
matplotlib.rcParams["font.family"] = "Fira Code"
matplotlib.rcParams["xtick.labelsize"] = 14
matplotlib.rcParams["ytick.labelsize"] = 14
matplotlib.rcParams["figure.titlesize"] = 12
matplotlib.rcParams["figure.figsize"] = 9, 5
matplotlib.rcParams["figure.dpi"] = 40
matplotlib.style.use('ggplot')

th.set_printoptions(sci_mode=False)
np.set_printoptions(formatter={'all': lambda x: str(x)})
