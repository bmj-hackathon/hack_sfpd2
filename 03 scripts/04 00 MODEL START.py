# =============================================================================
# Standard imports
# =============================================================================
import os
import logging
import datetime

# =============================================================================
# External imports - reimported for code completion! 
# =============================================================================
print_imports()

# Import again for code completion!
import pandas as pd 
import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn as sk
import sklearn
import sklearn.linear_model
import sklearn.cross_validation
import sklearn.kernel_approximation
import sklearn.linear_model
import sklearn.grid_search 
#import pickle
#sklearn.__version__
from sklearn_pandas import DataFrameMapper
import sklearn.externals
# to make this notebook's output stable across runs
np.random.seed(42)

# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
warnings.filterwarnings(action="ignore", module="pandas", message="^internal gelsd")
warnings.simplefilter(action='ignore', category=FutureWarning)
#warnings.simplefilter(action='ignore', category=warnings)

#%% ENSURE CLEAN STATE
print("******************************")
del_vars =[
        "clf",
        "param_grid",
        "clf_grid",
        ]
cnt = 0
for name in dir():
    if name in del_vars:
        cnt+=1
        del globals()[name]
logging.info(f"Removed {cnt} variables from memory")
del cnt, name, del_vars
