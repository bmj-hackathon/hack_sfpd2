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

#sklearn.__version__
from sklearn_pandas import DataFrameMapper

# to make this notebook's output stable across runs
np.random.seed(42)

# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

#%% =============================================================================
# Stochatisic GD RBF
# =============================================================================
# Grid serach
clf_sgd_rbf = sk.pipeline.make_pipeline(sk.kernel_approximation.RBFSampler(gamma=0.1, random_state=1), 
                                        sk.linear_model.SGDClassifier())

############ Parameter grid #################
alpha_range = 10.0**-np.arange(1,7)
#alpha_range = 10.0**-np.arange(1,2)
loss_function = ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"]
#loss_function = ["hinge"]
max_iter = [1000]
tol = [1e-3]

params = dict(
    sgdclassifier__alpha=alpha_range,
    sgdclassifier__loss=loss_function,
    sgdclassifier__max_iter = max_iter,
    sgdclassifier__tol = tol,
)

params = dict(
    #sgdclassifier__alpha=alpha_range,
    sgdclassifier__loss=["log"],
    sgdclassifier__max_iter = max_iter,
    sgdclassifier__tol = tol,
)
############################################

# FIT
start = datetime.datetime.now()
grid = sk.grid_search.GridSearchCV(estimator=clf_sgd_rbf, 
                                   param_grid=params, 
                                   cv=5, 
                                   scoring='neg_log_loss', 
                                   n_jobs=-1, 
                                   verbose=2)
grid.fit(train_X, train_Y)
logging.debug("Elapsed: {}".format(datetime.datetime.now()-start))

#%% Analysis
print("best score:", grid.best_score_)
print("parameter:", grid.best_params_)

df_grid = grid_scores_to_df(grid.grid_scores_)

df_grid_agg = df_grid.groupby(['param_set','sgdclassifier__loss']).agg(['mean','var']).reset_index()
df_grid_agg.drop('fold',1,inplace=True)
#df_grid_agg.columns
df_grid_agg.index = df_grid_agg['param_set']
df_grid_agg.drop('fold',1,inplace=True)

clf_sgd_rbf_BEST = grid.best_estimator_

#%% SAVE 
if 0:
    print("Save to file...")
    g = h5py.File("data_%s.hdf5" % "logreg", "w")
    g.create_dataset("allypred", data=allypred)
    g.create_dataset("avg_loss", data=avg_loss)
    g.close()

#%% Train on these params
predicted = clf_sgd_rbf_BEST.predict_proba(test_X)
predicted_cat_num = predicted.argmax(axis=1)
predicted_cat_str = pd.Series(le_cat.inverse_transform(predicted_cat_num))