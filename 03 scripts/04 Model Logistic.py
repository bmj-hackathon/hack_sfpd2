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


#%% =============================================================================
# Stochatisic GD PLAIN
# =============================================================================
#SAMPLE_SIZE = 100000
# Grid serach
clf = sk.linear_model.LogisticRegression(penalty='l2', 
                                         dual=False, 
                                         tol=0.0001, 
                                         C=1.0, 
                                         fit_intercept=True, 
                                         intercept_scaling=1, 
                                         class_weight=None, 
                                         random_state=None, 
                                         solver='liblinear', 
                                         max_iter=100, 
                                         multi_class='ovr', 
                                         verbose=0, 
                                         warm_start=False, 
                                         n_jobs=1)

############ Parameter grid #################
param_grid = {
    'penalty': ['l2','l1'],
    'class_weight': [None,'balanced'],
    #'alpha': [10 ** x for x in range(-4, -2)],
    #'l1_ratio': [ 0.15], NOT NEEDED UNLESS ELASTIC
    #'max_iter': [1000],
    #'tol': [1e-3],
}
############################################

start = datetime.datetime.now()

clf_grid = sk.grid_search.GridSearchCV(estimator=clf, 
                                           param_grid=param_grid,cv=5,
                                           n_jobs=-1, 
                                           scoring='neg_log_loss',
                                           verbose=2)

clf_grid.fit(train_X, train_Y)
logging.debug("Elapsed: {}".format(datetime.datetime.now()-start))

#%% Analysis of grid search
print("best score:", clf_grid.best_score_)
print("parameter:", clf_grid.best_params_)

df_grid = grid_scores_to_df(clf_grid.grid_scores_)

df_grid_agg = df_grid.groupby(['param_set','loss', 'penalty']).agg(['mean','var']).reset_index()
df_grid_agg.drop('fold',1,inplace=True)
#df_grid_agg.columns
df_grid_agg.index = df_grid_agg['param_set']
df_grid_agg.drop('fold',1,inplace=True)

clf_grid_BEST = clf_grid.best_estimator_

#%% Predict on Test set
predicted = clf_grid_BEST.predict_proba(test_X)

#%% Results on TEST

predicted_cat_num = predicted.argmax(axis=1)
predicted_cat_str = pd.Series(le_cat.inverse_transform(predicted_cat_num))
test_logloss = sk.metrics.log_loss(test_Y, predicted)
print(test_logloss)

#predicted_cat_num = predicted.argmax(axis=1)
#predicted_cat_str = pd.Series(le_cat.inverse_transform(predicted_cat_num))

test_accuracy = sk.metrics.accuracy_score(test_Y, predicted_cat_num)
print(test_accuracy)
#print(sk.metrics.log_loss(res, predicted))

confusion_matrix = pd.DataFrame(sklearn.metrics.confusion_matrix(test_Y, predicted_cat_num))
confusion_matrix.columns = selected_crimes
confusion_matrix.index = selected_crimes


#%% Confusion
plot_confusion_matrix(confusion_matrix,
                          selected_crimes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.inferno
                          )
#dir(plt.cm)
#%% One example
sample_selection = range(0,10)
sample1 = test_X.iloc[sample_selection,:]
print(sample1)
predict1 = clf_grid_BEST.predict_proba(sample1)
predicted_cat_num1 = predict1.argmax(axis=1)
predicted_cat_str1 = pd.Series(le_cat.inverse_transform(predicted_cat_num1))
#print()
predicted_values = predicted_cat_str1.values

actual_values = test_Y.iloc[sample_selection].astype(int)
actual_values = le_cat.inverse_transform(actual_values)

for pred,act in zip(predicted_values,actual_values):
    print("{:50} {}".format(pred,act))


#test_logloss = sk.metrics.log_loss(test_Y, predicted)
#print(test_logloss)

