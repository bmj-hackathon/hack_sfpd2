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

#%% =============================================================================
# Stochatisic GD PLAIN
# =============================================================================
#SAMPLE_SIZE = 100000
# Grid serach
clf_sgd = sk.linear_model.SGDClassifier(loss='log', 
                                        penalty='l2', 
                                        alpha=0.0001, 
                                        l1_ratio=0.15, 
                                        fit_intercept=True, 
                                        max_iter=None, 
                                        tol=None, 
                                        shuffle=True, 
                                        verbose=0, 
                                        epsilon=0.1, 
                                        n_jobs=-2, 
                                        random_state=None, 
                                        learning_rate='optimal', 
                                        eta0=0.0, 
                                        power_t=0.5, 
                                        class_weight="balanced", 
                                        warm_start=False, 
                                        average=False, 
                                        n_iter=None)

############ Parameter grid #################
param_grid = {
    'loss': ['log'],
    'penalty': ['l2','l1'],
    'alpha': [10 ** x for x in range(-4, -2)],
    #'l1_ratio': [ 0.15], NOT NEEDED UNLESS ELASTIC
    'max_iter': [1000],
    'tol': [1e-3],
}
############################################

start = datetime.datetime.now()

clf_grid_sgd = sk.grid_search.GridSearchCV(estimator=clf_sgd, 
                                           param_grid=param_grid,cv=5,
                                           n_jobs=-1, 
                                           scoring='neg_log_loss',
                                           verbose=2)

clf_grid_sgd.fit(train_X, train_Y)
logging.debug("Elapsed: {}".format(datetime.datetime.now()-start))

#%% Analysis
print("best score:", clf_grid_sgd.best_score_)
print("parameter:", clf_grid_sgd.best_params_)

df_grid_sgd_plain = grid_scores_to_df(clf_grid_sgd.grid_scores_)

df_grid_sgd_plain_agg = df_grid.groupby(['param_set','sgdclassifier__loss']).agg(['mean','var']).reset_index()
df_grid_sgd_plain_agg.drop('fold',1,inplace=True)
#df_grid_agg.columns
df_grid_sgd_plain_agg.index = df_grid_sgd_plain_agg['param_set']
df_grid_sgd_plain_agg.drop('fold',1,inplace=True)

clf_grid_sgd_plain_BEST = clf_grid_sgd.best_estimator_


#%% Train on these params
predicted = clf_grid_sgd_plain_BEST.predict_proba(test_X)
predicted_cat_num = predicted.argmax(axis=1)
print(sk.metrics.log_loss(test_Y, predicted))
predicted_cat_str = pd.Series(le_cat.inverse_transform(predicted_cat_num))
res = sk.preprocessing.LabelBinarizer().fit_transform(predicted_cat_str)

#print(sk.metrics.log_loss(res, predicted))





#%% =============================================================================
# Stochatisic GD PLAIN
# =============================================================================
#SAMPLE_SIZE = 100000
# Grid serach
clf_sgd = sk.linear_model.SGDClassifier(loss='log', 
                                        penalty='l2', 
                                        alpha=0.0001, 
                                        l1_ratio=0.15, 
                                        fit_intercept=True, 
                                        max_iter=None, 
                                        tol=None, 
                                        shuffle=True, 
                                        verbose=0, 
                                        epsilon=0.1, 
                                        n_jobs=-2, 
                                        random_state=None, 
                                        learning_rate='optimal', 
                                        eta0=0.0, 
                                        power_t=0.5, 
                                        class_weight="balanced", 
                                        warm_start=False, 
                                        average=False, 
                                        n_iter=None)

############ Parameter grid #################
param_grid = {
    'loss': ['log'],
    'penalty': ['l2','l1'],
    'alpha': [10 ** x for x in range(-4, -2)],
    #'l1_ratio': [ 0.15], NOT NEEDED UNLESS ELASTIC
    'max_iter': [1000],
    'tol': [1e-3],
}
############################################

start = datetime.datetime.now()

clf_grid_sgd = sk.grid_search.GridSearchCV(estimator=clf_sgd, 
                                           param_grid=param_grid,cv=5,
                                           n_jobs=-1, 
                                           scoring='neg_log_loss',
                                           verbose=2)

clf_grid_sgd.fit(train_X, train_Y)
logging.debug("Elapsed: {}".format(datetime.datetime.now()-start))

#%% Analysis
print("best score:", clf_grid_sgd.best_score_)
print("parameter:", clf_grid_sgd.best_params_)

df_grid_sgd_plain = grid_scores_to_df(clf_grid_sgd.grid_scores_)

df_grid_sgd_plain_agg = df_grid.groupby(['param_set','sgdclassifier__loss']).agg(['mean','var']).reset_index()
df_grid_sgd_plain_agg.drop('fold',1,inplace=True)
#df_grid_agg.columns
df_grid_sgd_plain_agg.index = df_grid_sgd_plain_agg['param_set']
df_grid_sgd_plain_agg.drop('fold',1,inplace=True)

clf_grid_sgd_plain_BEST = clf_grid_sgd.best_estimator_


#%% Train on these params
predicted = clf_grid_sgd_plain_BEST.predict_proba(test_X)
predicted_cat_num = predicted.argmax(axis=1)
print(sk.metrics.log_loss(test_Y, predicted))
predicted_cat_str = pd.Series(le_cat.inverse_transform(predicted_cat_num))
res = sk.preprocessing.LabelBinarizer().fit_transform(predicted_cat_str)

#print(sk.metrics.log_loss(res, predicted))








#%% 
def set_param():
    # setup parameters for xgboost
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.4
    param['silent'] = 0
    param['nthread'] = 4
    param['num_class'] = num_class
    param['eval_metric'] = 'mlogloss'

    # Model complexity
    param['max_depth'] = 8 #set to 8
    param['min_child_weight'] = 1
    param['gamma'] = 0 
    param['reg_alfa'] = 0.05

    param['subsample'] = 0.8
    param['colsample_bytree'] = 0.8 #set to 1

    # Imbalanced data
    param['max_delta_step'] = 1
    
    return param

#%% Analaysis of fit
if 0:
    importances = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest_reg.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    
    # Print the feature ranking
    print("Feature ranking:")
    
    for f in range(train_df_numeric.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
        print(train_df_numeric.columns[indices[f]])
    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.show()


#%%**************************************************************************************
# Gradient Boosting Regression
#****************************************************************************************
if 0:
    params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
              'learning_rate': 0.01, 'loss': 'ls'}
    clf = sk.ensemble.GradientBoostingRegressor(**params)
    clf.fit(train_df_numeric, np.log1p(y_train))


#%% Predict
if 0:
    y_train_predicted = clf.predict(train_df_numeric)
    y_test_predicted = clf.predict(test_df_numeric)
    
    res = pd.DataFrame(y_train_predicted)
    res.describe()
    res.hist(bins=1000)

#%% Evaluate
# Calculate exp(x) - 1 for all elements in the array.
#y_train_predicted_cut[y_train_predicted > 100] = 100
if 0:
    y_train_theor = np.expm1(y_train_predicted)
    y_test_theor = np.expm1(y_test_predicted)
    print()
    print("Training set")
    print("RMSLE:   ", rmsle(y_train_predicted, y_train_theor))
    
    sk.metrics.mean_squared_error(y_train,y_train_predicted)


#%%**************************************************************************************
# Random Forest 
#****************************************************************************************
if 0:
    from sklearn import ensemble
    
    forest_reg = sk.ensemble.RandomForestRegressor(n_jobs=-1)
    forest_reg.fit(train_df_numeric, np.log1p(y_train))

#%% Predict
if 0:
    y_train_predicted = forest_reg.predict(train_df_numeric)
    y_test_predicted = forest_reg.predict(test_df_numeric)
    
    res = pd.DataFrame(y_train_predicted)
    res.describe()
    res.hist(bins=1000)

#%% Evaluate
if 0:
    y_train_theor = np.expm1(y_train_predicted)
    y_test_theor = np.expm1(y_test_predicted)
    print()
    print("Training set")
    print("RMSLE:   ", rmsle(y_train_predicted, y_train_theor))
    
    sk.metrics.mean_squared_error(y_train,y_train_predicted)

#%%**************************************************************************************
# Stochastic Gradient Descent
#****************************************************************************************
if 0:
        
    from sklearn import linear_model
    #params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
    #          'learning_rate': 0.01, 'loss': 'ls'}
    #clf = sk.ensemble.GradientBoostingRegressor(**params)
    clf = sk.linear_model.SGDRegressor()
    print(clf)
    clf.fit(train_df_numeric, np.log1p(y_train))

#%% Predict
if 0:
    y_train_predicted = clf.predict(train_df_numeric)
    y_test_predicted = clf.predict(test_df_numeric)
    
    res = pd.DataFrame(y_train_predicted)
    res.describe()
    res.hist(bins=1000)

#%% Evaluate
if 0:
        
    # Calculate exp(x) - 1 for all elements in the array.
    #y_train_predicted_cut[y_train_predicted > 100] = 100
    
    y_train_theor = np.expm1(y_train_predicted)
    y_test_theor = np.expm1(y_test_predicted)
    print()
    print("Training set")
    print("RMSLE:   ", rmsle(y_train_predicted, y_train_theor))
    
    sk.metrics.mean_squared_error(y_train,y_train_predicted)

