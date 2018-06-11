# =============================================================================
# Standard imports
# =============================================================================
import os
import logging
#import datetime
#import gc
#import zipfile

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


#from sklearn_pandas import DataFrameMapper
#from sklearn_features.transformers import DataFrameSelector
#from pandas.tseries.holiday import USFederalHolidayCalendar
#from sklearn.cross_validation import KFold, cross_val_score
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.linear_model import SGDClassifier
#from sklearn.grid_search import GridSearchCV
#from sklearn.kernel_approximation import RBFSampler
#from sklearn.pipeline import make_pipeline
#from sklearn.preprocessing import StandardScaler, LabelEncoder, LabelBinarizer
#from sklearn_pandas import DataFrameMapper

# to make this notebook's output stable across runs
np.random.seed(42)

# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")



days_off = USFederalHolidayCalendar().holidays(start='2003-01-01', end='2015-05-31').to_pydatetime()







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

