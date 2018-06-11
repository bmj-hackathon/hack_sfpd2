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


#%% =============================================================================
# Grid search
# =============================================================================
###############################################################################
MODEL_STRING = "SGD Vanilla"
###############################################################################

# Smaller subset for gridsearch
SAMPLE_FRAC = 0.6
SAMPLE_SIZE = int(len(train_X)*SAMPLE_FRAC)
train_X.sample(SAMPLE_SIZE)
sample_rows = pd.Series(train_X.index).sample(SAMPLE_SIZE).astype(str).values
logging.debug("Subset for Grid Search, {} rows".format(len(sample_rows)))

# Grid serach
clf = sk.linear_model.SGDClassifier(loss='log', 
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
    #'penalty': ['l2','l1'],
    'class_weight': [None,'balanced'],
    'alpha': [10 ** x for x in range(-7, 1)],
    
    #'l1_ratio': [ 0.15], NOT NEEDED UNLESS ELASTIC
    'max_iter': [1000],
    #'tol': [1e-3],
}
############################################

cv_folds=5
clf_grid = sk.grid_search.GridSearchCV(estimator=clf, 
                                           param_grid=param_grid,cv=cv_folds,
                                           n_jobs=-1, 
                                           scoring='neg_log_loss',
                                           verbose=2)


###############################################################################
#%% Fit
start = datetime.datetime.now()
clf_grid.fit(train_X.loc[sample_rows,:], train_Y.loc[sample_rows])
grid_search_elapsed = datetime.datetime.now()-start
grid_search_elapsed = strfdelta(grid_search_elapsed, "{hours:02d}:{minutes:02d}:{seconds:02d}")
#grid_search_elapsed.strptime
logging.debug("Elapsed H:m:s: {}".format(grid_search_elapsed))
print("best score:", clf_grid.best_score_)
print("Bast parameters:", clf_grid.best_params_)

#%% Save the grid search

path_grid_search_out = os.path.join(PATH_MODELS,MODEL_STRING+" grid search fit"+".pkl")
sk.externals.joblib.dump(clf_grid, path_grid_search_out)

#clf_grid_old = clf_grid
#clf_grid = sk.externals.joblib.load(path_confusionmatrix_out)

#%% Analysis of grid search
# Get the paramater space
df_grid = grid_scores_to_df(clf_grid.grid_scores_)
df_grid.fillna('None',inplace=True)

# Add the string labels (hack)
grid_labels = dict(df_grid.dtypes)
string_labels = list()
for k in grid_labels: 
    if grid_labels[k] == 'object': string_labels.append(k)
    
# Aggregate over each parameter set
df_grid_agg = df_grid.groupby(['param_set']+string_labels).agg(['mean','var']).reset_index()
df_grid_agg.drop('fold',1,inplace=True)
#df_grid_agg.columns
df_grid_agg.index = df_grid_agg['param_set']
df_grid_agg.drop('param_set',1,inplace=True)

# Write to Excel
path_grid_params = os.path.join(PATH_MODELS,MODEL_STRING+" grid space"+".xlsx")
with pd.ExcelWriter(path_grid_params) as writer:
    df_grid.to_excel(writer,'Grid Space over Folds')
    df_grid_agg.to_excel(writer,'Grid Space aggregate')
    writer.save()


clf_grid_BEST = clf_grid.best_estimator_

# Save the best
path_best_out = os.path.join(PATH_MODELS,MODEL_STRING+" best fit"+".pkl")
sk.externals.joblib.dump(clf_grid_BEST, path_best_out)

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
                          title=MODEL_STRING + ' Confusion matrix',
                          cmap=plt.cm.inferno,
                          )

path_confusionmatrix_out = os.path.join(PATH_MODELS,MODEL_STRING+" confusion"+".pdf")
plt.savefig(path_confusionmatrix_out,dpi=300,format='pdf',papertype='A4')
# Save the confusion matrix

#dir(plt.cm)
#%% One example
sample_selection = range(0,100)
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

#%% Save summary text
    
path_text_summary_out = os.path.join(PATH_MODELS,MODEL_STRING+" summary"+".txt")
with open(path_text_summary_out, 'a') as f:
    print(MODEL_STRING, file=f)
    print("Subset for Grid Search; {} rows (Fraction {})".format(len(sample_rows),SAMPLE_FRAC), file=f)
    print("Grid search completed over {} folds, {} parameter sets, over {} (H:M:S)".format(cv_folds,len(df_grid_agg),grid_search_elapsed))
    print("Best grid score:", clf_grid.best_score_, file=f)
    print("Best grid parameters:", clf_grid.best_params_, file=f)
    print("Predicted on test set with {} records".format(len(test_X)), file=f)
    print("Test set log-loss",test_logloss, file=f)
    print("Test set accurcacy",test_accuracy, file=f)
    print("{:50} {}".format('pred','act'),file=f)
    for pred,act in zip(predicted_values,actual_values):
        this_pred = "Wrong"
        if pred == act:
            this_pred = "Correct"
        print("{:10} {:50} {}".format(this_pred,pred,act),file=f)



