#%% ===========================================================================
# Grid search
# =============================================================================
###############################################################################
MODEL_STRING = "Random Forest"
###############################################################################

# Smaller subset for gridsearch
SAMPLE_FRAC = 0.5
SAMPLE_SIZE = int(len(train_X)*SAMPLE_FRAC)
train_X.sample(SAMPLE_SIZE)
sample_rows = pd.Series(train_X.index).sample(SAMPLE_SIZE).astype(str).values
logging.debug("Subset for Grid Search, {} rows".format(len(sample_rows)))

# Grid serach

clf = sk.ensemble.RandomForestClassifier(n_estimators=10, 
                                             criterion='gini', 
                                             max_depth=None, 
                                             min_samples_split=2, 
                                             min_samples_leaf=1, 
                                             min_weight_fraction_leaf=0.0, 
                                             max_features='auto', 
                                             max_leaf_nodes=None, 
                                             min_impurity_decrease=0.0, 
                                             min_impurity_split=None, 
                                             bootstrap=True, 
                                             oob_score=False, 
                                             n_jobs=1, 
                                             random_state=None, 
                                             verbose=0, 
                                             warm_start=False, 
                                             class_weight=None)

############ Parameter grid #################
param_grid = {
    'n_estimators':[400,600],
    'max_depth':[10],
    'min_samples_split':[5,10],
    'min_samples_leaf':[3,6],
}
############################################


cv_folds=3
clf_grid = sk.grid_search.GridSearchCV(estimator=clf, 
                                           param_grid=param_grid,cv=cv_folds,
                                           n_jobs=-1, 
                                           scoring='neg_log_loss',
                                           verbose=2)

###############################################################################
