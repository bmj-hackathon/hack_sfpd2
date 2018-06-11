#%% ===========================================================================
# Grid search
# =============================================================================
###############################################################################
MODEL_STRING = "Logistic Regression"
###############################################################################

# Smaller subset for gridsearch
SAMPLE_FRAC = 0.5
SAMPLE_SIZE = int(len(train_X)*SAMPLE_FRAC)
train_X.sample(SAMPLE_SIZE)
sample_rows = pd.Series(train_X.index).sample(SAMPLE_SIZE).astype(str).values
logging.debug("Subset for Grid Search, {} rows".format(len(sample_rows)))

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
    #'penalty': ['l2','l1'],
    'penalty': ['l2'],
    'class_weight': [None,'balanced'],
    'fit_intercept': [True, False],
    'C':[0.25, 0.5,0.75,1.0]
    #'alpha': [10 ** x for x in range(-4, -2)],
    #'l1_ratio': [ 0.15], NOT NEEDED UNLESS ELASTIC
    #'max_iter': [1000],
    #'tol': [1e-3],
}
############################################

start = datetime.datetime.now()


cv_folds=5
clf_grid = sk.grid_search.GridSearchCV(estimator=clf, 
                                           param_grid=param_grid,cv=cv_folds,
                                           n_jobs=-1, 
                                           scoring='neg_log_loss',
                                           verbose=2)

###############################################################################
