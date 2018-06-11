

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