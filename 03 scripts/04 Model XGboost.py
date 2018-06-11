import xgboost as xgb

#%% =============================================================================
# XG boost
# =============================================================================

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

