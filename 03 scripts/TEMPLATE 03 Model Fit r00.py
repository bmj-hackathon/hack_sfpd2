# =============================================================================
# Standard imports
# =============================================================================
import tarfile
import urllib as urllib
import os
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info('Started logging')

# =============================================================================
# External imports - reimported for code completion! 
# =============================================================================
print_imports()
import pandas as pd # Import again for code completion
import numpy as np # Import again for code completion
import matplotlib as mpl
import matplotlib.pyplot as plt
#sns.set(style="ticks")
import sklearn as sk
import sklearn
import sklearn.linear_model
# to make this notebook's output stable across runs
np.random.seed(42)
from sklearn_pandas import DataFrameMapper
from sklearn_features.transformers import DataFrameSelector

# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

# Plotly
import plotly.plotly as py
py.plotly.tools.set_credentials_file(username='notbatman', api_key='1hy2cho61mYO4ly5R9Za')
import plotly.graph_objs as go
print(sk)
#%%
sns.set()

def load_data(filename):
    data = pd.read_csv(filename, sep="\t", index_col='id')
    msg = "Reading the data ({} rows). Columns: {}"
    print(msg.format(len(data), data.columns))
    # Select the columns (feel free to select more)
    X = data.loc[:, ['question_text', 'answer_text']]
    try:
        y = data.loc[:, "answer_score"]
    except KeyError: # There are no answers in the test file
        return X, None
    return X, y

#%% Reload the data
DATA_ROOT = r"/home/batman/git/KaggleDaysReddit"

X_train_path = os.path.join(DATA_ROOT,"train_with_tfidf.csv")
assert os.path.exists(X_train_path)
train_df = pd.read_csv(X_train_path,index_col=0)

X_test_path = os.path.join(DATA_ROOT,"test_with_tfidf.csv")
test_df = pd.read_csv(X_test_path,index_col=0,encoding='utf-8')

y_train_path = os.path.join(DATA_ROOT,"y_train.csv")
y_train = pd.read_csv(y_train_path,index_col=0,header=None)
y_train= y_train.iloc[:,0]


#%% This is the metric, as defined in baseline code
def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(
        np.mean( (np.log1p(y) - np.log1p(y0)) ** 2 )
        )

#%% Drop non-numeric columns
cols_to_drop = ['question_id','id','subreddit', 'question_utc', 'answer_utc', 'question_text', 'answer_text']
train_df_numeric = train_df.drop(cols_to_drop,axis=1)
test_df_numeric = test_df.drop(cols_to_drop,axis=1)

train_df_numeric.info()


#%% JOIN NEW DATA
train_df2 = model.fit_transform(train_df)
train_df3 = pd.DataFrame(train_df2.toarray())
test_df2 = model.fit_transform(test_df)
test_df3 = pd.DataFrame(test_df2.toarray())

train_df = pd.concat([train_df, train_df3], axis=1)
test_df = pd.concat([test_df, test_df3], axis=1)

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
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
clf = sk.ensemble.GradientBoostingRegressor(**params)
clf.fit(train_df_numeric, np.log1p(y_train))


#%% Predict
y_train_predicted = clf.predict(train_df_numeric)
y_test_predicted = clf.predict(test_df_numeric)

res = pd.DataFrame(y_train_predicted)
res.describe()
res.hist(bins=1000)

#%% Evaluate
# Calculate exp(x) - 1 for all elements in the array.
#y_train_predicted_cut[y_train_predicted > 100] = 100

y_train_theor = np.expm1(y_train_predicted)
y_test_theor = np.expm1(y_test_predicted)
print()
print("Training set")
print("RMSLE:   ", rmsle(y_train_predicted, y_train_theor))

sk.metrics.mean_squared_error(y_train,y_train_predicted)

#%% Export
X_val, _ = load_data('/home/batman/git/KaggleDaysReddit/data/test.csv')
#X_val, _ = load_data('test.csv')

solution = pd.DataFrame(index=X_val.index)
solution['answer_score'] = np.expm1(y_test_predicted)
solution.to_csv('/home/batman/git/KaggleDaysReddit/submission_gradientboosted.csv')

#%%**************************************************************************************
# Random Forest 
#****************************************************************************************
from sklearn import ensemble

forest_reg = sk.ensemble.RandomForestRegressor(n_jobs=-1)
forest_reg.fit(train_df_numeric, np.log1p(y_train))

#%% Predict
y_train_predicted = forest_reg.predict(train_df_numeric)
y_test_predicted = forest_reg.predict(test_df_numeric)

res = pd.DataFrame(y_train_predicted)
res.describe()
res.hist(bins=1000)

#%% Evaluate

y_train_theor = np.expm1(y_train_predicted)
y_test_theor = np.expm1(y_test_predicted)
print()
print("Training set")
print("RMSLE:   ", rmsle(y_train_predicted, y_train_theor))

sk.metrics.mean_squared_error(y_train,y_train_predicted)

#%%**************************************************************************************
# Stochastic Gradient Descent
#****************************************************************************************
from sklearn import linear_model
#params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
#          'learning_rate': 0.01, 'loss': 'ls'}
#clf = sk.ensemble.GradientBoostingRegressor(**params)
clf = sk.linear_model.SGDRegressor()
print(clf)
clf.fit(train_df_numeric, np.log1p(y_train))

#%% Predict
y_train_predicted = clf.predict(train_df_numeric)
y_test_predicted = clf.predict(test_df_numeric)

res = pd.DataFrame(y_train_predicted)
res.describe()
res.hist(bins=1000)

#%% Evaluate
# Calculate exp(x) - 1 for all elements in the array.
#y_train_predicted_cut[y_train_predicted > 100] = 100

y_train_theor = np.expm1(y_train_predicted)
y_test_theor = np.expm1(y_test_predicted)
print()
print("Training set")
print("RMSLE:   ", rmsle(y_train_predicted, y_train_theor))

sk.metrics.mean_squared_error(y_train,y_train_predicted)

#%% Export
X_val, _ = load_data('/home/batman/git/KaggleDaysReddit/data/test.csv')
#X_val, _ = load_data('test.csv')

solution = pd.DataFrame(index=X_val.index)
solution['answer_score'] = np.expm1(y_test_predicted)
solution.to_csv('/home/batman/git/KaggleDaysReddit/submission_gradientboosted.csv')

