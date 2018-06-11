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

#df = df_ORIGINAL.sample(100000)

aaaNOTE = "SUBSET ENABLED"
df = df_sub

#%% Category encoding
le_cat = sk.preprocessing.LabelEncoder()
le_cat.fit(df.Category)
#dir(le_cat)
#.inverse_transform

#%% Pipeline
data_mapper = DataFrameMapper([
    ("district", sk.preprocessing.LabelBinarizer()),
    (["hour"], sk.preprocessing.StandardScaler()),
    (["weekday"], sk.preprocessing.StandardScaler()),
    (["dayofyear"], sk.preprocessing.StandardScaler()),
    (["month"], sk.preprocessing.StandardScaler()),
    (["year"], sk.preprocessing.StandardScaler()),
    (["lon"], sk.preprocessing.StandardScaler()),
    (["lat"], sk.preprocessing.StandardScaler()),
    ("holiday",  sk.preprocessing.LabelEncoder()),
    ("corner", sk.preprocessing.LabelEncoder()),
    ("weekend", sk.preprocessing.LabelEncoder()),
    ("workhour",  sk.preprocessing.LabelEncoder()),
    ("sunlight",  sk.preprocessing.LabelEncoder()),
    ("fri",  sk.preprocessing.LabelEncoder()),
    ("sat",  sk.preprocessing.LabelEncoder()),
    ('Category', sk.preprocessing.LabelEncoder()),
#    ("address", [sk.preprocessing.LabelEncoder(), sk.preprocessing.StandardScaler()]),
#    ("address", sk.preprocessing.LabelEncoder()),    
], df_out=True)

#


#print("Label encoder is working...")
#
#le_cat = sk.preprocessing.LabelEncoder()
#le_cat.fit(df.Category)
#
#le_day = sk.preprocessing.LabelEncoder()
#le_day.fit(df.DayOfWeek)
#
#le_pol = sk.preprocessing.LabelEncoder()
#le_pol.fit(df.PdDistrict);
#

#mapper = DataFrameMapper([
#        ("PdDistrict",None),
#        ("X", None),
#    
#        
#        
#        ('day', sk.preprocessing.LabelBinarizer()),
#    #('pet', sk.preprocessing.LabelBinarizer()),
#    #(['children'], sk.preprocessing.StandardScaler())
#], df_out=True)

for step in data_mapper.features:
    print(step)

#%% FIT TRANSFORM
df_trf = data_mapper.fit_transform(df.copy())
df_trf_head = df_trf.head()

#%% Split into train/test ONLY
if 0: 
    #train_data, validate_data = sk.cross_validation.train_test_split(res, test_size=0.15, random_state=42)
    train, test = sk.cross_validation.train_test_split(df_trf, test_size=0.20, random_state=42)
    print(len(train), len(test))
    
    train_X = train.drop('Category', 1)
    train_Y = train.Category
    test_X = test.drop('Category',1)
    test_Y = test.Category
    
    del train, test


#%% Split into train/test ONLY SAMLL VERSION
if 1: 
    #train_data, validate_data = sk.cross_validation.train_test_split(res, test_size=0.15, random_state=42)
    train, test = sk.cross_validation.train_test_split(df_trf, test_size=0.20, random_state=42)
    print(len(train), len(test))
    train = train.sample(frac=0.2)
    test= test.sample(frac=0.2)
    
    train_X = train.drop('Category', 1)
    train_Y = train.Category
    test_X = test.drop('Category',1)
    test_Y = test.Category
    
    del train, test, 


#%% Split into train/validate/test
if 0:
    train, validate, test = np.split(df_trf.sample(frac=1), [int(.6*len(df_trf)), int(.8*len(df_trf))])
    print(len(train), len(validate), len(test))
    
    train_X = train.drop('Category', 1)
    train_Y = train.Category
    validate_X = validate.drop('Category',1)
    validate_Y = validate.Category
    test_X = test.drop('Category',1)
    test_Y = test.Category


#%% Split into train/validate test SMALL VERSION
if 0:
    #train_data, validate_data = sk.cross_validation.train_test_split(res, test_size=0.15, random_state=42)
    train, validate, test = np.split(df_trf.sample(frac=1), [int(.6*len(df_trf)), int(.8*len(df_trf))])
    
    train = train.sample(frac=0.2)
    validate= validate.sample(frac=0.2)
    test= test.sample(frac=0.2)
    
    print(len(train), len(validate), len(test))
    
    train_X = train.drop('Category', 1)
    train_Y = train.Category
    validate_X = validate.drop('Category',1)
    validate_Y = validate.Category
    test_X = test.drop('Category',1)
    test_Y = test.Category


#%% DONE HERE - DELETE UNUSED
print("******************************")

del_vars =[
        "df_original",
        "df_sub",
        "df",
        'df_head',
        "df_trf",
        "df_trf_head",
        "step",
        "test",
        ]
cnt = 0
for name in dir():
    if name in del_vars:
        cnt+=1
        del globals()[name]
logging.info(f"Removed {cnt} variables from memory")
del cnt, name, del_vars
