# =============================================================================
# Standard imports
# =============================================================================

# =============================================================================
# External imports - reimported for code completion! 
# =============================================================================
#print_imports()
#import pandas as pd # Import again for code completion
#import numpy as np # Import again for code completion
#import matplotlib as mpl
#import matplotlib.pyplot as plt
import sklearn as sk
#import sklearn
#from ExergyUtilities import util_sk_transformers as trn
from sklearn import pipeline
    
#%%********************************************
# Pipeline!
#**********************************************

feature_adding_pipeline = sk.pipeline.Pipeline([
        ('answer counts', trn.ValueCounter('question_id')),
        ('subreddit counts', trn.ValueCounter('subreddit')),
        ('count words in question', trn.WordCounter('question_text','no_of_words_in_question')),
        ('count words in answer', trn.WordCounter('answer_text','no_of_words_in_answer')),
        ('question hour', trn.TimeProperty('question_utc','question_hour','hour')),
        ('question day', trn.TimeProperty('question_utc','question_day','hour')),
        ('question month', trn.TimeProperty('question_utc','question_month','hour')),
        ('Answer delay seconds', trn.AnswerDelay('answer_delay_seconds',divisor=1)),
        ])

logging.info("Applying pipeline:")
for i,step in enumerate(feature_adding_pipeline.steps):
    print(i,step)

#%% Apply the pipeline
X_train_eng = feature_adding_pipeline.fit_transform(Xy_tr)
X_test_eng = feature_adding_pipeline.fit_transform(X_te)

#%% Add the sentiment
df_train_sentiment = pd.read_csv("./features/train_sentiment_score.csv")
df_test_sentiment = pd.read_csv("./features/test_sentiment_score.csv")

train_df = pd.merge(X_train_eng, df_train_sentiment, on='id')
test_df = pd.merge(X_test_eng, df_test_sentiment, on='id')

#%% Save to CSV
train_df.to_csv('dataset_train_simple_and_sentiment.csv')
test_df.to_csv('dataset_test_simple_and_sentiment.csv')

#%% DONE HERE 

print("******************************")
raise
