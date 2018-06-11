
# =============================================================================
# External imports - reimported for code completion! 
# =============================================================================
from ExergyUtilities import util_sk_transformers as trn

#%% DATE TIME conversion
Xy_tr = trn.ConvertToDatetime('question_utc').transform(Xy_tr)
Xy_tr = trn.ConvertToDatetime('answer_utc').transform(Xy_tr)
X_te = trn.ConvertToDatetime('question_utc').transform(Xy_tr)
logging.info(f"Converted columns to datetime in both train and test")

#%% 
Xy_tr_desc = Xy_tr.describe(include = 'all')
X_te_desc = X_te.describe(include = 'all')

count_question_ids = Xy_tr.loc[:,['subreddit','question_id']].groupby('subreddit').count()
count_uniques = Xy_tr.groupby('subreddit').nunique()
#counts = pd.merge(count_answers,count_questions)


#%%
#Xy_tr.hist()


