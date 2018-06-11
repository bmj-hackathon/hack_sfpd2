# =============================================================================
# External imports - reimported for code completion! 
# =============================================================================
from ExergyUtilities import util_sk_transformers
import ExergyUtilities.util_sk_transformers
from ExergyUtilities import util_sk_transformers as trn

#import util_sk_transformers from ExergyUtilities

import itertools as itertools

#np.set_printoptions(suppress=True)
import sklearn as sk
#import sklearn
#from ExergyUtilities import util_sk_transformers as trn
from sklearn import pipeline
from pandas.tseries.holiday import USFederalHolidayCalendar
import sklearn.preprocessing

#%% Make some copies, one to keep, one to transform
df = sfpd_all.copy()
df_original = sfpd_all.copy()
#del sfpd_all
#df_sub = sfpd_all.iloc[0:10000,:]
#df_source["X"]

#%%********************************************
# Pipeline!
#**********************************************

this_pipeline_OLD = sk.pipeline.Pipeline([
        #('Empty', trn.Empty()),
        ('create dt column', 
             trn.ConvertDoubleColToDatetime(new_col_name="dt",name_col1="Date", name_col2="Time",this_format=r'%m/%d/%Y %H:%M')),
         
        ('create utm column',
             trn.UTMGridConvert(new_col_name='UTM',lat_col="Y", long_col="X"))
        ])

this_pipeline = sk.pipeline.Pipeline([
        #('Empty', trn.Empty()),
        ('create dt column', 
             trn.ConvertDoubleColToDatetime(new_col_name="dt",name_col1="Date", name_col2="Time",this_format=r'%m/%d/%Y %H:%M')),

        ])


logging.info("Created pipeline:")
for i,step in enumerate(this_pipeline.steps):
    print(i,step[0],step[1].__str__()[:60])

#%% Apply the pipeline

logging.info("Applying pipeline:")
#df = 
this_pipeline.fit_transform(df)
logging.info("Finished pipeline to {} columns".format(len(df.columns)))

#%% Feature columns

# Clean up
df.rename(index=str, columns={"X": "lat", "Y": "lon","PdDistrict":"district"},inplace=True)

# Time features
days_off = USFederalHolidayCalendar().holidays(start='2000-01-01', end='2020-01-01').to_pydatetime()
df['day'] = df['dt'].dt.weekday_name
df['dayofyear'] = df['dt'].dt.dayofyear
df['weekday'] = df['dt'].dt.weekday
df['month'] = df['dt'].dt.month
df['year'] = df['dt'].dt.year
df['hour'] = df['dt'].dt.hour
df["corner"] = df["Address"].map(lambda x: "/" in x) 

# Binary time features
df['holiday'] = df['dt'].dt.round('D').isin(days_off)
sum(df['holiday'])
df['weekend'] = df['dt'].dt.weekday >= 5
df['workhour'] = df['hour'].isin(range(9,17)) & ~df["dt"].isin(days_off) & ~df['weekend']
df['sunlight'] = df['hour'].isin(range(7,19))
df["fri"] = df['dt'].dt.weekday_name == "Friday"
df["sat"] = df['dt'].dt.weekday_name == "Saturday"
# Address feature
df['address'] = df["Address"].map(lambda x: x.split(" ", 1)[1] if x.split(" ", 1)[0].isdigit() else x)

print(df.columns)
df.info()


#%% Label encoding for target variables


#%% 

#df['Category'] = df['Category'].astype("category")
df_head = df.head()


#%% CREATE SUBSET HERE
#selected_num = 8
#pareto_crime
#Main_Crime_Categories = list(pareto_crime[0:selected_num].index)
#print("The following categories :")
#print(Main_Crime_Categories)
#print("make up to {:.2%} of the crimes".format(pareto_crime[selected_num]))
selected_crimes = ['LARCENY/THEFT', 
                   'OTHER OFFENSES', 
                   'NON-CRIMINAL', 
                   'ASSAULT', 
                   'VEHICLE THEFT', 
                   'DRUG/NARCOTIC', 
                   'VANDALISM', 
                   'BURGLARY',
                   'SUSPICIOUS OCC',
                   'MISSING PERSON',
                   'ROBBERY',
                   'WEAPON LAWS',
                   'TRESPASS',
                   'PROSTITUTION',
                   ]
#print("make up to {:.2%} of the crimes".format(pareto_crime[selected_crimes]))

#df_all = df
df_sub = df[df['Category'].isin(selected_crimes)]
print(df_sub['Category'].unique())
#print("make up to {:.2%} of the crimes".format(pareto_crime[selected_num]))

#%% DONE HERE - DELETE UNUSED
print("******************************")

del_vars =[
        "days_off",
        "dropped_rows",
        "i",
        "sfpd_all",
        "step",
        "total_rows",
        ]
cnt = 0
for name in dir():
    if name in del_vars:
        cnt+=1
        del globals()[name]
logging.info(f"Removed {cnt} variables from memory")
del cnt, name, del_vars


