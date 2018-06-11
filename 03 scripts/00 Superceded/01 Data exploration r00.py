#%% Summary
df_summary = df.describe(include = 'all')
df_summary.loc["count",:].astype(int)

#%%=============================================================================
# Overall counts
# =============================================================================
# How many crimes per *category*? 
# Category || Count

title_str = 'Total crime incidents'

categories = df["Category"].value_counts()
# == OR ==
categories = df.groupby(["Category"]).size()
categories.columns = "Count"
categories.sort_values(inplace=True, ascending=False)

# How many crimes per *category* & *description*? 
# idx || Category | Descript | Count
#desc_counts = df.groupby(["Category","Descript",]).size().reset_index(name='counts')

#%% Overall count PLOT

min_time = df.dt.min().strftime("%Y-%m-%d")
max_time = df.dt.max().strftime("%Y-%m-%d")
num_recs = len(df)

plt.figure(figsize=LANDSCAPE_A3)
plt.style.use('ggplot')
_n_crime_plot = sns.barplot(x=categories.index,y=categories)
_n_crime_plot.set_xticklabels(categories.index,rotation=90)
_n_crime_plot.set_ylabel("Total count")
plt.tight_layout(pad=5)
plt.suptitle(title_str,fontname = 'Arial', fontsize=16)
plt.title("{} to {}, {} records".format(min_time,max_time,num_recs))

# Save figure   
path_this_report_out = os.path.join(PATH_REPORTING,title_str+".pdf")
with open(path_this_report_out, 'wb') as f:
    plt.savefig(f,dpi=600)
logging.debug("Wrote to {}".format(this_cat, elapsed_time.total_seconds()))
    

#%% =============================================================================
# Select only 80% of crime categories
# =============================================================================

title_str = 'Cumulative count of crime categories'
pareto_crime = categories / sum(categories)
pareto_crime = pareto_crime.cumsum()

#%% Plot
plt.figure(figsize=LANDSCAPE_A3)
plt.style.use('ggplot')
_pareto_crime_plot = sns.tsplot(data=pareto_crime)
_pareto_crime_plot.set_xticklabels(pareto_crime.index,rotation=90)
_pareto_crime_plot.set_xticks(np.arange(len(pareto_crime)))
_pareto_crime_plot.set_ylabel("Cumulative")
_pareto_crime_plot.set_ylim([0,1.05])
plt.yticks(np.arange(0, 1.1, .1))
plt.tight_layout(pad=5)
plt.suptitle(title_str,fontname = 'Arial', fontsize=16)
plt.title("{} to {}, {} records".format(min_time,max_time,num_recs))

# Save figure   
path_this_report_out = os.path.join(PATH_REPORTING,title_str+".pdf")
with open(path_this_report_out, 'wb') as f:
    plt.savefig(f,dpi=600)
logging.debug("Wrote to {}".format(this_cat, elapsed_time.total_seconds()))


#%% CREATE SUBSET HERE
selected_num = 8
pareto_crime
Main_Crime_Categories = list(pareto_crime[0:selected_num].index)
print("The following categories :")
print(Main_Crime_Categories)
print("make up to {:.2%} of the crimes".format(pareto_crime[selected_num]))
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
#print("make up to {:.2%} of the crimes".format(pareto_crime[selected_num]))


#%%=============================================================================
# Yearly counts ALL
# =============================================================================
# Break down *category* over years
# idx || Category | Year | Count
cat_counts_years = df.groupby(["Category",df['dt'].apply(lambda x: x.year)]).size().reset_index(name='counts')
cat_counts_years.columns = ["Category","Year","Count"]
cat_counts_years = pd.pivot_table(cat_counts_years,values="Count", index=["Category"], columns='Year')
cat_counts_years.fillna(value=0)
cat_counts_years.sort_values(by=cat_counts_years.columns[0],inplace=True, ascending=False)

# Break down *category* & *description* over years
# idx || Category | Descript | Year | Count
desc_counts_years = df.groupby(
        ["Category",
         "Descript",
         df['dt'].apply(lambda x: x.year),
        ]).size().reset_index(name='Count')
desc_counts_years.columns = ["Category","Descript","Year","Count"]
desc_counts_years = pd.pivot_table(desc_counts_years,values="Count", index=["Category","Descript"], columns='Year')
#index = list(itertools.product(df['Category'].unique(), df_all['Descript'].unique()))
#desc_counts_years = desc_counts_years.reindex(index)

#%%
def autolabel(rects):
"""
Attach a text label above each bar displaying its height
"""
for rect in rects:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
            '%d' % int(height),
            ha='center', va='bottom')


#%% plot
title_str = 'Crimes over years, 2012 - 2017'

plt.figure(figsize=LANDSCAPE_A3)
plt.style.use('ggplot')

#axhandles=cat_counts_years.iloc[:,-6:-1].plot(kind='bar',subplots=True,figsize=PORTRAIT_A3,sharex=True,sharey=True,rot=90,table=True)
axhandles=cat_counts_years.iloc[:,-7:-1].plot(kind='bar',subplots=True,figsize=PORTRAIT_A3,sharex=True,sharey=True,rot=90,table=False)

i=1
for ax in axhandles:
    ax.legend(loc='best')
    ax.set_title("")
    ax.set_ylabel("Count")
    ax.set_ylim([0,50000])
    
#    totals = list()
#    for rec in ax.patches:
#        print(i)
#        totals.append(rec.get_height())
#    
#    
#    for rec in ax.patches:
#        ax.text(rec.get_height()+.3, rec.get_width()+.38, str(int(rec.get_height())), fontsize=15,color='dimgrey')
    i+=1

plt.suptitle(title_str,fontname = 'Arial', fontsize=16)

plt.tight_layout(pad=2.5)

#plt.title("{} to {}, {} records".format(min_time,max_time,num_recs))

# Save figure   
path_this_report_out = os.path.join(PATH_REPORTING,title_str+".pdf")
with open(path_this_report_out, 'wb') as f:
    plt.savefig(f,dpi=600)
logging.debug("Wrote to {}".format(this_cat, elapsed_time.total_seconds()))

#%% plot
title_str = 'Crimes over years, 2006 - 2012'

plt.figure(figsize=LANDSCAPE_A3)
plt.style.use('ggplot')

axhandles=cat_counts_years.iloc[:,-13:-7].plot(kind='bar',subplots=True,figsize=PORTRAIT_A3,sharex=True,sharey=True,rot=90)

i=1
for ax in axhandles:
    ax.legend(loc='best')
    ax.set_title("")
    ax.set_ylabel("Count")
    ax.set_ylim([0,50000])
    i+=1
    

plt.suptitle(title_str,fontname = 'Arial', fontsize=16)

plt.tight_layout(pad=2.5)

#plt.title("{} to {}, {} records".format(min_time,max_time,num_recs))

# Save figure   
path_this_report_out = os.path.join(PATH_REPORTING,title_str+".pdf")
with open(path_this_report_out, 'wb') as f:
    plt.savefig(f,dpi=600)
    
logging.debug("Wrote to {}".format(this_cat, elapsed_time.total_seconds()))


#%% plot
title_str = 'Crimes over years, 2003 - 2008'

plt.figure(figsize=LANDSCAPE_A3)
plt.style.use('ggplot')

axhandles=cat_counts_years.iloc[:,:-10].plot(kind='bar',subplots=True,figsize=PORTRAIT_A3,sharex=True,sharey=True,rot=90)

i=1
for ax in axhandles:
    ax.legend(loc='best')
    ax.set_title("")
    ax.set_ylabel("Count")
    ax.set_ylim([0,50000])
    i+=1
    

plt.suptitle(title_str,fontname = 'Arial', fontsize=16)

plt.tight_layout(pad=2.5)

#plt.title("{} to {}, {} records".format(min_time,max_time,num_recs))

# Save figure   
path_this_report_out = os.path.join(PATH_REPORTING,title_str+".pdf")
with open(path_this_report_out, 'wb') as f:
    plt.savefig(f,dpi=600)
    
logging.debug("Wrote to {}".format(this_cat, elapsed_time.total_seconds()))


#%% =============================================================================
# #%% Counts, Overall monthly
# =============================================================================
cat_counts_months = df.groupby(["Category",df['dt'].apply(lambda x: x.month)]).size().reset_index(name='Count')
cat_counts_months.columns = ["Category","Month","Count"]
cat_counts_months = pd.pivot_table(cat_counts_months,values="Count", index=["Category"], columns='Month')
cat_counts_months.fillna(value=0)
cat_counts_months.sort_values(by=cat_counts_months.columns[0],inplace=True, ascending=False)

#%% Plot
plt.loc[:,Main_Crime_Categories].plot(ylim=0)



#%% =============================================================================
# #%% Counts, Overall daily over year
# =============================================================================
cat_counts_daynum = df.groupby(["Category",df['dt'].apply(lambda x: x.timetuple().tm_yday)]).size().reset_index(name='Count')
cat_counts_daynum.columns = ["Category","Day","Count"]
cat_counts_daynum = pd.pivot_table(cat_counts_daynum,values="Count", index=["Category"], columns='Day')
cat_counts_daynum.fillna(value=0)
cat_counts_daynum.sort_values(by=cat_counts_daynum.columns[0],inplace=True, ascending=False)


#%% =============================================================================
# #%% Counts, Overall day of week
# =============================================================================
#cat_counts_dayname = df.groupby(["Category",df['dt'].apply(lambda x: x.weekday())]).size().reset_index(name='Count')
cat_counts_dayname1 = df.groupby(["Category",df['dt'].apply(lambda x: x.day_name())]).size()
cat_counts_dayname = cat_counts_dayname1.reset_index()
cat_counts_dayname.columns = ["Category","Day","Count"]
cat_counts_dayname = pd.pivot_table(cat_counts_dayname,values="Count", index=["Category"], columns='Day')
cat_counts_dayname.fillna(value=0, inplace=True)
cat_counts_dayname.sort_values(by=cat_counts_dayname.columns[0],inplace=True, ascending=False)

#%% DAY NAME PLOT
ax=cat_counts_dayname.T.iloc[:,0:12].plot(kind='line',subplots=False,figsize=PORTRAIT_A3,sharex=True,sharey=True,rot=90)
ax.legend(loc='upper center', bbox_to_anchor=(1, 1), ncol=1, fancybox=True, shadow=True)
ax.set_title("ASDF")
i+=1


#%% Counts, Each month for each year 
# NOTE THE HACK - RENAME THE COLUMN TO AVOID DUPLICATE ERROR
cat_counts_years_months = df.groupby([
            "Category",
            df['dt'].apply(lambda x: x.month).rename('Month'),
            df['dt'].apply(lambda x: x.year).rename('Year')
        ]).size().reset_index(name='Count')
cat_counts_years_months = pd.pivot_table(cat_counts_years_months,
                                         values="Count", 
                                         index=["Category"], columns=['Year','Month'])

#cat_counts_years_months.T.plot.bar()

#%% Write to excel
if 0:
    writer = pd.ExcelWriter('output.xlsx')
    desc_counts_years.to_excel(writer,'Sheet1')
    writer.save()








#df.groupby(df['dt'].dt.year).size()
#res = df.groupby(df['dt'].apply(lambda x: x.year)).value_col.sum()
#res.columns

#grp = df.groupby(by=[
#        df['dt'].map(lambda x : x.hour),
#        df['dt'].map(lambda x : x.minute)
#            ]).size()

#%% EXAMPLE
if 0:
    gb = df.groupby(['col1', 'col2'])
    counts = gb.size().to_frame(name='counts')
    (counts
     .join(gb.agg({'col3': 'mean'}).rename(columns={'col3': 'col3_mean'}))
     .join(gb.agg({'col4': 'median'}).rename(columns={'col4': 'col4_median'}))
     .join(gb.agg({'col4': 'min'}).rename(columns={'col4': 'col4_min'}))
     .reset_index()
    )
    
#    Out[6]: 
#      col1 col2  counts  col3_mean  col4_median  col4_min
#    0    A    B       4  -0.372500       -0.810     -1.32
#    1    C    D       3  -0.476667       -0.110     -1.65
#    2    E    F       2   0.455000        0.475     -0.47
#    3    G    H       1   1.480000       -0.630     -0.63