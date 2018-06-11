
min_time = df.dt.min().strftime("%Y-%m-%d")
max_time = df.dt.max().strftime("%Y-%m-%d")
num_recs = len(df)

plt.style.use('ggplot')

#%% Top crimes
top_crimes = df.Category.value_counts()[:10]
plt.figure(figsize=(12, 8))
pos = np.arange(len(top_crimes))
plt.barh(pos, top_crimes.values);
plt.yticks(pos, top_crimes.index);


#%% Top addresses

###############
title_str = "Most criminal addresses"
###############

plt.figure(figsize=LANDSCAPE_A3);

top_addresses = df.Address.value_counts()[:20]
#plt.figure(figsize=(12, 8))

pos = np.arange(len(top_addresses))
plt.barh(pos, top_addresses.values)
plt.yticks(pos, top_addresses.index);
plt.tight_layout(pad=5)
plt.suptitle(title_str,fontname = 'Arial', fontsize=16)
plt.title("{} to {}, {} records".format(min_time,max_time,num_recs))

# Save figure   
path_this_report_out = os.path.join(PATH_REPORTING,title_str+".pdf")
with open(path_this_report_out, 'wb') as f:
    plt.savefig(f,dpi=600)
logging.debug("Wrote to {}".format(path_this_report_out))

#%% Crimes and locations
###############
title_str = "Top Locations vs Top Crimes"
###############

cmap = sns.cm.rocket_r

#plt.figure(figsize=LANDSCAPE_A3)

fig, ax = plt.subplots(figsize=LANDSCAPE_A3);         # Sample figsize in inches
#sns.heatmap(df1.iloc[:, 1:6:], annot=True, linewidths=.5, )

plt.style.use('ggplot')

subset = df[df.Address.isin(top_addresses.index) & df.Category.isin(top_crimes.index)]
addr_cross_cat = pd.crosstab(subset.Address, subset.Category)

#sns.heatmap(addr_cross_cat, linewidths=.5,cmap='jet');
sns.heatmap(addr_cross_cat, linewidths=.5,cmap = cmap,ax=ax)
plt.tight_layout(pad=5)
plt.suptitle(title_str,fontname = 'Arial', fontsize=16)
plt.title("{} to {}, {} records".format(min_time,max_time,num_recs))

# Save figure   
path_this_report_out = os.path.join(PATH_REPORTING,title_str+".pdf")
with open(path_this_report_out, 'wb') as f:
    plt.savefig(f,dpi=600)
logging.debug("Wrote to {}".format(path_this_report_out))


#%% Crimes and days
###############
title_str = "Crimes over days of week"
###############

cmap = sns.cm.rocket_r

#plt.figure(figsize=LANDSCAPE_A3)

fig, ax = plt.subplots(figsize=LANDSCAPE_A3);         # Sample figsize in inches
#sns.heatmap(df1.iloc[:, 1:6:], annot=True, linewidths=.5, )

plt.style.use('ggplot')

day_cross_cat = pd.crosstab(df.Category, df.DayOfWeek).apply(lambda r: r/r.sum(), axis=1)
cols = ['Monday','Tuesday', 'Wednesday', 'Thursday', 'Friday','Saturday',  'Sunday', ]
day_cross_cat  = day_cross_cat[cols]
sns.heatmap(day_cross_cat, linewidths=.9,cmap = cmap,ax=ax);
plt.tight_layout(pad=5)
plt.suptitle(title_str,fontname = 'Arial', fontsize=16)
plt.title("{} to {}, {} records".format(min_time,max_time,num_recs))

# Save figure   
path_this_report_out = os.path.join(PATH_REPORTING,title_str+".pdf")
with open(path_this_report_out, 'wb') as f:
    plt.savefig(f,dpi=600)
logging.debug("Wrote to {}".format(path_this_report_out))


#%% Crimes and hours
###############
title_str = "Crimes over hours of the day"
###############

cmap = sns.cm.rocket_r

#plt.figure(figsize=LANDSCAPE_A3)

fig, ax = plt.subplots(figsize=LANDSCAPE_A3);         # Sample figsize in inches
#sns.heatmap(df1.iloc[:, 1:6:], annot=True, linewidths=.5, )

plt.style.use('ggplot')

hour_cross_cat = pd.crosstab(df.Category, df.hour).apply(lambda r: r/r.sum(), axis=1)

#cols = ['Monday','Tuesday', 'Wednesday', 'Thursday', 'Friday','Saturday',  'Sunday', ]
#day_cross_cat  = day_cross_cat[cols]
sns.heatmap(hour_cross_cat, linewidths=.9,cmap = cmap,ax=ax);
plt.tight_layout(pad=5)
plt.suptitle(title_str,fontname = 'Arial', fontsize=16)
plt.title("{} to {}, {} records".format(min_time,max_time,num_recs))

# Save figure   
path_this_report_out = os.path.join(PATH_REPORTING,title_str+".pdf")
with open(path_this_report_out, 'wb') as f:
    plt.savefig(f,dpi=600)
logging.debug("Wrote to {}".format(path_this_report_out))
