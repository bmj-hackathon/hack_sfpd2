
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