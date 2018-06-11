
#%% Plotly
import plotly.plotly as py
py.plotly.tools.set_credentials_file(username='notbatman', api_key='1hy2cho61mYO4ly5R9Za')
import plotly.graph_objs as go

#%% Seaborn

sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})




#%% DAY NUMBER PLOT
ax=cat_counts_daynum.T.iloc[:,:10].plot(kind='line',subplots=False,figsize=PORTRAIT_A3,sharex=True,sharey=True,rot=90)

ax.legend(loc='upper center', bbox_to_anchor=(1, 1), ncol=1, fancybox=True, shadow=True)

ax.set_title("ASDF")
i+=1


#%% DAY NUMBER PLOT ZOOM
ax=cat_counts_daynum.T.iloc[0:31,:10].plot(kind='line',subplots=False,figsize=PORTRAIT_A3,sharex=True,sharey=True,rot=90)
ax.legend(loc='upper center', bbox_to_anchor=(1, 1), ncol=1, fancybox=True, shadow=True)
ax.set_title("ASDF")
i+=1




#%% Example 
# Create the data
rs = np.random.RandomState(1979)
x = rs.randn(500)
g = np.tile(list("ABCDEFGHIJ"), 50)
df = pd.DataFrame(dict(x=x, g=g))
m = df.g.map(ord)
df["x"] += m

# Initialize the FacetGrid object
pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
g = sns.FacetGrid(df, row="g", hue="g", aspect=15, size=.5, palette=pal)

# Draw the densities in a few steps
g.map(sns.kdeplot, "x", clip_on=False, shade=True, alpha=1, lw=1.5, bw=.2)
g.map(sns.kdeplot, "x", clip_on=False, color="w", lw=2, bw=.2)
g.map(plt.axhline, y=0, lw=2, clip_on=False)

# Define and use a simple function to label the plot in axes coordinates
def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .2, label, fontweight="bold", color=color, 
            ha="left", va="center", transform=ax.transAxes)

g.map(label, "x")

# Set the subplots to overlap
g.fig.subplots_adjust(hspace=-.25)

# Remove axes details that don't play will with overlap
g.set_titles("")
g.set(yticks=[])
g.despine(bottom=True, left=True)
