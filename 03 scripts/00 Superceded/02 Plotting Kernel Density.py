
import pandas as pd
import numpy as np
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
import matplotlib
# Supplied map bounding box:
#    ll.lon     ll.lat   ur.lon     ur.lat
#    -122.52469 37.69862 -122.33663 37.82986
PATH_MAP = "/home/batman/git/hack_sfpd/INPUT/sf_map_copyright_openstreetmap_contributors.txt"
mapdata = np.loadtxt(PATH_MAP)
asp = mapdata.shape[0] * 1.0 / mapdata.shape[1]

lon_lat_box = (-122.5247, -122.3366, 37.699, 37.8299)
clipsize = [[-122.5247, -122.3366],[ 37.699, 37.8299]]


PATH_OUT = r"/home/batman/git/hack_sfpd/Out"

#%% Cmap mapper
# http://scipy-cookbook.readthedocs.io/items/Matplotlib_ColormapTransformations.html

def cmap_map(function, cmap):
    """ Applies function (which should operate on vectors of shape 3: [r, g, b]), on colormap cmap.
    This routine will break any discontinuous points in a colormap.
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # Firt get the list of points where the segments start or end
    for key in ('red', 'green', 'blue'):
        step_dict[key] = list(map(lambda x: x[0], cdict[key]))
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step : np.array(cmap(step)[0:3])
    old_LUT = np.array(list(map(reduced_cmap, step_list)))
    new_LUT = np.array(list(map(function, old_LUT)))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(['red','green','blue']):
        this_cdict = {}
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j, i]
            elif new_LUT[j,i] != old_LUT[j, i]:
                this_cdict[step] = new_LUT[j, i]
        colorvector = list(map(lambda x: x + (x[1], ), this_cdict.items()))
        colorvector.sort()
        cdict[key] = colorvector

    return matplotlib.colors.LinearSegmentedColormap('colormap',cdict,1024)
#%%Subset!
df_sub = df[1:10000] 

#%% Create figure ALL
start_time = time.time()

#Seaborn FacetGrid, split by crime Category
g= sns.FacetGrid(df_sub, col="Category", col_wrap=6, size=5, aspect=1/asp)

#Show the background map
for ax in g.axes:
    ax.imshow(mapdata, cmap=plt.get_cmap('gray'), 
              extent=lon_lat_box, 
              aspect=asp)
#Kernel Density Estimate plot
g.map(sns.kdeplot, "X", "Y", clip=clipsize)

this_fig = plt.gcf()
# your code

elapsed_time = time.time() - start_time
logging.debug("{}".format(elapsed_time))
#%% Modify figure
this_fig.axes[0]

this_fig.show()

fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
fig.savefig('test2png.png', dpi=100)

this_fig.size

this_fig = pl.gcf()

#%% Save figure
plt.savefig('category_density_plot.png')
with open('myplot.pkl','wb') as fid:
    pickle.dump(this_fig, fid)

#%% Reload
this_path = r"/home/batman/git/hack_sfpd/Out/myplot.pkl"
with open(this_path,'rb') as fid:
    ax = pickle.load(fid)
plt.show()

#%% Do a larger plot with 1 category only


this_cat = 'ASSAULT'

for this_cat in df.Category.unique():
    start_time = datetime.now()
    
    df_1cat = df[df.Category == this_cat][0:10000] 
    df_1cat = df[df.Category == this_cat]
    #light_jet = cmap_map(lambda x: x/2 + 0.4, matplotlib.cm.gray)
    light_jet = cmap_map(lambda x: x*1.1, matplotlib.cm.gray)
    
    
    this_cat_fig =plt.figure(figsize=LANDSCAPE_A4)
    ax = sns.kdeplot(df_1cat.X, df_1cat.Y, clip=clipsize, aspect=1/asp, shade=False, color="r",cmap="seismic")
    #ax.imshow(mapdata, cmap=plt.get_cmap('Greys'), 
    #              extent=lon_lat_box, 
    #              aspect=asp)
    ax.imshow(mapdata, cmap=light_jet, 
                  extent=lon_lat_box, 
                  aspect=asp)
    
    min_time = df_1cat.dt.min().strftime("%Y-%m-%d")
    max_time = df_1cat.dt.max().strftime("%Y-%m-%d")
    num_recs = len(df_1cat)
    plt.suptitle("KDE plot for {} category".format(this_cat),y=0.95,fontsize=16)
    plt.title("{} to {}, {} records".format(min_time,max_time,num_recs))
    
    path_this_cat_out = os.path.join(PATH_OUT,this_cat+".png")
    
    plt.savefig(path_this_cat_out)
    
    elapsed_time = datetime.now() - start_time
    logging.debug("Wrote {} category to KDE map over {:0.1f}s".format(this_cat, elapsed_time.total_seconds()))

#%% Get countours

def get_contour_verts(cn):
    contours = []
    # for each contour line
    for cc in cn.collections:
        paths = []
        # for each separate section of the contour line
        for pp in cc.get_paths():
            xy = []
            # for each segment of that section
            for vv in pp.iter_segments():
                xy.append(vv[0])
            paths.append(np.vstack(xy))
        contours.append(paths)

    return contours


countours = get_contour_verts(ax)

