
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import matplotlib

#%% SF Map
# Supplied map bounding box:
#    ll.lon     ll.lat   ur.lon     ur.lat
#    -122.52469 37.69862 -122.33663 37.82986
PATH_MAP = "/home/batman/git/hack_sfpd/INPUT/sf_map_copyright_openstreetmap_contributors.txt"
mapdata = np.loadtxt(PATH_MAP)
asp = mapdata.shape[0] * 1.0 / mapdata.shape[1]

lon_lat_box = (-122.5247, -122.3366, 37.699, 37.8299)
clipsize = [[-122.5247, -122.3366],[ 37.699, 37.8299]]


#%% =============================================================================
# KDE per Category
# =============================================================================
finished = [
        'BURGLARY',
        'VANDALISM',
        'ASSAULT',
        'ROBBERY',
        'NON-CRIMINAL',
        ]
df_kde_subset = df_sub[~df_sub['Category'].isin(finished)]

for this_cat in df_kde_subset.Category.unique():
    #print(this_cat)
    
    start_time = datetime.now()
    
    # Get this single category df
    df_1cat = df_kde_subset[df_kde_subset.Category == this_cat]
        
    logging.debug("Processing KDE for {}, {} records".format(this_cat, len(df_1cat)))
    #continue
    
    #df_1cat = df_kde_subset[df_kde_subset.Category == this_cat][0:100] 
    
    
    # Make map brighter
    light_jet = cmap_map(lambda x: x*1.1, matplotlib.cm.gray)
    
    # Create figure
    this_cat_fig =plt.figure(figsize=LANDSCAPE_A3)
    ax = sns.kdeplot(df_1cat.X, df_1cat.Y, clip=clipsize, aspect=1/asp, shade=False, color="r",cmap="seismic")
    ax.set_xlabel("Longitude [Decimal deg.]")
    ax.set_ylabel("Latitude [Decimal deg.]")
    ax.imshow(mapdata, cmap=light_jet, 
                  extent=lon_lat_box, 
                  aspect=asp)
    
    # Create title
    min_time = df_1cat.dt.min().strftime("%Y-%m-%d")
    max_time = df_1cat.dt.max().strftime("%Y-%m-%d")
    num_recs = len(df_1cat)
    plt.suptitle("KDE plot for {} category".format(this_cat),y=0.95,fontsize=16)
    plt.title("{} to {}, {} records".format(min_time,max_time,num_recs))
    
    # Save figure PNG
    this_sanitized_cat = this_cat.replace("/", " ")
    
    path_this_cat_out = os.path.join(PATH_OUT_KDE,this_sanitized_cat+".pdf")
    plt.savefig(path_this_cat_out,dpi=600)
    elapsed_time = datetime.now() - start_time
    logging.debug("Wrote {} category to KDE map over {:0.1f}s".format(this_cat, elapsed_time.total_seconds()))
    
    # Save contours
    countours = get_contour_verts(ax)
    path_this_contour_out = os.path.join(PATH_OUT_KDE,this_sanitized_cat+"_contour.pck")
    with open(path_this_contour_out, 'wb') as f:
        pickle.dump(countours,f)
    logging.debug("Wrote {} contours".format(this_cat))



#%% =============================================================================
# KDE ALL
# =============================================================================


start_time = datetime.now()

# Get this single category df
df_1cat = df.sample(frac=0.2)
this_cat = "ALL DATA"
logging.debug("Processing KDE for ALL, {} records".format(len(df_1cat)))

# Make map brighter
light_jet = cmap_map(lambda x: x*1.1, matplotlib.cm.gray)

# Create figure
this_cat_fig =plt.figure(figsize=LANDSCAPE_A3)
ax = sns.kdeplot(df_1cat.X, df_1cat.Y, clip=clipsize, aspect=1/asp, shade=False, color="r",cmap="seismic")
ax.set_xlabel("Longitude [Decimal deg.]")
ax.set_ylabel("Latitude [Decimal deg.]")
ax.imshow(mapdata, cmap=light_jet, 
              extent=lon_lat_box, 
              aspect=asp)

# Create title
min_time = df_1cat.dt.min().strftime("%Y-%m-%d")
max_time = df_1cat.dt.max().strftime("%Y-%m-%d")
num_recs = len(df_1cat)
plt.suptitle("KDE plot for {} category".format(this_cat),y=0.95,fontsize=16)
plt.title("{} to {}, {} records".format(min_time,max_time,num_recs))

# Save figure PNG
this_sanitized_cat = this_cat.replace("/", " ")

path_this_cat_out = os.path.join(PATH_OUT_KDE,this_sanitized_cat+".pdf")
plt.savefig(path_this_cat_out,dpi=600)
elapsed_time = datetime.now() - start_time
logging.debug("Wrote {} category to KDE map over {:0.1f}s".format(this_cat, elapsed_time.total_seconds()))

# Save contours
countours = get_contour_verts(ax)
path_this_contour_out = os.path.join(PATH_OUT_KDE,this_sanitized_cat+"_contour2.pck")
with open(path_this_contour_out, 'wb') as f:
    pickle.dump(countours,f)
logging.debug("Wrote {} contours".format(this_cat))