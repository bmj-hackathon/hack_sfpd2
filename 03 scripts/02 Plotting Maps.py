#%% Scatter plot over map

# We want to create a scatterplot of crime occurences for the whole city
# Borrowing the map and information from Ben's script
SF_map= np.loadtxt("../INPUT/sf_map_copyright_openstreetmap_contributors.txt")
# Supplied map bounding box:
#    ll.lon     ll.lat   ur.lon     ur.lat
#    -122.52469 37.69862 -122.33663 37.82986
lon_lat_box = (-122.5247, -122.3366, 37.699, 37.8299)
asp = SF_map.shape[0] * 1.0 / SF_map.shape[1]
fig = plt.figure(figsize=(16,16))
plt.imshow(SF_map,cmap='gray',extent=lon_lat_box,aspect=1/asp)
ax=plt.gca()
df.plot(x='X',y='Y',ax=ax,kind='scatter',marker='o',s=2,color='green',alpha=0.01)

ax.set_axis_off()
plt.savefig('TotalCrimeonMap.png')