#from IPython.core.display import HTML
#
#with open('creative_commons.txt', 'r') as f:
#    html = f.read()
#    
#name = '2014-03-10-google-earth'
#
#html = """
#<small>
#<p> This post was written as an IPython notebook.  It is available for
#<a href="https://ocefpaf.github.com/python4oceanographers/downloads/
#notebooks/%s.ipynb">download</a> or as a static
#<a href="https://nbviewer.ipython.org/url/ocefpaf.github.com/
#python4oceanographers/downloads/notebooks/%s.ipynb">html</a>.</p>
#<p></p>
#%s """ % (name, name, html)
#
##%matplotlib inline
from matplotlib import style
style.use('ggplot')

#%% 

#Here is a quick example on how to create a kmzfile with image overlays using matplotlib and simplekml.
#
#The make_kml() function below is just a wrapper around simplekml. It takes as arguments:
#
#A list of matplotlib figures;
#The figure(s) LatLon box (all overlays must have the same bbox);
#An optional keyword for the colorbar for one of the overlays;
#Some keyword options to tweak the kml/kmz file.
#All keywords (kw) can be modified during the make_kml() call. You can read more about them here.
#
#from simplekml import (Kml, OverlayXY, ScreenXY, Units, RotationXY,
#                       AltitudeMode, Camera)
#

def make_kml(llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat,
             figs, colorbar=None, **kw):
    """TODO: LatLon bbox, list of figs, optional colorbar figure,
    and several simplekml kw..."""

    kml = Kml()
    altitude = kw.pop('altitude', 2e7)
    roll = kw.pop('roll', 0)
    tilt = kw.pop('tilt', 0)
    altitudemode = kw.pop('altitudemode', AltitudeMode.relativetoground)
    camera = Camera(latitude=np.mean([urcrnrlat, llcrnrlat]),
                    longitude=np.mean([urcrnrlon, llcrnrlon]),
                    altitude=altitude, roll=roll, tilt=tilt,
                    altitudemode=altitudemode)

    kml.document.camera = camera
    draworder = 0
    for fig in figs:  # NOTE: Overlays are limited to the same bbox.
        draworder += 1
        ground = kml.newgroundoverlay(name='GroundOverlay')
        ground.draworder = draworder
        ground.visibility = kw.pop('visibility', 1)
        ground.name = kw.pop('name', 'overlay')
        ground.color = kw.pop('color', '9effffff')
        ground.atomauthor = kw.pop('author', 'ocefpaf')
        ground.latlonbox.rotation = kw.pop('rotation', 0)
        ground.description = kw.pop('description', 'Matplotlib figure')
        ground.gxaltitudemode = kw.pop('gxaltitudemode',
                                       'clampToSeaFloor')
        ground.icon.href = fig
        ground.latlonbox.east = llcrnrlon
        ground.latlonbox.south = llcrnrlat
        ground.latlonbox.north = urcrnrlat
        ground.latlonbox.west = urcrnrlon

    if colorbar:  # Options for colorbar are hard-coded (to avoid a big mess).
        screen = kml.newscreenoverlay(name='ScreenOverlay')
        screen.icon.href = colorbar
        screen.overlayxy = OverlayXY(x=0, y=0,
                                     xunits=Units.fraction,
                                     yunits=Units.fraction)
        screen.screenxy = ScreenXY(x=0.015, y=0.075,
                                   xunits=Units.fraction,
                                   yunits=Units.fraction)
        screen.rotationXY = RotationXY(x=0.5, y=0.5,
                                       xunits=Units.fraction,
                                       yunits=Units.fraction)
        screen.size.x = 0
        screen.size.y = 0
        screen.size.xunits = Units.fraction
        screen.size.yunits = Units.fraction
        screen.visibility = 1

    kmzfile = kw.pop('kmzfile', 'overlay.kmz')
    kml.savekmz(kmzfile)
    
#%% We will also need a gearth_fig() function. It is actually is a wrapper around matplotlib's Figure and Axes to create a figure that is "Google-Earth KML" friendly. The ideas for this function are originally from the octant library.
    
import numpy as np
import matplotlib.pyplot as plt


def gearth_fig(llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat, pixels=1024):
    """Return a Matplotlib `fig` and `ax` handles for a Google-Earth Image."""
    aspect = np.cos(np.mean([llcrnrlat, urcrnrlat]) * np.pi/180.0)
    xsize = np.ptp([urcrnrlon, llcrnrlon]) * aspect
    ysize = np.ptp([urcrnrlat, llcrnrlat])
    aspect = ysize / xsize

    if aspect > 1.0:
        figsize = (10.0 / aspect, 10.0)
    else:
        figsize = (10.0, 10.0 * aspect)

    if False:
        plt.ioff()  # Make `True` to prevent the KML components from poping-up.
    fig = plt.figure(figsize=figsize,
                     frameon=False,
                     dpi=pixels//10)
    # KML friendly image.  If using basemap try: `fix_aspect=False`.
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(llcrnrlon, urcrnrlon)
    ax.set_ylim(llcrnrlat, urcrnrlat)
    return fig, ax

#%% We will test it with the Mean Dynamic Topography from AVISO. Below are examples for two overlays, the Mean Dynamic Topography and the velocity vectors derived from it.
    
import numpy.ma as ma
from netCDF4 import Dataset, date2index, num2date

nc = Dataset('./data/mdt_cnes_cls2009_global_v1.1.nc')

u = nc.variables['Grid_0002'][:]
v = nc.variables['Grid_0003'][:]

lat = nc.variables['NbLatitudes'][:]
lon = nc.variables['NbLongitudes'][:]
lat, lon = np.meshgrid(lat, lon)

mdt = nc.variables['Grid_0001'][:]
mdt = ma.masked_equal(mdt, 9999.0)

#%% TEST

lon_lat_box = (-122.5247, -122.3366, 37.699, 37.8299)

llcrnrlon=-122.5247    #lon.min(),
llcrnrlat=37.699       #lat.min(),
urcrnrlon=-122.3366    #lon.max(),
urcrnrlat= 37.8299     #lat.max(),

pixels = 1024 * 10

fig, ax = gearth_fig(llcrnrlon=llcrnrlon,
                     llcrnrlat=llcrnrlat,
                     urcrnrlon=urcrnrlon,
                     urcrnrlat=urcrnrlat,
                     pixels=pixels)



#%% Overlay 1:

from palettable import colorbrewer

pixels = 1024 * 10
cmap = colorbrewer.get_map('RdYlGn', 'diverging', 11, reverse=True).mpl_colormap

fig, ax = gearth_fig(llcrnrlon=lon.min(),
                     llcrnrlat=lat.min(),
                     urcrnrlon=lon.max(),
                     urcrnrlat=lat.max(),
                     pixels=pixels)
cs = ax.pcolormesh(lon, lat, mdt, cmap=cmap)
ax.set_axis_off()
fig.savefig('overlay1.png', transparent=False, format='png')

#%% ... and overlay 2
fig, ax = gearth_fig(llcrnrlon=lon.min(),
                     llcrnrlat=lat.min(),
                     urcrnrlon=lon.max(),
                     urcrnrlat=lat.max(),
                     pixels=pixels)
Q = ax.quiver(lon[::10, ::10], lat[::10, ::10], u[::10, ::10], v[::10, ::10], scale=30)
ax.quiverkey(Q, 0.86, 0.45, 1, '1 m s$^{-1}$', labelpos='W')
ax.set_axis_off()
fig.savefig('overlay2.png', transparent=True, format='png')


#%%

make_kml(llcrnrlon=lon.min(), llcrnrlat=lat.min(),
         urcrnrlon=lon.max(), urcrnrlat=lat.max(),
         figs=['overlay1.png', 'overlay2.png'], colorbar='legend.png',
         kmzfile='mdt_uv.kmz', name='Mean Dynamic Topography and velocity')
#%%
#from IPython.core.display import Image
#Image('../../figures/gearth.jpg', retina=True)