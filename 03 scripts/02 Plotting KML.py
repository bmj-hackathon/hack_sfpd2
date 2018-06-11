import simplekml
import pylab


#%% Utility
def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a


#%% Color mappers
from matplotlib import colors as mcolors

def get_color_hex(color_name):
    my_colors = dict(mpl.colors.BASE_COLORS, **mpl.colors.CSS4_COLORS)
    this_color = my_colors[color_name]
    return "ff{}".format(this_color[1:])

#get_color_hex('green')

def get_some_colors(num_colors):
    cm = pylab.get_cmap('gist_rainbow')
    colors = list()
    for i in range(num_colors):
        color = cm(1.*i/num_colors)  # color will now be an RGBA tuple
        color = color[0:3]
        color = [int(c * 255) for c in color]
        colors.append(color)
    return colors

#these_colors = get_some_colors(15)


#%% Generate one poly

def gen_poly(kml,countour_in,height,hex_color):
    pol = kml.newpolygon(name='A Polygon')
    
    # Get a contour
    this_countour_arr= countour_in
    num_v,_ = this_countour_arr.shape
    #this_countour_arr
    
    # Add z coord
    this_h = height
    height_col_vec = np.ones(num_v)*this_h
    this_countour_arr = np.c_[ this_countour_arr, height_col_vec ]  
    
    # Convert to KML contour
    this_countour = totuple(this_countour_arr)
    
    # Set height mode
    pol.altitudemode = 'relativeToGround'
    pol.extrude = 1
    
    # Assign contour to this poly
    pol.outerboundaryis = this_countour
    #pol.style.linestyle.color = simplekml.Color.red
    pol.style.linestyle.color = hex_color
    pol.style.linestyle.width = 1
    #pol.style.polystyle.color = simplekml.Color.changealphaint(100, simplekml.Color.red)
    pol.style.polystyle.color = simplekml.Color.changealphaint(100, hex_color)

    logging.debug("\t\tAdded contour at {} m with color {}".format(height,hex_color))
    return kml 


def gen_3D_kml(contour_set,height_offset, color_mode, color=None):
    # Start a new KML
    kml = simplekml.Kml()
    
    #if color_mode : cmap_mode = False
    #else: cmap_mode = True
        
    if color_mode=='cmap':
        # Generate each contour as a new color
        values = range(len(contour_set)) # Levels
        cNorm  = mpl.colors.Normalize(vmin=0, vmax=values[-1]) # Spread
        cmap = plt.get_cmap('seismic')
        cmap = plt.get_cmap('jet')
        scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=cmap)
    
    for i, level in enumerate(contour_set):
        if color_mode=='cmap':
            colorVal = scalarMap.to_rgba(values[i])
            rgb = [val * 255 for val in list(colorVal[0:3])]
            rgb = [int(val) for val in rgb]
            hex_color = "ff{0:02x}{1:02x}{2:02x}".format(*rgb)
        elif color_mode=='named': 
            hex_color = get_color_hex(color)
        elif color_mode=='rgb': 
            #hex_color = get_color_hex(color)
            hex_color = "ff{0:02x}{1:02x}{2:02x}".format(*color)
        else:
            raise
    
        #logging.debug("Curre{}".format(hex_color))
        logging.debug("\tProcessing level {}, color = {}".format(i,hex_color))
    
        for contour in level:
            this_height = i * height_offset
            kml = gen_poly(kml,contour,this_height,hex_color)
    return kml

#kml = gen_3D_kml(countours,height_offset, color_name='yellow')



#%% Get countour files over each cat
import os

# Get files
contour_files = list()
for file in os.listdir(PATH_OUT_KDE):
    if file.endswith(".pck"):
        contour_files.append(os.path.join(PATH_OUT_KDE, file))

height_offset = 50
# Process each
these_colors = get_some_colors(15)
for i,this_path in enumerate(contour_files):
    root,fname = os.path.split(this_path)
    #print(fname)
    catname, _x = fname.split("_")
    if catname == 'ALL DATA':
        continue
    logging.debug("Processing file {} {},  {}".format(i,catname, fname))
    
    # Get color for this set
    this_color = these_colors[i]
    logging.debug("Color {}".format(this_color))
    
    contour_set = pickle.load( open(this_path, "rb" ) )
    #contour_set,height_offset, color_mode, color=None
    kml = gen_3D_kml(contour_set,height_offset,color_mode = 'rgb', color=this_color)

    # Save KML
    path_this_contour_out = os.path.join(PATH_OUT_KDE,catname+"_google_earth_contour.kml")
    kml.save(path_this_contour_out)
    logging.debug("Wrote {} kml to {}".format(catname, path_this_contour_out))
    
#%% Get countours ALL
import os

# Get files
contour_files = list()
for file in os.listdir(PATH_OUT_KDE):
    if file.endswith(".pck"):
        contour_files.append(os.path.join(PATH_OUT_KDE, file))

height_offset = 50
# Process each
these_colors = get_some_colors(15)
for i,this_path in enumerate(contour_files):
    root,fname = os.path.split(this_path)
    #print(fname)
    catname, _x = fname.split("_")
    if catname != 'ALL DATA':
        continue
    
    logging.debug("Processing file {} {},  {}".format(i,catname, fname))
    
    # Get color for this set
    #this_color = these_colors[i]
    #logging.debug("Color {}".format(this_color))
    
    contour_set = pickle.load( open(this_path, "rb" ) )
    #contour_set,height_offset, color_mode, color=None
    kml = gen_3D_kml(contour_set,height_offset,color_mode = 'cmap', color=None)
    #break
    # Save KML
    path_this_contour_out = os.path.join(PATH_OUT_KDE,catname+"_google_earth_contour.kml")
    kml.save(path_this_contour_out)
    logging.debug("Wrote {} kml to {}".format(catname, path_this_contour_out))
    



