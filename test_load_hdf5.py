# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
    test_load_hdf5.py
"""

""" Clear all variables """
from IPython import get_ipython
get_ipython().magic('reset -sf') 

""" Load libraries """
import h5py
import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry.polygon import LinearRing
from shapely.geometry import Polygon
from shapely.geometry import Point

""" 
    There is a problem with the coordinates which are read by the "read_roi"
    package, using the ijroi alternative. This is unfortunate because:
    "read_roi" results in a dictionary describing all the ImageJ ROI such as 
    name, type, ... while ijroi only returns the name of the roi (bla.roi) and 
    the coordinates as (y, x) numpy array
"""
#from read_roi import read_roi_file
#from read_roi import read_roi_zip
from ijroi import read_roi_zip


def openHdf5ImageFirst( filePath ):
    """
    Open (hdf5) image
    """
    f = h5py.File( filePath,'r+') 
    #print("Keys: %s" % f.keys())
    a_group_key = list( f.keys() )[0]
    #print("group_key: %s" % a_group_key )
    img = f.get(a_group_key)
    #print(img.shape)
    return img, a_group_key


def loadRegions( filePath ):
    """
    Read in regions for an image
    """
    roisArray = read_roi_zip( filePath )
    rois = {}
    for el in roisArray:
        rois[el[0].replace(".roi","")] = np.fliplr( el[1] )

    return rois


def loadRegionsAsPolygon( filePath ):
    """
    Read in regions for an image
    """
    roisArray = read_roi_zip( filePath )
    rois = {}
    for el in roisArray:
        rois[el[0].replace(".roi","")] = Polygon( np.fliplr( el[1] ) )

    return rois

def doesRegionContainsPoint( polygon, point ):
    """
        Point in region
    """
    if ( polygon.contains( point ) ):
        return True
    else:
        return False

def doRegionsOverlap( polygon1, polygon2 ):
    """
    Check whether 2 regions overlap, returning a boolean value
    """
    polygonIntersection = polygon1.intersection(polygon2)
    if ( polygonIntersection == None ):
        return False
    else:
        return True


def intersectRegions( polygon1, polygon2 ):
    """
    Intersect 2 regions
    """
    polygonIntersection = polygon1.intersection(polygon2)
    return polygonIntersection


def tilePolygon( x0, y0, w, h ):
    """ 
    Define a rectangle polygon with counterclockwise rectangle coords, the 
    returned polygon is a "shapely.geometry.Polygon"
    """
    xyCoords = [ [x0,y0], [x0+w,y0], [x0+w,y0+h], [x0,y0+h], [x0,y0] ]
    polygon = Polygon( xyCoords )
    return polygon


def intersectTileWithRegion( x0, y0, w, h, polygon ):
    """
    Intersect tile with region
    """
    tile = tilePolygon( x0, y0, w, h )
    return intersectRegions( tile, polygon )


"""
    Read in an image tile from open hdf5 file
"""
""" 
    ---------------------------------------------------------------------------
"""


"""
    SCRIPT
"""


#print(polygon.contains(point))

rois = loadRegions('/data/mbarbier/external/axioscan_jnj/B31/large_rois/B31-01.zip')
roisPoly = loadRegionsAsPolygon('/data/mbarbier/external/axioscan_jnj/B31/large_rois/B31-01.zip')

fig1 = plt.figure(1, dpi=90)
plt.axis('equal')
for key in rois.keys():
    xy = rois[key]
    polygon = Polygon( xy )
    ax1 = fig1.add_subplot(111)
    x, y = polygon.exterior.xy
    plt.plot(x, y)
    plt.show()

fig2 = plt.figure(2, dpi=90)
plt.axis('equal')
for key in roisPoly.keys():
    pxy = roisPoly[key]
    ax2 = fig2.add_subplot(111)
    x, y = pxy.exterior.xy
    plt.plot(x, y)
    plt.show()


roisPoly["cx"]

[img, imgName] = openHdf5ImageFirst( '/home/mbarbier/Documents/prog/DeepSlice/B31.h5' ) 
x0 = 6000
y0 = 5000
w = 5000
h = 4000
#imgTile = img[x0:w+x0, y0:h+y0]
#fig2 = plt.figure()
#ax2 = fig2.add_subplot(111)
#plt.imshow( imgTile, cmap='gray')

polygon = intersectTileWithRegion( x0, y0, w, h, roisPoly["mb"])
#polygon = tilePolygon( x0, y0, w, h )
fig3 = plt.figure(3, dpi=90)
plt.axis('equal')
pxy = polygon
ax3 = fig3.add_subplot(111)
x, y = pxy.exterior.xy
plt.plot(x, y)
plt.show()

"""
roi_cb = rois["cb"]

x = [1177731072.0, 1177944064.0, 1177747456.0]
y = [1153302528.0, 1151467520.0, 1151336448.0]

xx = x[0] - (2**31)
print(x[0])
print(2**30)
print(2**31)
print(x[0] - (2**30))
print(x[0] - (2**31))


minx = min(x)
maxx = max(x)
miny = min(y)
maxy = max(y)
x0 = minx
y0 = miny
w = maxx - minx
h = maxy - miny

wf = 208
hf = 240
x0f = 11440
y0f = 1280
minxf = x0f
maxxf = x0f + wf
minyf = y0f
maxyf = y0f + hf

factorw = w/wf
factorh = w/wf
factorx = x0/x0f
factory = y0/y0f

print(x[0] - 1177719632.0)
print(x[1] - 1177719632.0)
print(x[2] - 1177719632.0)
"""


"""
point = Point(0.5, 0.5)

x = rois["hp"]["x"]
y = rois["hp"]["y"]
polygon = Polygon( zip(x,y) )

fig = plt.figure(1, dpi=90)
ax = fig.add_subplot(111)
plt.plot(x, y)
plt.show()

fig2 = plt.figure(2, dpi=90)
ax2 = fig2.add_subplot(111)
x, y = polygon.exterior.xy
plt.plot(x, y)
plt.show()
"""

"""
ring = LinearRing([(0, 0), (1, 1), (1, 0), (0, 0)])
print(ring.area)
print(ring.length)

polygon = Polygon([(0, 0), (1, 1), (1, 0), (0, 0)])
print(polygon.area)
print(polygon.length)

polygon2 = Polygon([(0, 0), (0, 1), (1, 0), (0, 0)])
print(polygon2.area)
print(polygon2.length)

polyIntersection = polygon.intersection(polygon2)

fig = plt.figure(1, dpi=90)
ax = fig.add_subplot(111)

ax.set_title('Polygon intersection')
xrange = [-1, 2]
yrange = [-1, 2]
ax.set_xlim(*xrange)
ax.set_xticks(list(range(*xrange)) + [xrange[-1]])
ax.set_ylim(*yrange)
ax.set_yticks(list(range(*yrange)) + [yrange[-1]])
ax.set_aspect(1)

x, y = polygon.exterior.xy
plt.plot(x, y)
x, y = polygon2.exterior.xy
plt.plot(x, y)
x, y = polyIntersection.exterior.xy
plt.plot(x, y)

plt.show()

[img, imgName] = openHdf5ImageFirst( '/home/mbarbier/Documents/prog/DeepSlice/B31.h5' ) 
x0 = 10000
y0 = 8000
w = 1000
h = 2000
imgTile = img[x0:w+x0, y0:h+y0]
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
plt.imshow( imgTile, cmap='gray')
"""

""" 
    ---------------------------------------------------------------------------
"""

