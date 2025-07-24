from vedo import *
import numpy as np

msh = Mesh(dataurl+"man.vtk")
arr = msh.points()

# site  patient counts as of 9/30/2021
LLE = 33
RLE = 35

LUE = 11
RUE = 8 

Axilla = 2
Neck = 1
Pelvis = 7
Trunk = 9

# site locations 
RLE_loc = np.all([arr[:,0]<0, arr[:,0]>-0.5, arr[:,2]<0], axis=0)
LLE_loc = np.all([arr[:,0]>0, arr[:,0]<0.5, arr[:,2]<0], axis=0)

RUE_loc = np.all([arr[:,0]<-0.4, arr[:,2]>-0.2], axis=0)
LUE_loc = np.all([arr[:,0]>0.4, arr[:,2]>-0.2], axis=0)

Axilla_loc = np.all([np.any([np.all([arr[:,0]>-0.4, arr[:,0]<-0.25], axis=0), np.all([arr[:,0]>0.25, arr[:,0]<0.4], axis=0)], axis=0), arr[:,2]>0.7, arr[:,2]<1.0], axis=0)

Neck_loc = np.all([arr[:,2]>1.1, arr[:,2]<1.3], axis=0)

Pelvis_loc = np.all([arr[:,0]>-0.5, arr[:,0]<0.5, arr[:,2]>0, arr[:,2]<0.2], axis=0)
Trunk_loc = np.all([arr[:,0]>-0.4, arr[:,0]<0.4, arr[:,2]>0.2, arr[:,2]<1.1], axis=0)

# assign values 
colors = np.zeros_like(arr)

colors[LLE_loc] = 33
colors[RLE_loc] = 35

colors[LUE_loc] = 11
colors[RUE_loc] = 8 

colors[Neck_loc] = 1
colors[Pelvis_loc] = 7
colors[Trunk_loc] = 9
colors[Axilla_loc] = 2

# colors[LLE_loc] = 1
# colors[RLE_loc] = 2

# colors[LUE_loc] = 3
# colors[RUE_loc] = 4 

# colors[Neck_loc] = 6
# colors[Pelvis_loc] = 7
# colors[Trunk_loc] = 8
# colors[Axilla_loc] = 5


msh.pointdata["myxcoords"] = colors
msh.cmap("jet", "myxcoords")

plt = Plotter(axes=0)

plt.camera.SetPosition( [-0.514, -7.305, 0.572] )
plt.camera.SetFocalPoint( [0.0, -0.034, 0.057] )
plt.camera.SetViewUp( [0.014, 0.07, 0.997] )
plt.camera.SetDistance( 7.307 )
plt.camera.SetClippingRange( [6.291, 8.603] )
plt.show(msh)

video = Video("body.mp4", duration=8, backend='opencv')

video.action(azimuth_range=(0,359), elevation_range=(0,0), resetcam=False)

video.close() 

# show(msh, axes=1)

# # make sure it's in range 0-255
# print(np.min(arr, axis=0),np.max(arr, axis=0))
# pts = arr.astype(np.uint8)

# msh.pointdata["myxcoords"] = arr
# msh.mapPointsToCells().print()

# msh.celldata.select("myxcoords")

# show(msh, axes=1)