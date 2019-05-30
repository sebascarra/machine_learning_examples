import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from timeit import timeit
#from matplotlib.ticker import LinearLocator, FormatStrFormatter
#from mayavi import mlab

#The information here alone does not work (later we'll see why): https://matplotlib.org/examples/mplot3d/surface3d_demo.html

#Read data from csv into X and y:
X = [] #N samples of size K (K = 2 since this is 2D)
T = [] #actual target outputs

for line in open("../linear_regression_class/data_2d.csv"):
    x1, x2, t = line.split(',')
    #We append inputs directly into X:
    X.append([float(x1), float(x2), 1]) #y = a*x1 + b*x2 + c
    T.append(float(t))

#After data loading, convert matrix X (each row is an x input)and column vector Y into numpy arrays:
X = np.array(X)
T = np.array(T)

# Thanks to:
# http://docs.enthought.com/mayavi/mayavi/auto/example_surface_from_irregular_data.html
# https://stackoverflow.com/questions/12423601/simplest-way-to-plot-3d-surface-given-3d-points
# https://stackoverflow.com/questions/53246874/why-z-has-to-be-2-dimensional-for-3d-plotting-in-matplotlib
# But this will not run on windows so forget about it...

# Define the points in 3D space
# including color code based on Z coordinate.
#pts = mlab.points3d(X[:,1], X[:,1], T, T)
#pts = mlab.points3d(X, Y, T)

# Triangulate based on x1, x2 with Delaunay 2D algorithm.
# Save resulting triangulation.
#mesh = mlab.pipeline.delaunay2d(pts)

# Remove the point representation from the plot
#pts.remove()

# Draw a surface based on the triangulation
#surf = mlab.pipeline.surface(mesh)

# Simple plot.
# mlab.xlabel("x")
# mlab.ylabel("y")
# mlab.zlabel("z")
# mlab.show()

# Plot the samples and targets to see what the data looks like:
# Create a figure object:
fig = plt.figure()
# Get the axes object of the plot added to the figure
ax = fig.add_subplot(111, projection='3d')
# Add the scatter to the plot:
ax.scatter(X[:,0], X[:,1], T)
# Draw the plot on screen:
plt.show()

doprint = True

#Y = Xw = w' * X' (because X rows are inputs x):
def wi(): return np.linalg.solve(np.dot(X.T, X), np.dot(X.T, T))
def wii(): return np.linalg.solve(np.dot(np.transpose(X),X), np.dot(np.transpose(X), T))
def wiii(): return np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X)), T)

w = wi()

# names = ['wi', 'wii', 'wiii']
# times = []

# for w in names:
#     times.append(timeit('{w}()'.format(w=w))) #Forget about this. It runs everything on a separate environment/process/session/terminal, so it's useless -> Don't lose focus!!

# if doprint:
#     for t, w in times, names:
#         print('{w}_time=', t)

#Now calculate Y = That:
#Y = X*w #Element-wise multiplication!!!
Y = X.dot(w) #Since X and w are not both 1-D, this is simple matrix multiplication. 

#Check the quality of the linear regression using r2:
d1 = T - Y
d2 = T - T.mean()

r2 = 1 - d1.dot(d1) / d2.dot(d2)

print(r2)