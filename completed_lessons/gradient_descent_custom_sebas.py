import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

N = 30
lamb = 0.49

#Create some data to work with:
x = np.array([np.linspace(-100, 100, N), np.linspace(-100, 100, N)])
#x = x.T.reshape(N, 2) #Matrix of inputs. This tutorial is god: https://stackoverflow.com/questions/22053050/difference-between-numpy-array-shape-r-1-and-r/22053783
#But because of the nature of the problem to understand and solve, we don't really need to reshape in this case.

#Plot the output:

#Some output examples (only explored in one dimension:)
t = np.power(x[0,:], 2) + np.power(x[1,:], 4)
#We could also take into account different directions:
#https://stackoverflow.com/questions/49302035/plot-paraboloid-surface-fitting

#Create a new figure:
fig = plt.figure()
#Create the subplot:
ax = fig.add_subplot(111, projection='3d')
#Plot the data:
ax.scatter(x[0,:], x[1,:], t, color='b')
ax.plot(x[0,:], x[1,:], t, color='g')
#Draw the plot:
#plt.show()

#In order to make more interesting plots, we are going to do the same thing that appears in the link mentioned in line 17:
u, v = np.meshgrid(x[0,:], x[1,:])
#Repeat the procedure:
z = np.power(u, 2) + np.power(v, 4) #All these operations are fortunately element-wise :)
ax.plot_wireframe(u, v, z) #See https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html
#ax.scatter(u, v, z, color='b')
plt.show() #See the difference after using meshgrid? :)

#Anyways, let's continue implementing the gradient descent:
#In this case the formula for the derivative is simple:

wnow = np.array([20, -15])
for i in range(0, 300):
    gradWnow = np.array([2*wnow[0], 4*wnow[1]])
    wnext = wnow - lamb * gradWnow
    print(wnext)
    wnow = wnext
