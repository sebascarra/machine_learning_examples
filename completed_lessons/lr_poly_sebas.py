import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Load the data to be worked on:

X = []
T = []

for line in open('..\linear_regression_class/data_poly.csv'):
    rawdata = line.split(',')
    x, t = float(rawdata[0]), float(rawdata[1])
    x = [np.square(x), x, 1]
    X.append(x)
    T.append(t)

X = np.array(X)
T = np.array(T)

#Plot the data:

#Create a figure object:
fig = plt.figure()
#Create a subplot and get its axes:
ax = fig.add_subplot(111) #For example, "111" means "1x1 grid (of plots), first subplot" and "234" means "2x3 grid, 4th subplot". Must be a 3-digit number.
#Scatter plot the data:
ax.scatter(X[:,1], T)
#Draw the plot:
#plt.show()

#Let's do a 3D plot by re-using info in X=X1 as X2 (just to practise drawing plots)
# fig2 = plt.figure()
# ax2 = fig2.add_subplot(111, projection="3d")
# ax2.scatter(X, X, T)
# plt.show()

#Calculate w:
w = np.linalg.solve(X.T.dot(X), X.T.dot(T))

print(w)

#We can plot this one properly:
#Calculate Y:
Y = X.dot(w)
ax.plot(np.sort(X[:,1]), np.sort(Y))

#For practice purposes, apply the formula directly:
#x_line = np.linspace(X[:,1].min(), X[:,1].max())
#y_line = w[0] * x_line * x_line + w[1] * x_line + w[2] #Element-wise multiplication is very handy here.
#ax.plot(x_line, y_line)

#Calculate r1:
d1 = T - Y
d2 = T - np.mean(T)

r2 = 1 - d1.dot(d1) / d2.dot(d2)

print(r2)

plt.show()

