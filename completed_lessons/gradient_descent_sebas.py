import numpy as np 
import matplotlib.pyplot as plt 

#Create some sample inputs:

N = 10 #Amount of sample input-output pairs
D = 3 #Although the rank of X is going to be 2.
learning_rate = 0.01
epochs = 50

X = np.zeros((N,D))
X[0:5, 0] = 1
X[5:10, 2] = 1
X[:, 1] = 1
#print(X)
#Some targets:
#T = X[:,0] * 4 + X[:,1] * 2 + X[:,2]
#T = T.reshape(N,1)
T = np.zeros((N,1))
T[0:5] = 1 

#Regular solution will fail (verified that it fails):
#w = np.linalg.solve(X.T.dot(X), X.T.dot(T))

#Let's find the w using gradient descent:

#D = np.linalg.matrix_rank(X) #In case we wanted to use X's rank (= 2)
sigma = 1 / np.sqrt(D)
wnow = sigma * np.random.randn() * np.ones((3,1))
# wnow = np.random.randn(D) * sigma -> better
# wnow = wnow.reshape(D,1)

error = []

for i in range(epochs):
    delta = X.dot(wnow) - T
    wnext = wnow - learning_rate * X.T.dot(delta)
    error.append(np.ndarray.item(delta.T.dot(delta))) #The tutorial divides/scales error/cost by N but I don't think that's necessary.
    wnow = wnext

#Get the linear regression:
Y = X.dot(wnow)

#Let's plot the error and compare Y and T:
fig = plt.figure()
ax1 = fig.add_subplot('211')
ax1.plot(range(epochs), error)
ax2 = fig.add_subplot('212') #See: https://matplotlib.org/3.1.0/gallery/subplots_axes_and_figures/subplot.html
ax2.plot(T)
ax2.plot(Y)
plt.show()