import numpy as np
import matplotlib.pyplot as plt

N = 50 #amount of samples
mu = 0
sigma = 1
lamb = 1000

X = np.linspace(0, 10, N) #Sample inputs
T = sigma * np.random.randn(N) + mu + 0.5 * X #Sample outputs/targets

#Corrupt data:
T[-1] += 30
T[-2] += 30

#Create a figure object:
fig = plt.figure()
#Add a subplot and get its axes:
ax = fig.add_subplot(111)
#Plot the data:
ax.scatter(X, T)
#Draw the plot on screen:
#plt.show()

#Convert to usual X matrix adding bias term:
X = np.vstack([X, np.ones(N)]).T #This means that a matrix will be created, where each column (hence why we .T afterwards) will be composed of one element of each of the 1-D arrays passed as arguments.

# 'vstack' stacks arrays in sequence vertically (row wise):
# >>> a = np.array([1, 2, 3])
# >>> b = np.array([2, 3, 4])
# >>> np.vstack((a,b)) -> vstack can receive a tuple of arrays or an array of arrays, such as [a ,b]
# array([[1, 2, 3],
#        [2, 3, 4]])

#Find out old w (simple linear regression):
w = np.linalg.solve(X.T.dot(X), X.T.dot(T))
Y = X.dot(w)
#Plot the data:
ax.plot(X[:,0], Y)
ax.plot(X[:,0], np.ones(N) * np.mean(T))
#Draw the plot on screen:
#plt.show()

#Calculate r2 for this case:
d1 = Y - T
d2 = Y - np.mean(T)

r2 = 1 - d1.dot(d1) / d2.dot(d2)
print("Usual linear regularization r-squared sucks:", r2, " (<0 so you're better off using just the mean of the samples)")

#Find out w with L2 regularization:
w = np.linalg.solve(X.T.dot(X) + lamb * np.eye(2), X.T.dot(T))
Y = X.dot(w)
#Plot the data:
ax.plot(X[:,0], Y)
ax.plot(X[:,0], np.ones(N) * np.mean(T))
#Draw the plot on screen:
plt.show()

#Calculate r2 for this case:
d1 = Y - T
d2 = Y - np.mean(T)

r2 = 1 - d1.dot(d1) / d2.dot(d2)
print("With L2 reg:", r2, " (It's even worse, but much worse is considering plain wrong data as a valid sample output.")

