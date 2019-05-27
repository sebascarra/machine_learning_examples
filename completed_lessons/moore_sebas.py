
"""
Sebastian Carrazzoni's implementation of Moore's law demonstration using linear regression.
"""

# Built-in/Generic Imports
import re
import argparse
import time

# Libs
import numpy as np
import matplotlib.pyplot as plt

__author__ = 'Sebastian Carrazzoni'
__email__ = 'sebascarra@gmail.com'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--draw", help="Whether to draw plots or not (1 means draw -default-, 0 means do not draw)")
    args = vars(parser.parse_args())
    
    x = []
    y = []

    non_decimal = re.compile(r'[^\d]+')

    #amount_of_transistors_regex = re.compile(r'\d+,\d+')
    #year_regex = re.compile(r'\d+,\d+')

    for line in open("../linear_regression_class/moore.csv"):
        r = line.split('\t')
        x.append(int(non_decimal.sub('', r[2].split('[')[0])))
        y.append(int(non_decimal.sub('', r[1].split('[')[0])))
        

    #Convert x and y to numpy arrays:
    x = np.array(x)
    y = np.array(y)

    # #Plot x and y (both original values and in log scale):
    # # log = natural logarithm in numpy
    if args["draw"] is "1":
        plt.scatter(x, y)
        plt.show()

    #Let's time our code from here:
    start = time.time()

    #Start to work with log(y):
    y = np.log(y)

    # #Calculate yhat = a * x + b (we assume that yhat is log(nthat) where nthat = exponential regression of the number of transistors)
    denominator = x.dot(x) - x.sum() * x.mean()

    a = (x.dot(y) - x.sum() * y.mean()) / denominator
    b = (y.mean() * x.dot(x) - x.mean() * x.dot(y)) / denominator

    # #The linear regression is:
    yhat = a * x + b

    # #Compare results:
    if args["draw"] is "1":
        plt.scatter(x, y)
        plt.plot(x, yhat) #Plots joined by lines in an orderly fashion.
        plt.show()

    # #Calculate R2:
    d1 = y - yhat
    d2 = y - y.mean()
    
    r2 = 1 - d1.dot(d1) / d2.dot(d2)

    #Let's check how much time has passed:
    end = time.time()

    # #Print out the results:
    print("The linear regression model characteristic values are a =", a, " and b =", b)
    print("The R2 value of the regression is ", r2)
    print("The time it takes for the number of transistors to double is", np.log(2) / a)
    print("(Not working because execution takes less than 0.1s-->) Execution took", end - start, "s") #Check: https://stackoverflow.com/questions/1557571/how-do-i-get-time-of-a-python-programs-execution




