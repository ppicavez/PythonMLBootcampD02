import numpy as np


def simple_predict(x, theta):
    X = np.c_[np.ones((len(x), 1)), x]
    return np.dot(X, theta)    



if __name__ == "__main__":
    x = np.arange(1,13).reshape((4,3))
    # Example 1:
    theta1 = np.array([5, 0, 0, 0])
    print(simple_predict(x, theta1))
    # Ouput:
    # array([5., 5., 5., 5.])
    # Do you understand why y_hat contains only 5's here?
    # Example 2:
    theta2 = np.array([0, 1, 0, 0])
    print(simple_predict(x, theta2))
    # Output:
    # array([ 1., 4., 7., 10.])
    # Do you understand why y_hat == x[:,0] here?
    # Example 3:
    theta3 = np.array([-1.5, 0.6, 2.3, 1.98])
    print(simple_predict(x, theta3))
    # Output:
    # array([ 9.64, 24.28, 38.92, 53.56])
    #Example 4:
    theta4 = np.array([-3, 1, 2, 3.5])
    print(simple_predict(x, theta4))
    # Output:
    # array([12.5, 32. , 51.5, 71. ])