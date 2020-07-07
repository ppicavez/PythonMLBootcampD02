import numpy as np

def gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.ndarray, without any for loop. The three arrays must have compatible dimensions.
    Args:
    x: has to be a numpy.ndarray, a matrix of dimension m * 1.
    y: has to be a numpy.ndarray, a vector of dimension m * 1.
    theta: has to be a numpy.ndarray, a 2 * 1 vector.
    Returns:
    The gradient as a numpy.ndarray, a vector of dimension 2 * 1.
    None if x, y, or theta is an empty numpy.ndarray.
    None if x, y and theta do not have compatible dimensions.
    Raises:
    This function should not raise any Exception.
    """
    
    if len(x) != len(y):
        return None

    X_prime = np.c_[np.ones((len(x), 1)), x]
    result = (X_prime.T @ (X_prime @ theta - y)) / len(x)
    return result
