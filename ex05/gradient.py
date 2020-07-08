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


if __name__ == "__main__":
    x = np.array([
    [ -6, -7, -9],
    [ 13, -2, 14],
    [ -7, 14, -1],
    [ -8, -4, 6],
    [ -5, -9, 6],
    [ 1, -5, 11],
    [ 9, -11, 8]])
    y = np.array([2, 14, -13, 5, 12, 4, -19])
    theta1 = np.array([0, 3, 0.5, -6])
    # Example :
    print(gradient(x, y, theta1))
    # Output:
    # array([ -37.35714286, 183.14285714, -393. ])
    # Example :
    theta2 = np.array([0,0,0,0])
    print(gradient(x, y, theta2))
    # Output:
    # array([ 0.85714286, 23.28571429, -26.42857143])