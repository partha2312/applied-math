import numpy as np


def get_rosenbrock(d):
    def f(x):
        #print(x)
        coupled_term = 100 * (x[1:] ** 2 - x[:-1]) ** 2
        diagonal_term = (x - 1.) ** 2
        return np.sum(coupled_term) + np.sum(diagonal_term)

    def gradf(x):
        grad = 2.0 * (x - 1.0)
        diffs = x[1:] ** 2 - x[:-1]
        grad[:-1] += -200.0 * diffs
        grad[1:] += 400.0 * x[1:] * diffs
        return grad

    def hessf(x):
        diag_main = np.full(d, 2.0)
        diag_main[:-1] += 200.0
        diag_main[1:] += 1200.0 * x[1:] ** 2 - 400.0 * x[:-1]
        off_diag = np.zeros(d - 1)
        off_diag = -400.0 * x[1:]
        H = (
                np.diag(diag_main) +
                np.diag(off_diag, k=1) +
                np.diag(off_diag, k=-1)
        )
        return H

    x0 = -1 * np.ones(d)

    return f, gradf, hessf, x0


#MNIST Logistic regression
mnist_data = np.load('mnist01.npy', allow_pickle=True)
#
A_lgt = mnist_data[0]
b_lgt = mnist_data[1]
x0 = np.zeros(A_lgt.shape[1])


def get_lgt_obj(lam_lgt):
    # define function, gradient and Hessian
    def lgt_func(x):
        y = A_lgt.dot(x)
        return np.sum(np.log(1.0 + np.exp(y))) - b_lgt.dot(y) + 0.5 * lam_lgt * np.sum(x ** 2)

    #
    def lgt_grad(x):
        y = A_lgt.dot(x)
        z = 1.0 / (1.0 + np.exp(-y)) - b_lgt
        return A_lgt.T.dot(z) + lam_lgt * x

    #
    def lgt_hess(x):
        y = A_lgt.dot(x)
        z = np.exp(-y) / (1.0 + np.exp(-y)) ** 2
        return A_lgt.T.dot(np.diag(z).dot(A_lgt)) + lam_lgt * np.eye(x.size)

    return lgt_func, lgt_grad, lgt_hess, x0


def get_zakharov(x):
    def f(x):
        x = np.asarray(x)
        n = len(x)
        c = np.arange(1, n + 1) / 2
        s = np.dot(c, x)
        return np.sum(x ** 2) + s ** 2 + s ** 4

    def gradf(x):
        x = np.asarray(x)
        n = len(x)
        c = np.arange(1, n + 1) / 2
        s = np.dot(c, x)
        return 2 * x + s * c + 2 * s ** 3 * c

    def hessf(x):
        x = np.asarray(x)
        n = len(x)
        c = np.arange(1, n + 1) / 2  # (1/2, 2/2, ..., n/2)
        s = np.dot(c, x)  # Compute s = sum(i/2 * x_i)
        return 2 * np.eye(n) + (0.5 + 3 * s ** 2) * np.outer(c, c)  # Hessian formula

    x0 = -1 * np.ones(x)

    return f, gradf, hessf, x0


def get_mccormick(d):
    def f(x):
        x = np.array(x)
        return np.sum(np.sin(x) + np.sin(10 * x))

    def gradf(x):
        x = np.array(x)
        return np.cos(x) + 10 * np.cos(10 * x)

    def hessf(x):
        x = np.array(x)
        diag = -np.sin(x) - 100 * np.sin(10 * x)  # Second derivatives for each variable
        return np.diag(diag)

    x0 = -1 * np.ones(d)

    return f, gradf, hessf, x0


def get_chung_reynolds(d):
    def f(x):
        x = np.array(x)
        return np.sum((x ** 2) ** 2)

    def gradf(x):
        x = np.array(x)
        return 4 * x ** 3  # Derivative of x^4 is 4x^3

    def hessf(x):
        x = np.array(x)
        return np.diag(12 * x**2)

    x0 = -10 * np.ones(d)

    return f, gradf, hessf, x0

def get_trid(d):

    def f(x):
        d = len(x)
        term1 = np.sum((x - 1) ** 2)
        term2 = np.sum(x[1:] * x[:-1])
        return term1 - term2

    def gradf(x):
        d = len(x)
        grad = np.zeros(d)

        grad[0] = 2 * (x[0] - 1) - x[1]  # First element
        for i in range(1, d - 1):
            grad[i] = 2 * (x[i] - 1) - x[i - 1] - x[i + 1]
        grad[-1] = 2 * (x[-1] - 1) - x[-2]  # Last element

        return grad

    def hessf(x):
        d = len(x)
        hessian = 2 * np.eye(d)  # Diagonal elements 2
        for i in range(d - 1):
            hessian[i, i + 1] = -1
            hessian[i + 1, i] = -1
        return hessian

    x0 = -1 * np.ones(d)

    return f, gradf, hessf, x0