import numpy as np
import time
import objectives
#import matplotlib.pyplot as plt


#Throughout this code,you may want to modify the structure of these functions to
#include additional parameters such as stopping tolerance or maximum iterations!

def bisection_linesearch(x, d, f, gradf, c0, c1, t0=1., max_iter=1000):
    """

    Parameters
    ----------
    x : Previous iterate
    d : Search direction
    f : Objective function
    gradf : Gradient of objective function
    c0 : Parameter for W1 condition (you may want to set a default)
    c1 : Parameter for W2 condition (you may want to set a default)
    t0 : Initial stepsize, by default 1.

    Returns
    -------
    alpha: accepted line search stepsize
    fval: function value at location of accepted step
    g: gradient at the location of accepted step
    
    Note: Returning fval and g can possibly save on extra computation (why!)
    """

    #Your code here
    def w1(t):
        return (f(x + t * d) - f(x)) / (t * np.dot(gradf(x), d))

    def w2(t):
        return np.dot(gradf(x + t * d), d) / np.dot(gradf(x), d)

    a, b, t = 0, None, t0
    for _ in range(max_iter):
        w1_val, w2_val = w1(t), w2(t)

        if w1_val >= c0 and w2_val <= c1:
            return t, f(x + t * d), gradf(x + t * d)

        if w1_val < c0:
            b = t
            t = (b + a) / 2
        elif w2_val > c1:
            a = t
            t = 2 * a if b is None else min(2 * a, (b + a) / 2)
        if b is not None and abs(b - a) < 1e-8:
            break

    # fallback
    return t0, f(x + t0 * d), gradf(x + t0 * d)


def steepest_descent(x0, f, gradf, c0, c1, t0, TOL=1e-6, max_iter=1000):
    """Steepest descent with weak Wolfe line bisection search

    Parameters
    ----------
    x0 : initialization
    f : objective function
    gradf : gradient of f
    c0 : Parameter for W1 condition (you may want to set a default)
    c1 : Parameter for W2 condition (you may want to set a default)
    t0 : Initial stepsize for line search
    TOL : tolerance level to target for gradient norm
    """
    function_history, cumulative_times, gradient_norms = [], [], []
    iter = 0
    x_sol = x0
    f_x, grad_f_x = f(x_sol), gradf(x_sol)
    while True:
        start_time = time.perf_counter()
        grad_norm = np.linalg.norm(grad_f_x)

        function_history.append(f_x)
        gradient_norms.append(grad_norm)

        if grad_norm <= TOL or np.isnan(grad_norm):
            break

        d = -grad_f_x
        alpha, f_x, grad_f_x = bisection_linesearch(x_sol, d, f, gradf, c0, c1, t0, max_iter ** 2)
        x_sol = x_sol + alpha * d

        cumulative_times.append(time.perf_counter() - start_time)

        iter += 1
        if iter >= max_iter:
            return None, function_history, cumulative_times, gradient_norms

    print(f"converged in {iter} steps")
    return x_sol, function_history, cumulative_times, gradient_norms


def DFP(x0, f, gradf, c0, c1, t0, TOL=1e-6, max_iter=1000):
    """DFP with weak Wolfe line bisection search

    Parameters
    ----------
    x0 : initialization
    f : objective function
    gradf : gradient of f
    c0 : Parameter for W1 condition (you may want to set a default)
    c1 : Parameter for W2 condition (you may want to set a default)
    t0 : Initial stepsize for line search (you may want to set a default)
    """
    function_history, cumulative_times, gradient_norms = [], [], []
    iter = 0
    x_sol = x0
    f_x, grad_f_x, b = f(x_sol), gradf(x_sol), np.eye(len(x_sol))
    while True:
        start_time = time.perf_counter()
        grad_norm = np.linalg.norm(grad_f_x)

        function_history.append(f_x)
        gradient_norms.append(grad_norm)

        if grad_norm <= TOL or np.isnan(grad_norm):
            break

        d = -np.dot(b, grad_f_x)
        old_x_sol, old_grad_f_x = x_sol, grad_f_x
        alpha, f_x, grad_f_x = bisection_linesearch(x_sol, d, f, gradf, c0, c1, t0, max_iter)
        x_sol = x_sol + alpha * d
        s = x_sol - old_x_sol
        y = grad_f_x - old_grad_f_x
        by = np.dot(b, y)
        b1 = np.outer(by, by) / np.dot(y, by)
        b2 = np.outer(s, s) / np.dot(s, y)
        b = b - b1 + b2

        cumulative_times.append(time.perf_counter() - start_time)

        iter += 1
        if iter >= max_iter:
            return None, function_history, cumulative_times, gradient_norms

    print(f"converged in {iter} steps")
    return x_sol, function_history, cumulative_times, gradient_norms


def BFGS(x0, f, gradf, c0, c1, t0, TOL=1e-6, max_iter=1000):
    """BFGS with weak Wolfe line bisection search

    Parameters
    ----------
    x0 : initialization
    f : objective function
    gradf : gradient of f
    c0 : Parameter for W1 condition (you may want to set a default)
    c1 : Parameter for W2 condition (you may want to set a default)
    t0 : Initial stepsize for line search (you may want to set a default)
    """
    function_history, cumulative_times, gradient_norms = [], [], []
    iter = 0
    x_sol = x0
    f_x, grad_f_x, b = f(x_sol), gradf(x_sol), np.eye(len(x_sol))
    while True:
        start_time = time.perf_counter()
        grad_norm = np.linalg.norm(grad_f_x)

        function_history.append(f_x)
        gradient_norms.append(grad_norm)

        if grad_norm <= TOL or np.isnan(grad_norm):
            break

        d = -np.dot(b, grad_f_x)
        old_x_sol, old_grad_f_x = x_sol, grad_f_x
        alpha, f_x, grad_f_x = bisection_linesearch(x_sol, d, f, gradf, c0, c1, t0, max_iter)
        x_sol = x_sol + alpha * d
        s = x_sol - old_x_sol
        y = grad_f_x - old_grad_f_x
        i = np.eye(len(x_sol)) - (np.outer(s, y) / np.dot(y, s))
        b1 = np.dot(i, np.dot(b, i.T))
        b2 = np.dot(s, np.transpose(s)) / np.dot(np.transpose(y), s)
        b = b1 + b2

        cumulative_times.append(time.perf_counter() - start_time)

        iter += 1
        if iter >= max_iter:
            return None, function_history, cumulative_times, gradient_norms

    print(f"converged in {iter} steps")
    return x_sol, function_history, cumulative_times, gradient_norms


def newton(x0, f, gradf, hessf, c0, c1, t0, TOL=1e-6, max_iter=1000):
    """Newton's method with weak Wolfe line bisection search

    Parameters
    ----------
    x0 : initialization
    f : objective function
    gradf : gradient of f
    hessf : hessian of f
    c0 : Parameter for W1 condition (you may want to set a default)
    c1 : Parameter for W2 condition (you may want to set a default)
    t0 : Initial stepsize for line search (you may want to set a default)
    """
    function_history, cumulative_times, gradient_norms = [], [], []
    iter = 0
    x_sol = x0
    f_x, grad_f_x = f(x_sol), gradf(x_sol)
    while True:
        start_time = time.perf_counter()
        grad_norm = np.linalg.norm(grad_f_x)

        function_history.append(f_x)
        gradient_norms.append(grad_norm)

        if grad_norm <= TOL or np.isnan(grad_norm):
            break

        hess_f_x = hessf(x_sol)
        hess_f_x_inv = np.linalg.inv(hess_f_x)
        d = -np.dot(hess_f_x_inv, grad_f_x)
        alpha, f_x, grad_f_x = bisection_linesearch(x_sol, d, f, gradf, c0, c1, t0, max_iter ** 2)
        x_sol = x_sol + alpha * d

        cumulative_times.append(time.perf_counter() - start_time)

        iter += 1
        if iter >= max_iter:
            return None, function_history, cumulative_times, gradient_norms

    print(f"converged in {iter} steps")
    return x_sol, function_history, cumulative_times, gradient_norms


def run_algorithm(obj_fn, param, c0, c1, t0, TOL, max_iter, algo_name):
    f, g, h, x0 = obj_fn(param)
    x_sol, function_history, cumulative_times, gradient_norms = None, None, None, None
    if algo_name == "steepest_descent":
        x_sol, function_history, cumulative_times, gradient_norms = (
            steepest_descent(x0, f, g, c0, c1, t0, TOL, max_iter))
    elif algo_name == "newton":
        x_sol, function_history, cumulative_times, gradient_norms = (
            newton(x0, f, g, h, c0, c1, t0, TOL, max_iter))
    elif algo_name == "dfp":
        x_sol, function_history, cumulative_times, gradient_norms = (
            DFP(x0, f, g, c0, c1, t0, TOL, max_iter))
    elif algo_name == "bfgs":
        x_sol, function_history, cumulative_times, gradient_norms = (
            BFGS(x0, f, g, c0, c1, t0, TOL, max_iter))
    if x_sol is not None and not np.isnan(x_sol[0]):
        print("converged")
        print(gradient_norms[len(gradient_norms) - 1])
    else:
        print("did not converge")
    print()


algos = {"steepest_descent": (0.1, 0.9)}
    #, "newton": (0.1, 0.9), "dfp": (0.15, 0.2), "bfgs": (0.1, 0.9)}
n = [2, 5, 10, 50, 100, 500, 1000]
l = [0.001, 0.01, 0.1]
t0 = 10
max_iter = 1000000
rosenbrock = False
logistic = False
zakharov = False
mccormick = False
reynolds = False
trid = True

for algo_name, c_vals in algos.items():
    c0 = c_vals[0]
    c1 = c_vals[1]
    if rosenbrock:
        TOL = 1e-6
        for i in n:
            print(f"running {algo_name} for rosenbrock with sample size {i} c0: {c0} c1: {c1} TOL: {TOL}")
            run_algorithm(objectives.get_rosenbrock, i, c0, c1, t0, TOL, max_iter, algo_name)
    if logistic:
        TOL = 1e-5
        for i in l:
            print(f"running {algo_name} for logistic with lambda {i} c0: {c0} c1: {c1} TOL: {TOL}")
            run_algorithm(objectives.get_lgt_obj, i, c0, c1, t0, TOL, max_iter, algo_name)
    if zakharov:
        TOL = 1e-12
        for i in n:
            print(f"running {algo_name} for zakharov with sample size {i} c0: {c0} c1: {c1} TOL: {TOL}")
            run_algorithm(objectives.get_zakharov, i, c0, c1, t0, TOL, max_iter, algo_name)
    if mccormick:
        TOL = 1e-6
        for i in n:
            print(f"running {algo_name} for mccormick with sample size {i} c0: {c0} c1: {c1} TOL: {TOL}")
            run_algorithm(objectives.get_mccormick, i, c0, c1, t0, TOL, max_iter, algo_name)
    if reynolds:
        TOL = 1e-6
        for i in n:
            print(f"running {algo_name} for chung reynolds with sample size {i} c0: {c0} c1: {c1} TOL: {TOL}")
            run_algorithm(objectives.get_chung_reynolds, i, c0, c1, t0, TOL, max_iter, algo_name)
    if trid:
        TOL = 1e-6
        for i in n:
            print(f"running {algo_name} for trid with sample size {i} c0: {c0} c1: {c1} TOL: {TOL}")
            run_algorithm(objectives.get_trid, i, c0, c1, t0, TOL, max_iter, algo_name)

'''
def plot(title, gradient_norms):
    iters = np.arange(len(gradient_norms))  # Iteration indices

    plt.figure(figsize=(8, 5))
    plt.plot(iters, gradient_norms, marker='o', linestyle='-', markersize=4, linewidth=1.5, label='Gradient Norm')

    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Gradient Norm', fontsize=12)
    plt.title(title, fontsize=14)

    plt.yscale('log')  # Log scale for better visualization
    plt.legend()

    plt.show()
'''