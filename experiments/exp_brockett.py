import os
import csv
import autograd.numpy as np
from sklearn.datasets import make_spd_matrix

import pymanopt
from pymanopt.manifolds import Stiefel

from algorithms import ConjugateGradient, BetaTypes


def create_cost(matrix, dmatrix):
    @pymanopt.function.Autograd
    def cost(X):
        return np.trace(X.T @ matrix @ X @ dmatrix)

    return cost


if __name__ == "__main__":
    experiment_name = 'brockett'
    n_exp = 10

    if not os.path.isdir('result'):
        os.makedirs('result')
    path = os.path.join('result', experiment_name + '.csv')

    m = 20
    n = 5
    A = make_spd_matrix(m)
    N = np.diag([i for i in range(n)])

    cost = create_cost(A, N)
    manifold = Stiefel(m, n)
    problem = pymanopt.Problem(manifold, cost, egrad=None)

    for i in range(n_exp):
        res_list = []

        for beta_type in BetaTypes:
            solver = ConjugateGradient(beta_type=beta_type, maxiter=10000)
            res = solver.solve(problem)
            res_list.append(res[1])
            res_list.append(res[2])

        with open(path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(res_list)

