import os
import csv
import autograd.numpy as np
from numpy import linalg as la, random as rnd

import pymanopt
from pymanopt.manifolds import Oblique

from algorithms import ConjugateGradient, BetaTypes


def create_cost(matrices):
    def cost(X):
        _sum = 0.
        for matrix in matrices:
            Y = X.T @ matrix @ X
            _sum += np.linalg.norm(Y - np.diag(np.diag(Y))) ** 2
        return _sum

    return cost


if __name__ == "__main__":
    experiment_name = 'off-diag'
    n_exp = 10

    if not os.path.isdir('result'):
        os.makedirs('result')
    path = os.path.join('result', experiment_name + '.csv')

    N = 5
    n = 10
    p = 5
    
    for i in range(n_exp):

        matrices = []
        for k in range(N):
            B = rnd.randn(n, n)
            C = (B + B.T) / 2
            matrices.append(C)

        cost = create_cost(matrices)
        manifold = Oblique(n, p)
        problem = pymanopt.Problem(manifold, cost=cost, egrad=None)
        
        res_list = []

        for beta_type in BetaTypes:
            solver = ConjugateGradient(beta_type=beta_type, maxiter=10000)
            res = solver.solve(problem)
            res_list.append(res[1])
            res_list.append(res[2])

        with open(path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(res_list)
