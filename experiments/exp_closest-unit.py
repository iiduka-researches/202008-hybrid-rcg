import os
import csv
import autograd.numpy as np
from numpy import linalg as la, random as rnd

import pymanopt
from pymanopt.manifolds import Oblique

from algorithms import ConjugateGradient, BetaTypes


def create_cost(A):
    @pymanopt.function.Autograd
    def cost(X):
        return np.sum((X - A) ** 2)

    return cost


if __name__ == "__main__":
    experiment_name = 'closest-unit'
    n_exp = 10

    if not os.path.isdir('result'):
        os.makedirs('result')
    path = os.path.join('result', experiment_name + '.csv')

    m = 10
    n = 1000
    
    for i in range(n_exp):
        matrix = rnd.randn(m, n)

        cost = create_cost(matrix)
        manifold = Oblique(m, n)
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
