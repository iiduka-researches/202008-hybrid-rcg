import os
import csv
from sklearn.datasets import make_spd_matrix
import autograd.numpy as np
from numpy import linalg as la, random as rnd

import pymanopt
from pymanopt.manifolds import Sphere

from algorithms import ConjugateGradient, BetaTypes


def create_cost(A):
    @pymanopt.function.Autograd
    def cost(x):
        return 0.1 * np.inner(x, A @ x)

    return cost


if __name__ == "__main__":
    experiment_name = 'rayleigh'
    n_exp = 10

    if not os.path.isdir('result'):
        os.makedirs('result')
    path = os.path.join('result', experiment_name + '.csv')

    n = 100
    
    for i in range(n_exp):
        matrix = make_spd_matrix(n)

        cost = create_cost(matrix)
        manifold = Sphere(n)
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