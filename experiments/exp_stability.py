import os
import csv
import networkx as nx
import autograd.numpy as np
from numpy import linalg as la, random as rnd

import pymanopt
from pymanopt.manifolds import Sphere

from algorithms import ConjugateGradient, BetaTypes


def create_cost(G):
    @pymanopt.function.Autograd
    def cost(x):
        _sum = np.sum(x ** 4)
        for e in G.edges:
            _sum += 2 * (x[e[0]] ** 2) * (x[e[1]] ** 2)
        return _sum

    return cost


if __name__ == "__main__":
    experiment_name = 'stability'
    n_exp = 10

    if not os.path.isdir('result'):
        os.makedirs('result')
    path = os.path.join('result', experiment_name + '.csv')

    n = 20
    p = 1 / 4

    for i in range(n_exp):
        G = nx.fast_gnp_random_graph(n, p, directed=False)
        cost = create_cost(G)
        manifold = Sphere(len(G.nodes))
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
