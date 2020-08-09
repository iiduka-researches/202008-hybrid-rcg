import time
from copy import deepcopy

import numpy as np

from pymanopt import tools
from pymanopt.solvers.solver import Solver


# BetaTypes of the conjugate gradient method in pymanopt was changed.
BetaTypes = tools.make_enum("BetaTypes", "DaiYuan PolakRibiere Hybrid1 Hybrid2".split())


class ConjugateGradient(Solver):
    """
    Module containing conjugate gradient algorithm based on
    conjugategradient.m from the manopt MATLAB package.
    """

    def __init__(self, beta_type=BetaTypes.DaiYuan, orth_value=np.inf, linesearch=None, *args, **kwargs):
        """
        Instantiate gradient solver class.
        Variable attributes (defaults in brackets):
            - beta_type (BetaTypes.HestenesStiefel)
                Conjugate gradient beta rule used to construct the new search
                direction
            - orth_value (numpy.inf)
                Parameter for Powell's restart strategy. An infinite
                value disables this strategy. See in code formula for
                the specific criterion used.
            - linesearch (LineSearchWolfe)
                The linesearch method to used.
        """
        super().__init__(*args, **kwargs)

        self._beta_type = beta_type
        self._orth_value = orth_value

        if linesearch is None:
            self._linesearch = LineSearchWolfe()
        else:
            self._linesearch = linesearch
        self.linesearch = None

    def solve(self, problem, x=None, reuselinesearch=False):
        """
        Perform optimization using nonlinear conjugate gradient method with
        linesearch.
        This method first computes the gradient of obj w.r.t. arg, and then
        optimizes by moving in a direction that is conjugate to all previous
        search directions.
        Arguments:
            - problem
                Pymanopt problem setup using the Problem class, this must
                have a .manifold attribute specifying the manifold to optimize
                over, as well as a cost and enough information to compute
                the gradient of that cost.
            - x=None
                Optional parameter. Starting point on the manifold. If none
                then a starting point will be randomly generated.
            - reuselinesearch=False
                Whether to reuse the previous linesearch object. Allows to
                use information from a previous solve run.
        Returns:
            - x
                Local minimum of obj, or if algorithm terminated before
                convergence x will be the point at which it terminated.
        """
        man = problem.manifold
        verbosity = problem.verbosity
        objective = problem.cost
        gradient = problem.grad

        if not reuselinesearch or self.linesearch is None:
            self.linesearch = deepcopy(self._linesearch)
        linesearch = self.linesearch

        # If no starting point is specified, generate one at random.
        if x is None:
            x = man.rand()

        # Initialize iteration counter and timer
        iter = 0
        stepsize = np.nan
        time0 = time.time()

        if verbosity >= 1:
            print("Optimizing...")
        if verbosity >= 2:
            print(" iter\t\t   cost val\t    grad. norm")

        # Calculate initial cost-related quantities
        cost = objective(x)
        grad = gradient(x)
        gradnorm = man.norm(x, grad)
        def _Pgrad(_x):
            return problem.precon(_x, gradient(_x))
        Pgrad = problem.precon(x, grad)
        gradPgrad = man.inner(x, grad, Pgrad)

        # Initial descent direction is the negative gradient
        desc_dir = -Pgrad

        self._start_optlog(extraiterfields=['gradnorm'],
                           solverparams={'beta_type': self._beta_type,
                                         'orth_value': self._orth_value,
                                         'linesearcher': linesearch})

        while True:
            if verbosity >= 2:
                print("%5d\t%+.16e\t%.8e" % (iter, cost, gradnorm))

            if self._logverbosity >= 2:
                self._append_optlog(iter, x, cost, gradnorm=gradnorm)

            stop_reason = self._check_stopping_criterion(time0, gradnorm=gradnorm, iter=iter + 1, stepsize=stepsize)

            if stop_reason:
                if verbosity >= 1:
                    print(stop_reason)
                    print('')
                break

            # The line search algorithms require the directional derivative of
            # the cost at the current point x along the search direction.
            df0 = man.inner(x, grad, desc_dir)

            # If we didn't get a descent direction: restart, i.e., switch to
            # the negative gradient. Equivalent to resetting the CG direction
            # to a steepest descent step, which discards the past information.
            if df0 >= 0:
                # Or we switch to the negative gradient direction.
                if verbosity >= 3:
                    print("Conjugate gradient info: got an ascent direction "
                          "(df0 = %.2f), reset to the (preconditioned) "
                          "steepest descent direction." % df0)
                # Reset to negative gradient: this discards the CG memory.
                desc_dir = -Pgrad
                df0 = -gradPgrad

            # Execute line search
            stepsize, newx = linesearch.search(objective, man, x, desc_dir, cost, df0, _Pgrad)

            # Compute the new cost-related quantities for newx
            newcost = objective(newx)
            newgrad = gradient(newx)
            newgradnorm = man.norm(newx, newgrad)
            Pnewgrad = problem.precon(newx, newgrad)
            newgradPnewgrad = man.inner(newx, newgrad, Pnewgrad)

            # Apply the CG scheme to compute the next search direction
            oldgrad = man.transp(x, newx, grad)
            orth_grads = man.inner(newx, oldgrad, Pnewgrad) / newgradPnewgrad

            # Powell's restart strategy (see page 12 of Hager and Zhang's
            # survey on conjugate gradient methods, for example)
            if abs(orth_grads) >= self._orth_value:
                beta = 0
                desc_dir = -Pnewgrad
            else:
                desc_dir = man.transp(x, newx, desc_dir)
                if self._beta_type == BetaTypes.DaiYuan:
                    diff = newgrad - oldgrad
                    beta = newgradPnewgrad / man.inner(newx, diff, desc_dir)
                elif self._beta_type == BetaTypes.PolakRibiere:
                    diff = newgrad - oldgrad
                    ip_diff = man.inner(newx, Pnewgrad, diff)
                    beta = ip_diff / gradPgrad
                elif self._beta_type == BetaTypes.Hybrid1:
                    diff = newgrad - oldgrad
                    beta_DY = newgradPnewgrad / man.inner(newx, diff, desc_dir)
                    ip_diff = man.inner(newx, Pnewgrad, diff)
                    try:
                        beta_HS = ip_diff / man.inner(newx, diff, desc_dir)
                    except ZeroDivisionError:
                        beta_HS = 1
                    beta = max(0, min(beta_DY, beta_HS))
                elif self._beta_type == BetaTypes.Hybrid2:
                    diff = newgrad - oldgrad
                    beta_DY = newgradPnewgrad / man.inner(newx, diff, desc_dir)
                    ip_diff = man.inner(newx, Pnewgrad, diff)
                    try:
                        beta_HS = ip_diff / man.inner(newx, diff, desc_dir)
                    except ZeroDivisionError:
                        beta_HS = 1
                    c2 = linesearch.c2
                    beta = max(-(1 - c2) / (1 + c2) * beta_DY, min(beta_DY, beta_HS))
                else:
                    types = ", ".join(["BetaTypes.%s" % t for t in BetaTypes._fields])
                    raise ValueError("Unknown beta_type %s. Should be one of %s." % (self._beta_type, types))

                desc_dir = -Pnewgrad + beta * desc_dir

            # Update the necessary variables for the next iteration.
            x = newx
            cost = newcost
            grad = newgrad
            Pgrad = Pnewgrad
            gradnorm = newgradnorm
            gradPgrad = newgradPnewgrad

            iter += 1
        
        return x, iter + 1, time.time() - time0


class LineSearchWolfe:
    def __init__(self, c1: float=1e-4, c2: float=0.9):
        self.c1 = c1
        self.c2 = c2

    def __str__(self):
        return 'Wolfe'

    def search(self, objective, man, x, d, f0, df0, gradient):
        '''
        Returns the step size that satisfies the strong Wolfe condition.
        Scipy.optimize.line_search in SciPy v1.4.1 modified to Riemannian manifold.

        ----------
        References
        ----------
        [1] SciPy v1.4.1 Reference Guide, https://docs.scipy.org/
        '''
        fc = [0]
        gc = [0]
        gval = [None]
        gval_alpha = [None]

        def phi(alpha):
            fc[0] += 1
            return objective(man.retr(x, alpha * d))

        def derphi(alpha):
            newx = man.retr(x, alpha * d)
            newd = man.transp(x, newx, d)
            gc[0] += 1
            gval[0] = gradient(newx)  # store for later use
            gval_alpha[0] = alpha
            return man.inner(newx, gval[0], newd)

        gfk = gradient(x)
        derphi0 = man.inner(x, gfk, d)

        stepsize = _scalar_search_wolfe(phi, derphi, self.c1, self.c2, maxiter=100)
        if stepsize is None:
            stepsize = 1e-6
        
        newx = man.retr(x, stepsize * d)
        
        return stepsize, newx


def _scalar_search_wolfe(phi, derphi, c1=1e-4, c2=0.9, maxiter=100):
    phi0 = phi(0.)
    derphi0 = derphi(0.)
    alpha0 = 0
    alpha1 = 1.0
    phi_a1 = phi(alpha1)
    phi_a0 = phi0
    derphi_a0 = derphi0
    for i in range(maxiter):
        if (phi_a1 > phi0 + c1 * alpha1 * derphi0) or ((phi_a1 >= phi_a0) and (i > 1)):
            alpha_star, phi_star, derphi_star = _zoom(alpha0, alpha1, phi_a0, phi_a1, derphi_a0, phi, derphi, phi0, derphi0, c1, c2)
            break

        derphi_a1 = derphi(alpha1)
        if (abs(derphi_a1) <= c2 * abs(derphi0)):
            alpha_star = alpha1
            phi_star = phi_a1
            derphi_star = derphi_a1
            break

        if (derphi_a1 >= 0):
            alpha_star, phi_star, derphi_star = _zoom(alpha1, alpha0, phi_a1, phi_a0, derphi_a1, phi, derphi, phi0, derphi0, c1, c2)
            break

        alpha2 = 2 * alpha1  # increase by factor of two on each iteration
        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi(alpha1)
        derphi_a0 = derphi_a1
    else:
        # stopping test maxiter reached
        alpha_star = alpha1
        phi_star = phi_a1
        derphi_star = None
        print('The line search algorithm did not converge')
    
    return alpha_star


def _zoom(a_lo, a_hi, phi_lo, phi_hi, derphi_lo, phi, derphi, phi0, derphi0, c1, c2):
    """
    Part of the optimization algorithm in `_scalar_search_wolfe`.
    """
    maxiter = 10
    i = 0
    delta1 = 0.2  # cubic interpolant check
    delta2 = 0.1  # quadratic interpolant check
    phi_rec = phi0
    a_rec = 0
    while True:
        dalpha = a_hi - a_lo
        if dalpha < 0:
            a, b = a_hi, a_lo
        else:
            a, b = a_lo, a_hi
        if (i > 0):
            cchk = delta1 * dalpha
            a_j = _cubicmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi, a_rec, phi_rec)
        if (i == 0) or (a_j is None) or (a_j > b - cchk) or (a_j < a + cchk):
            qchk = delta2 * dalpha
            a_j = _quadmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi)
            if (a_j is None) or (a_j > b-qchk) or (a_j < a+qchk):
                a_j = a_lo + 0.5*dalpha
        phi_aj = phi(a_j)
        if (phi_aj > phi0 + c1*a_j*derphi0) or (phi_aj >= phi_lo):
            phi_rec = phi_hi
            a_rec = a_hi
            a_hi = a_j
            phi_hi = phi_aj
        else:
            derphi_aj = derphi(a_j)
            if abs(derphi_aj) <= c2 * abs(derphi0):
                a_star = a_j
                val_star = phi_aj
                valprime_star = derphi_aj
                break
            if derphi_aj*(a_hi - a_lo) >= 0:
                phi_rec = phi_hi
                a_rec = a_hi
                a_hi = a_lo
                phi_hi = phi_lo
            else:
                phi_rec = phi_lo
                a_rec = a_lo
            a_lo = a_j
            phi_lo = phi_aj
            derphi_lo = derphi_aj
        i += 1
        if (i > maxiter):
            # Failed to find a conforming step size
            a_star = None
            val_star = None
            valprime_star = None
            break
    return a_star, val_star, valprime_star


def _cubicmin(a, fa, fpa, b, fb, c, fc):
    # f(x) = A *(x-a)^3 + B*(x-a)^2 + C*(x-a) + D
    with np.errstate(divide='raise', over='raise', invalid='raise'):
        try:
            C = fpa
            db = b - a
            dc = c - a
            denom = (db * dc) ** 2 * (db - dc)
            d1 = np.empty((2, 2))
            d1[0, 0] = dc ** 2
            d1[0, 1] = -db ** 2
            d1[1, 0] = -dc ** 3
            d1[1, 1] = db ** 3
            [A, B] = np.dot(d1, np.asarray([fb - fa - C * db,
                                            fc - fa - C * dc]).flatten())
            A /= denom
            B /= denom
            radical = B * B - 3 * A * C
            xmin = a + (-B + np.sqrt(radical)) / (3 * A)
        except ArithmeticError:
            return None
    if not np.isfinite(xmin):
        return None
    return xmin


def _quadmin(a, fa, fpa, b, fb):
    # f(x) = B*(x-a)^2 + C*(x-a) + D
    with np.errstate(divide='raise', over='raise', invalid='raise'):
        try:
            D = fa
            C = fpa
            db = b - a * 1.0
            B = (fb - D - C * db) / (db * db)
            xmin = a - C / (2.0 * B)
        except ArithmeticError:
            return None
    if not np.isfinite(xmin):
        return None
    return xmin