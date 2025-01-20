import torch
from abc import ABC, abstractmethod

import torch.linalg as la


class Optimizer(ABC):
    """
    Convex smooth optimization
    """

    def __init__(
        self,
        oracle,
        x_init: torch.Tensor,
        projection,
        n_iter: int,
        tol: float,
        **kwargs
    ):
        self.oracle = oracle
        self.n_iter = n_iter
        self.tol = tol
        self._x = x_init.clone()
        self.projection = projection
        self._f = self.oracle.f(self._x)
        self._grad = self.oracle.df(self._x).clone()
        self.early_stop = False

    def set_oracle(self, oracle):
        self.oracle = oracle

    def set_x(self, x):
        self._x = x.clone()

    @abstractmethod
    def step(self, t: int = 1):
        """
        Single step of the optimization algorithm.
        """
        raise NotImplemented

    def solve(self, n_iter: int = -1, verbose: int = 0):
        """
        Optimizers until `n_iter` iterations or gradient tolerance is met,
        whichever is first. Returns last iterate and its gradient.
        """
        n_iter = self.n_iter if n_iter <= 0 else n_iter
        f_arr = torch.zeros(n_iter, dtype=torch.float64)
        grad_arr = torch.zeros(n_iter, dtype=torch.float64)
        t = 1
        while t <= n_iter:
            self.step(t)
            f_arr[t - 1] = self._f
            grad_arr[t - 1] = la.norm(self._grad)
            if (la.norm(self._grad) <= self.tol and t >= 10) or self.early_stop:
                # print(f"Early termination at iteration {t+1}/{n_iter}")
                f_arr = f_arr[:t]
                break
            t += 1

        return self._x.clone(), f_arr[:t], grad_arr[:t]


class GradientDescent(Optimizer):
    def __init__(
        self,
        oracle,
        x_init: torch.Tensor,
        n_iter: int = 100,
        tol: float = 1e-10,
        stepsize: str = "constant",
        eta_0: float = 0.001,
        **kwargs
    ):
        super().__init__(oracle, x_init, n_iter, tol, *kwargs)
        self.stepsize = stepsize
        self.eta_0 = eta_0

    def step(self, t: int = 1):
        eta = self.eta_0
        if not (self.stepsize == "constant"):
            eta = self.eta_0 * torch.sqrt(1.0 / t)
        self._x -= eta * self._grad
        self._f = self.oracle.f(self._x)
        self._grad = self.oracle.df(self._x)
