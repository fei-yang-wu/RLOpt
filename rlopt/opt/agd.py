import torch

from .gd import Optimizer


class AcceleratedGradientDescent(Optimizer):
    def __init__(
        self, oracle, x_init, n_iter=100, tol=1e-10, eta_0=0.001, mu=0, **kwargs
    ):
        super().__init__(oracle, x_init, n_iter, tol, *kwargs)
        self._xt = torch.clone(self._x)
        self._uxt = torch.clone(self._x)
        self._oxt = torch.clone(self._x)
        self.eta_0 = eta_0
        self.mu = mu

    def step(self, t=1):
        a_t = q_t = 2.0 / (t + 1)
        # divide by /2 is important here
        eta_t = t * self.eta_0 / 2

        self._uxt = (1 - q_t) * self._oxt + q_t * self._xt
        grad = self.oracle.df(self._uxt)
        self._xt = (
            1.0
            / (1.0 + self.mu * eta_t)
            * (self._xt + self.mu * eta_t * self._uxt - eta_t * grad)
        )
        self._oxt = (1 - a_t) * self._oxt + a_t * self._xt

        self._x = self._oxt
        self._f = self.oracle.f(self._x)
        self._grad = self.oracle.df(self._x)
