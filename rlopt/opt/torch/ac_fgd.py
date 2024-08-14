import torch
from .gd import Optimizer


class AutoConditionedFastGradidentDescent(Optimizer):
    def __init__(
        self,
        oracle,
        x_init: torch.Tensor,
        projection,
        alpha: float,
        n_iter: int = 100,
        tol: float = 1e-10,
        stop_nonconvex=False,
        **kwargs
    ):
        super().__init__(oracle, x_init, projection, n_iter, tol, **kwargs)
        assert 0 <= alpha <= 1.0

        self._z = self._x.clone()
        self._y = self._x.clone()
        self.beta = 1.0 - torch.sqrt(3) / 2
        self.alpha = alpha
        self.L_t = ...
        self.eta_t = ...
        self.stop_nonconvex = stop_nonconvex
        self._detected_nonconvex = False
        self._first_eta = kwargs.get("first_eta", -1)

    @property
    def detected_nonconvex(self):
        return self._detected_nonconvex

    @property
    def first_eta(self):
        return self._first_eta

    def step(self, t: int = 1):
        if t == 1:
            self._first_eta = self.eta_t = self.line_search_eta()
            self.tau_t = 0
        elif t == 2:
            # TODO: This is a hacky solution
            if self.L_t == 0:
                self.eta_t = 2 * (1 - self.beta) * self._first_eta
            else:
                self.eta_t = torch.min(
                    2 * (1 - self.beta) * self._first_eta, self.beta / (2 * self.L_t)
                )
            self.tau_prev_t = self.tau_t
            self.tau_t = 2
        else:
            min_a = (self.tau_prev_t + 1) / self.tau_t * self.eta_t
            min_b = min_a if self.L_t == 0 else self.beta * self.tau_t / (4 * self.L_t)
            self.eta_t = torch.min(min_a, min_b)
            self.tau_prev_t = self.tau_t
            self.tau_t = self.tau_prev_t + self.alpha / 2
            self.tau_t += (
                2
                * (1 - self.alpha)
                * self.eta_t
                * self.L_t
                / (self.beta * self.tau_prev_t)
            )

        self._z = self.projection(self._y - self.eta_t * self._grad)
        self._y = (1 - self.beta) * self._y + self.beta * self._z
        next_x = (self._z + self.tau_t * self._x) / (1.0 + self.tau_t)
        next_f = self.oracle.f(next_x)
        next_grad = self.oracle.df(next_x)

        self.L_t = self.est_L(next_x, next_f, next_grad, first_iter=(t == 1))
        self._x = next_x
        self._f = next_f
        self._grad = next_grad

        self.early_stop = self.stop_nonconvex and self._detected_nonconvex
        # warnings.warn(f"Detected non-convex function (res: {linearization_diff:.4e})")

    def line_search_eta(self, first_eta: int = -1):
        """
        Line search for $eta$ so that
        $$
            frac{beta}{4(1-beta)L} leq eta leq frac{1}{3L}
        """
        eta = self._first_eta if self._first_eta > 0 else 1
        tau_t = 0

        phase_I = True
        first_iter = True
        eta_lb = ...
        eta_ub = ...
        phase_I_incr = ...

        while 1:
            z = self.projection(self._y - eta * self._grad)
            y = (1 - self.beta) * self._y + self.beta * z
            next_x = (z + tau_t * self._x) / (1.0 + tau_t)
            next_grad = self.oracle.df(next_x)

            L = self.est_L(next_x, ..., next_grad, first_iter=True)
            if L == 0 or (self.stop_nonconvex and self._detected_nonconvex):
                # TODO: This is a hacky solution
                return 1.0 / 3

            lb = self.beta / (4 * (1.0 - self.beta) * L)
            ub = 1.0 / (3 * L)
            if lb <= eta <= ub:
                return eta
            elif phase_I:
                # check if we should start/continue doubling search
                if (first_iter or phase_I_incr) and eta < lb:
                    eta *= 2
                    phase_I_incr = True
                elif (first_iter or not phase_I_incr) and ub < eta:
                    eta /= 2
                    phase_I_incr = False
                # below prepares line search for binary search
                elif phase_I_incr and ub < eta:
                    eta_ub = eta
                    eta_lb = eta / 2
                    eta = (eta_lb + eta_ub) / 2
                    phase_I = False
                else:
                    eta_ub = eta * 2
                    eta_lb = eta
                    eta = (eta_lb + eta_ub) / 2
                    phase_I = False
                first_iter = False
            else:
                if eta < lb:
                    eta_lb = eta
                else:
                    eta_ub = eta
                eta = (eta_lb + eta_ub) / 2

        return eta

    def est_L(self, next_x, next_f, next_grad, first_iter=False, tol=1e-10):
        """
        :param tol: tolerance for non-convexity/linearization
        """
        if first_iter:
            # need to add a small value below in case too close ...
            return torch.linalg.norm(next_grad - self._grad) / (
                1e-3 + torch.linalg.norm(next_x - self._x)
            )

        linearization_diff = self._f - next_f - torch.dot(next_grad, self._x - next_x)
        if linearization_diff > 0:
            L = torch.norm(self._grad - next_grad) ** 2 / (2 * linearization_diff)
        elif abs(linearization_diff) <= tol:
            L = 0
        else:
            self._detected_nonconvex = True
            L = 0

        return L
