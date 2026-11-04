"""
ANFIS (Non-Hybrid) -- Section 5 of the paper.
Key fix: n_mfs=3 (9 rules, close to paper's 10).
Premise params updated via scipy minimize on RMSE directly.
"""

import numpy as np
import sys
from sklearn.metrics import r2_score
from itertools import product
from scipy.optimize import minimize


class GaussianMF:
    def __init__(self, center, sigma):
        self.center = float(center)
        self.sigma  = max(float(sigma), 1e-4)

    def evaluate(self, x):
        return np.exp(-0.5 * ((x - self.center) / max(self.sigma, 1e-4)) ** 2)


class ANFIS:
    """
    5-layer Sugeno ANFIS.
    Consequents: closed-form regularized least squares (Ridge).
    Premises:    scipy L-BFGS-B (avoids gradient-stall issue).
    n_mfs=3 -> 9 rules (matches paper's 10 effective rules).
    """

    def __init__(self, n_inputs=2, n_mfs=3, epochs=100, random_state=42, alpha=1e-3, learning_rate=0.01, **kwargs):
        self.n_inputs  = n_inputs
        self.n_mfs     = n_mfs
        self.n_rules   = n_mfs ** n_inputs
        self.epochs    = epochs
        self.rs        = random_state
        self.alpha     = alpha
        self.lr        = learning_rate
        self.train_rmse_history = []
        self.mfs         = None
        self.consequents = None
        self._rules      = list(product(range(n_mfs), repeat=n_inputs))

    # ── Initialise MFs with percentile-spaced centers ─────────────────────────
    def _init_mfs(self, X):
        rng = np.random.default_rng(self.rs)
        self.mfs = []
        for i in range(self.n_inputs):
            pcts    = np.linspace(5, 95, self.n_mfs)
            centers = np.percentile(X[:, i], pcts)
            diffs   = np.diff(centers)
            sigma   = max(np.mean(diffs) * 0.8 if len(diffs) > 0 else 0.15, 0.05)
            self.mfs.append([GaussianMF(c, sigma) for c in centers])

    # ── Pack / unpack premise params as a flat vector ─────────────────────────
    def _pack_premises(self):
        v = []
        for i in range(self.n_inputs):
            for k in range(self.n_mfs):
                v += [self.mfs[i][k].center, self.mfs[i][k].sigma]
        return np.array(v)

    def _unpack_premises(self, v):
        idx = 0
        for i in range(self.n_inputs):
            for k in range(self.n_mfs):
                self.mfs[i][k].center = float(v[idx])
                self.mfs[i][k].sigma  = max(float(abs(v[idx+1])), 1e-4)
                idx += 2

    # ── Forward pass (fully vectorised) ──────────────────────────────────────
    def _forward(self, X):
        N = X.shape[0]
        mu = np.zeros((N, self.n_inputs, self.n_mfs))
        for i in range(self.n_inputs):
            for k, mf in enumerate(self.mfs[i]):
                mu[:, i, k] = mf.evaluate(X[:, i])

        w = np.ones((N, self.n_rules))
        for r, idxs in enumerate(self._rules):
            for inp, k in enumerate(idxs):
                w[:, r] *= mu[:, inp, k]
        w = np.clip(w, 1e-12, None)

        w_sum = w.sum(axis=1, keepdims=True) + 1e-9
        w_bar = w / w_sum

        X_aug = np.hstack([X, np.ones((N, 1))])
        f_mat = X_aug @ self.consequents.T
        y_hat = (w_bar * f_mat).sum(axis=1)
        return w_bar, y_hat

    # ── Closed-form regularized consequent fit (Ridge) ────────────────────────
    def _fit_consequents(self, X, y):
        N     = X.shape[0]
        X_aug = np.hstack([X, np.ones((N, 1))])
        w_bar, _ = self._forward(X)
        A_cols   = [w_bar[:, [r]] * X_aug for r in range(self.n_rules)]
        A        = np.hstack(A_cols)
        AtA      = A.T @ A
        theta    = np.linalg.solve(AtA + self.alpha * np.eye(AtA.shape[0]), A.T @ y)
        self.consequents = theta.reshape(self.n_rules, self.n_inputs + 1)

    # ── Objective for scipy optimizer ─────────────────────────────────────────
    def _objective(self, v, X, y):
        self._unpack_premises(v)
        self._fit_consequents(X, y)
        _, y_hat = self._forward(X)
        return float(np.mean((y - y_hat) ** 2))  # MSE

    # ── Training ──────────────────────────────────────────────────────────────
    def fit(self, X, y):
        np.random.seed(self.rs)
        self._init_mfs(X)
        self.consequents = np.zeros((self.n_rules, self.n_inputs + 1))

        print(f"\n{'-'*55}")
        print(f"  Training ANFIS | rules={self.n_rules}  iters={self.epochs}")
        print(f"{'-'*55}")
        sys.stdout.flush()

        # Initial consequent fit
        self._fit_consequents(X, y)

        # Optimise premise params using L-BFGS-B
        v0 = self._pack_premises()

        # Bounds: centers unconstrained, sigma > 0
        bounds = []
        for i in range(self.n_inputs):
            for k in range(self.n_mfs):
                bounds += [(None, None), (1e-4, None)]

        callback_rmse = []

        def cb(v):
            self._unpack_premises(v)
            self._fit_consequents(X, y)
            _, y_hat = self._forward(X)
            rmse = float(np.sqrt(np.mean((y - y_hat) ** 2)))
            r2   = float(r2_score(y, y_hat))
            callback_rmse.append(rmse)
            n = len(callback_rmse)
            if n == 1 or n % 10 == 0:
                print(f"  Iter [{n:>3}/{self.epochs}]  RMSE: {rmse:.5f}  R2: {r2:.4f}")
                sys.stdout.flush()

        result = minimize(
            self._objective, v0, args=(X, y),
            method='L-BFGS-B',
            bounds=bounds,
            callback=cb,
            options={'maxiter': self.epochs, 'ftol': 1e-12, 'gtol': 1e-8}
        )

        self._unpack_premises(result.x)
        self._fit_consequents(X, y)

        _, y_hat = self._forward(X)
        # Store history
        self.train_rmse_history = callback_rmse if callback_rmse else \
            [float(np.sqrt(np.mean((y - y_hat)**2)))]

        final_rmse = float(np.sqrt(np.mean((y - y_hat)**2)))
        final_r2   = float(r2_score(y, y_hat))
        print(f"\n  Converged: RMSE={final_rmse:.5f}  R2={final_r2:.4f}")
        print(f"{'-'*55}")
        sys.stdout.flush()
        return self

    def predict(self, X):
        _, y_hat = self._forward(X)
        return y_hat

    def score(self, X, y):
        return r2_score(y, self.predict(X))