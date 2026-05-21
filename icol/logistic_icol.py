import warnings
warnings.filterwarnings('ignore')

from time import time
from copy import deepcopy

import math

import numpy as np
import sympy as sp

from sklearn.base import clone
from sklearn.model_selection import train_test_split

from sklearn.metrics import log_loss, zero_one_loss, hinge_loss, roc_auc_score
from sklearn.linear_model import LogisticRegression

from itertools import combinations, permutations

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def corr(X, g):
    X = np.asarray(X)
    g = np.asarray(g).ravel()

    sigma_X = np.std(X, axis=0)
    sigma_Y = np.std(g)

    XY = X * g.reshape(-1, 1)
    E_XY = np.mean(XY, axis=0)
    E_X = np.mean(X, axis=0)
    E_Y = np.mean(g)

    cov = E_XY - E_X * E_Y
    sigma = sigma_X * sigma_Y

    pearsons = cov / sigma
    absolute_pearsons = np.abs(pearsons)

    absolute_pearsons[np.isnan(absolute_pearsons)] = 0.0
    absolute_pearsons[np.isinf(absolute_pearsons)] = 0.0
    absolute_pearsons[np.isneginf(absolute_pearsons)] = 0.0

    return absolute_pearsons

def squared(X, y, model):
    y_hat = model.predict(X).ravel()
    res = y - y_hat
    return corr(X, res)

def df_log_loss(X, y, model, clp=np.inf):
    eta = model.decision_function(X).ravel()
    eta = np.clip(eta, -clp, clp)

    p = 1.0 / (1.0 + np.exp(-eta))
    g = y - p

    return corr(X, g)

OBJECTIVE_DICT = {
    "squared": squared,
    "logistic": df_log_loss,
}

LOSS_DICT = {
    'squared': rmse,
    'zero_one': zero_one_loss,
    'hinge': hinge_loss,
    'logloss': log_loss,
    'logistic': log_loss
}

def count_nnz(coef, eps_nnz):
    return int(np.sum(np.abs(np.ravel(coef)) > eps_nnz))
    
def linear_search(models, X, y, d, eps_nnz=1e-3, verbose=False):
    if X.shape[1] <= d:
        mid = -1
        best = -1
        models[mid].fit(X, y)
        return best, models

    best_idx = 0
    for i, model in enumerate(models):
        if verbose: print('Fitting model {0} of {1} with C={2} and has '.format(i, len(models), model.C), end='')
        model.fit(X, y)
        nnz = count_nnz(model.coef_, eps_nnz)
        if verbose: print('{0} nonzero terms'.format(nnz))
        if nnz<=d:
            best_idx = i
        else:
            break
    return best_idx, models

def binary_search(models, X, y, d, eps_nnz=1e-3, verbose=False):
    start, stop = 0, len(models) - 1
    best = None
    count = 0

    if X.shape[1] <= d:
        mid = -1
        best = -1
        models[mid].fit(X, y)
        return best, models
    
    while start <= stop and count < len(models):
        mid = (start + stop) // 2
        if verbose:
            print(f'Fitting model {count} of {len(models)} with C={models[mid].C} and has ', end='')
        models[mid].fit(X, y)
        nnz = count_nnz(models[mid].coef_, eps_nnz)
        if verbose:
            print(f'{nnz} nonzero terms')
        if nnz <= d:
            best = mid         
            start = mid + 1
        else:
            stop = mid - 1
        count += 1
    return (best if best is not None else 0), models

SEARCH_DICT = {
    'linear': linear_search,
    'binary': binary_search
}

class generalized_SIS:
    def __init__(self, s, obj="squared", clp=np.inf):
        if int(s) < 1:
            raise ValueError("s must be at least 1.")

        if obj not in OBJECTIVE_DICT:
            raise ValueError(
                f"Unknown screening objective '{obj}'. "
                f"Available objectives are {list(OBJECTIVE_DICT.keys())}."
            )

        self.s = int(s)
        self.obj = obj
        self.clp = clp

    def __str__(self):
        return f"SIS(s={self.s}, obj={self.obj})"

    def __repr__(self):
        return self.__str__()

    def get_params(self, deep=True):
        return {
            "s": self.s,
            "obj": self.obj,
            "clp": self.clp,
        }

    def __call__(self, X, y, model, pool):
        X = np.asarray(X)
        y = np.asarray(y)

        p = X.shape[1]

        if pool is None:
            pool = []

        pool_set = set(int(j) for j in pool)

        if len(pool_set) >= p:
            return np.array([], dtype=float), np.array([], dtype=int)

        if self.obj == "logistic":
            scores = OBJECTIVE_DICT[self.obj](
                X=X,
                y=y,
                model=model,
                clp=self.clp,
            )
        else:
            scores = OBJECTIVE_DICT[self.obj](
                X=X,
                y=y,
                model=model,
            )

        scores = np.asarray(scores).ravel()

        if scores.shape[0] != p:
            raise ValueError(
                f"Screening score vector has length {scores.shape[0]}, "
                f"but X has {p} columns."
            )

        # corr already returns absolute correlations.
        idxs = np.argsort(scores)[::-1]

        chosen = []
        for j in idxs:
            j = int(j)

            if j not in pool_set:
                chosen.append(j)

                if len(chosen) == self.s:
                    break

        chosen = np.array(chosen, dtype=int)

        return scores[chosen], chosen

# Backwards-compatible British spelling alias.
generalised_SIS = generalized_SIS

class THRESHOLDED_LOGISTIC_REGRESSION:
    def __init__(
        self,
        random_state=None,
        fit_intercept=False,
        C=1e6,
        tol=1e-8,
        max_iter=1000,
        clp=30.0,
    ):
        self.random_state = random_state
        self.fit_intercept = fit_intercept
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.clp = clp

        if C != 1e6:
            print(
                "Warning: non-default C supplied for THRESHOLDED_LOGISTIC_REGRESSION. "
                "The written method assumes a large fixed C corresponding to a small "
                "ridge penalty."
            )

    def __str__(self):
        return "THRESHOLDED_LOGISTIC_REGRESSION"

    def __repr__(self, prec=3):
        return self.__str__()

    def get_params(self, deep=True):
        return {
            "random_state": self.random_state,
            "fit_intercept": self.fit_intercept,
            "C": self.C,
            "tol": self.tol,
            "max_iter": self.max_iter,
            "clp": self.clp,
        }

    def _new_logistic_model(self):
        return LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            C=self.C,
            fit_intercept=self.fit_intercept,
            random_state=self.random_state,
            tol=self.tol,
            max_iter=self.max_iter,
        )

    def fit(self, X, y, d, feature_names=None, verbose=False):
        X = np.asarray(X)
        y = np.asarray(y)

        n, p = X.shape

        if d < 1:
            raise ValueError("d must be at least 1.")

        if d > p:
            d = p

        if feature_names is None:
            feature_names = np.array([f"X{i}" for i in range(p)])
        else:
            feature_names = np.asarray(feature_names)

        if verbose:
            print("Fitting fixed-ridge logistic regression on full candidate pool")

        full_model = self._new_logistic_model()
        full_model.fit(X, y)

        if verbose:
            print(f"Selecting top {d} candidates")

        beta_hat_all = np.abs(full_model.coef_.ravel())
        idxs = np.argsort(beta_hat_all)[::-1]
        use_idxs = np.sort(idxs[:d])

        if verbose:
            print("Refitting fixed-ridge logistic regression on selected candidates")

        selected_model = self._new_logistic_model()
        selected_model.fit(X[:, use_idxs], y)

        beta_hat_sparse = selected_model.coef_.ravel()

        beta_hat = np.zeros(p)
        beta_hat[use_idxs] = beta_hat_sparse

        self.coef_ = beta_hat
        self.selected_features_ = use_idxs
        self.feature_names = feature_names[use_idxs]
        self.full_model_ = full_model
        self.selected_model_ = selected_model

        if self.fit_intercept:
            self.intercept_ = float(selected_model.intercept_[0])
        else:
            self.intercept_ = 0.0

        # Make self.model work on the full original feature space.
        self.model = self

        return self

    def decision_function(self, X):
        X = np.asarray(X)
        return np.dot(X, self.coef_.ravel()) + self.intercept_

    def predict_proba(self, X):
        z = self.decision_function(X)
        z = np.clip(z, -self.clp, self.clp)

        p1 = 1.0 / (1.0 + np.exp(-z))
        p0 = 1.0 - p1

        return np.column_stack([p0, p1])

    def predict(self, X, threshold=0.5):
        p1 = self.predict_proba(X)[:, 1]
        return (p1 >= threshold).astype(int)

class BSS_LOGISTIC:
    def __init__(
        self,
        random_state=None,
        metric="logloss",
        fit_intercept=False,
        C=1e6,
        tol=1e-8,
        max_iter=1000,
        clp=30.0,
        max_subsets=1000000,
    ):
        self.random_state = random_state
        self.metric = metric
        self.fit_intercept = fit_intercept
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.clp = clp
        self.max_subsets = max_subsets

        if C != 1e6:
            print(
                "Warning: non-default C supplied for BSS_LOGISTIC. "
                "The written method assumes a large fixed C corresponding to a small "
                "ridge penalty."
            )

        if metric != "logloss":
            print(
                "Warning: non-default metric supplied for BSS_LOGISTIC. "
                "The written method selects subsets using training log loss."
            )

    def __str__(self):
        return "BSS_LOGISTIC"

    def __repr__(self, prec=3):
        return self.__str__()

    def get_params(self, deep=True):
        return {
            "random_state": self.random_state,
            "metric": self.metric,
            "fit_intercept": self.fit_intercept,
            "C": self.C,
            "tol": self.tol,
            "max_iter": self.max_iter,
            "clp": self.clp,
            "max_subsets": self.max_subsets,
        }

    def _new_logistic_model(self):
        return LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            C=self.C,
            fit_intercept=self.fit_intercept,
            random_state=self.random_state,
            tol=self.tol,
            max_iter=self.max_iter,
        )

    def _subset_loss(self, y_true, y_prob):
        if self.metric == "logloss":
            return log_loss(y_true, y_prob, labels=[0, 1])

        raise ValueError(
            f"Unsupported metric: {self.metric}. "
            "Currently supported metric is 'logloss'."
        )

    def fit(self, X, y, d, feature_names=None, verbose=False):
        X = np.asarray(X)
        y = np.asarray(y)

        n, p = X.shape

        if d < 1:
            raise ValueError("d must be at least 1.")

        if d > p:
            d = p

        n_subsets = math.comb(p, d)

        if self.max_subsets is not None and n_subsets > self.max_subsets:
            warnings.warn(
                f"BSS_LOGISTIC will evaluate {n_subsets} subsets, "
                f"which exceeds max_subsets={self.max_subsets}. "
                "This may be computationally expensive.",
                RuntimeWarning,
            )

        if feature_names is None:
            feature_names = np.array([f"X{i}" for i in range(p)])
        else:
            feature_names = np.asarray(feature_names)

        base = self._new_logistic_model()

        best_i = None
        best_comb = None
        best_loss = None
        best_coef = None
        best_intercept = 0.0
        best_model = None

        for i, comb in enumerate(combinations(range(p), d)):
            if verbose:
                print(f"Attempting combination {i + 1}/{n_subsets}: {comb}")

            LR = clone(base)
            LR.fit(X[:, comb], y)

            y_prob = LR.predict_proba(X[:, comb])[:, 1]
            loss = self._subset_loss(y, y_prob)

            if best_i is None or loss < best_loss:
                best_i = i
                best_comb = comb
                best_loss = loss
                best_coef = LR.coef_.ravel()
                best_model = LR

                if self.fit_intercept:
                    best_intercept = float(LR.intercept_[0])
                else:
                    best_intercept = 0.0

        if verbose:
            print(f"The best combination was {best_comb} with a loss of {best_loss}")

        use_idxs = np.array(best_comb, dtype=int)

        beta_hat = np.zeros(p)
        beta_hat[use_idxs] = best_coef

        self.coef_ = beta_hat
        self.intercept_ = best_intercept
        self.use_idxs = list(use_idxs)
        self.selected_features_ = use_idxs
        self.feature_names = feature_names[use_idxs]
        self.best_loss_ = best_loss
        self.best_subset_index_ = best_i
        self.selected_model_ = best_model

        # Make self.model work on the full original feature space.
        self.model = self

        if verbose:
            print(f"Corresponding to the features: {self.feature_names}")

        return self

    def decision_function(self, X):
        X = np.asarray(X)
        return np.dot(X, self.coef_.ravel()) + self.intercept_

    def predict_proba(self, X):
        z = self.decision_function(X)
        z = np.clip(z, -self.clp, self.clp)

        p1 = 1.0 / (1.0 + np.exp(-z))
        p0 = 1.0 - p1

        return np.column_stack([p0, p1])

    def predict(self, X, threshold=0.5):
        p1 = self.predict_proba(X)[:, 1]
        return (p1 >= threshold).astype(int)

class LOGISTIC_ELASTIC_NET:
    def __init__(
        self,
        lambda_min=1e-4,
        lambda_max=None,
        r=1.1,
        lambda2=1e-6,
        class_weight=None,
        max_iter=5000,
        epsilon=1e-4,
        eps_nnz=1e-12,
        clp=30.0,
        random_state=None,
        fit_intercept=False,
        max_path_fits=200,
        scale_penalty_by_n=True,
        adaptive_lambda_max=True,
        require_standardized=True,
        standardization_mean_tol=1e-6,
        standardization_sd_tol=1e-3,
    ):
        if lambda_min <= 0:
            raise ValueError("lambda_min must be positive.")

        if lambda_max is not None and lambda_max <= lambda_min:
            raise ValueError("lambda_max must be greater than lambda_min.")

        if r <= 1:
            raise ValueError("r must be greater than 1.")

        if lambda2 <= 0:
            raise ValueError("lambda2 must be positive.")

        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.r = r
        self.lambda2 = lambda2
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.tol = epsilon
        self.eps_nnz = eps_nnz
        self.clp = clp
        self.random_state = random_state
        self.fit_intercept = fit_intercept
        self.max_path_fits = max_path_fits
        self.scale_penalty_by_n = scale_penalty_by_n
        self.adaptive_lambda_max = adaptive_lambda_max
        self.require_standardized = require_standardized
        self.standardization_mean_tol = standardization_mean_tol
        self.standardization_sd_tol = standardization_sd_tol

    def __str__(self):
        return "LOGISTIC_ELASTIC_NET"

    def __repr__(self, prec=3):
        return self.__str__()

    def get_params(self, deep=True):
        return {
            "lambda_min": self.lambda_min,
            "lambda_max": self.lambda_max,
            "r": self.r,
            "lambda2": self.lambda2,
            "class_weight": self.class_weight,
            "max_iter": self.max_iter,
            "epsilon": self.epsilon,
            "eps_nnz": self.eps_nnz,
            "clp": self.clp,
            "random_state": self.random_state,
            "fit_intercept": self.fit_intercept,
            "max_path_fits": self.max_path_fits,
            "scale_penalty_by_n": self.scale_penalty_by_n,
            "adaptive_lambda_max": self.adaptive_lambda_max,
            "require_standardized": self.require_standardized,
            "standardization_mean_tol": self.standardization_mean_tol,
            "standardization_sd_tol": self.standardization_sd_tol,
        }

    def _default_lambda_max(self, p):
        if self.lambda_max is not None:
            return self.lambda_max

        # Theoretical/heuristic starting value under standardized features.
        # This is treated as an initial guess, not an absolute guarantee,
        # if adaptive_lambda_max=True.
        return p / 4.0

    def _check_standardized(self, X):
        if not self.require_standardized:
            return

        if self.lambda_max is not None:
            return

        col_means = np.mean(X, axis=0)
        col_sds = np.std(X, axis=0)

        max_abs_mean = np.max(np.abs(col_means))
        max_abs_sd_error = np.max(np.abs(col_sds - 1.0))

        if (
            max_abs_mean > self.standardization_mean_tol
            or max_abs_sd_error > self.standardization_sd_tol
        ):
            warnings.warn(
                "LOGISTIC_ELASTIC_NET is using the default lambda_max=p/4, "
                "which assumes standardized features. The supplied X does not "
                "appear to be standardized. Either standardize X before fitting, "
                "provide an explicit lambda_max, or set require_standardized=False.",
                RuntimeWarning,
            )

    def _sklearn_elasticnet_params(self, lambda1, n):
        """
        Convert explicit penalties lambda1 and fixed lambda2 into scikit-learn's
        C and l1_ratio.

        We want the absolute L2 penalty to remain fixed as lambda1 varies.

        Without n scaling:
            C = 1 / (lambda1 + lambda2)

        With n scaling:
            C = 1 / (n * (lambda1 + lambda2))

        In both cases:
            l1_ratio = lambda1 / (lambda1 + lambda2)

        Thus changing lambda1 changes l1_ratio, while lambda2 remains fixed
        in the intended penalty scale.
        """
        total = lambda1 + self.lambda2

        if self.scale_penalty_by_n:
            C = 1.0 / (n * total)
        else:
            C = 1.0 / total

        l1_ratio = lambda1 / total

        return C, l1_ratio

    def _new_model(self, lambda1, n):
        C, l1_ratio = self._sklearn_elasticnet_params(lambda1=lambda1, n=n)

        return LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            C=C,
            l1_ratio=l1_ratio,
            class_weight=self.class_weight,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
            tol=self.epsilon,
            random_state=self.random_state,
        )

    def _count_nnz(self, coef):
        return int(np.sum(np.abs(np.ravel(coef)) > self.eps_nnz))

    def _fit_at_lambda(self, X, y, lambda1, verbose=False):
        n = X.shape[0]

        model = self._new_model(lambda1=lambda1, n=n)
        model.fit(X, y)

        coef = model.coef_.ravel()
        nnz = self._count_nnz(coef)

        prob = model.predict_proba(X)[:, 1]
        loss = log_loss(y, prob, labels=[0, 1])

        C, l1_ratio = self._sklearn_elasticnet_params(lambda1=lambda1, n=n)

        result = {
            "lambda1": float(lambda1),
            "lambda2": float(self.lambda2),
            "C": float(C),
            "l1_ratio": float(l1_ratio),
            "model": model,
            "coef": coef,
            "nnz": nnz,
            "loss": loss,
        }

        if verbose:
            print(
                f"lambda1={lambda1:g}, "
                f"lambda2={self.lambda2:g}, "
                f"C={C:g}, "
                f"l1_ratio={l1_ratio:g}, "
                f"nnz={nnz}, "
                f"logloss={loss:g}"
            )

        return result

    def _update_best_under(self, best_under, candidate, d):
        if candidate["nnz"] > d:
            return best_under

        if best_under is None:
            return candidate

        if candidate["nnz"] > best_under["nnz"]:
            return candidate

        if candidate["nnz"] == best_under["nnz"] and candidate["loss"] < best_under["loss"]:
            return candidate

        return best_under

    def _choose_fallback(self, candidates, d):
        under = [c for c in candidates if c["nnz"] <= d]

        if len(under) > 0:
            # Largest model that does not exceed d.
            # Break ties by lower training log loss.
            return max(under, key=lambda c: (c["nnz"], -c["loss"]))

        warnings.warn(
            "No fitted Logistic Elastic Net model had at most d nonzero "
            "coefficients. Choosing the smallest fitted model instead. "
            "This usually means lambda_max was not large enough or "
            "max_path_fits was reached too early.",
            RuntimeWarning,
        )

        return min(candidates, key=lambda c: (c["nnz"], c["loss"]))

    def _already_fit_lambda(self, candidates, lambda1):
        return any(np.isclose(c["lambda1"], lambda1) for c in candidates)

    def fit(self, X, y, d, feature_names=None, verbose=False):
        X = np.asarray(X)
        y = np.asarray(y)

        n, p = X.shape

        if d < 1:
            raise ValueError("d must be at least 1.")

        if d > p:
            d = p

        if feature_names is None:
            self.feature_names_all_ = np.array([f"X{i}" for i in range(p)])
        else:
            self.feature_names_all_ = np.asarray(feature_names)

        self._check_standardized(X)

        lambda_start = self._default_lambda_max(p)

        if lambda_start <= self.lambda_min:
            raise ValueError(
                f"lambda_max={lambda_start:g} must be greater than "
                f"lambda_min={self.lambda_min:g}."
            )

        candidates = []
        best_under = None
        n_fits = 0

        self.initial_lambda_max_ = lambda_start

        if verbose:
            print(
                "Starting Logistic Elastic Net path search with "
                f"initial lambda_max={lambda_start:g}"
            )
            print(
                f"scale_penalty_by_n={self.scale_penalty_by_n}, "
                f"adaptive_lambda_max={self.adaptive_lambda_max}"
            )

        # ------------------------------------------------------------
        # 0. Validate or enlarge lambda_max.
        # ------------------------------------------------------------
        lambda_max = lambda_start

        while True:
            if n_fits >= self.max_path_fits:
                warnings.warn(
                    "Maximum number of path fits reached while validating "
                    "lambda_max.",
                    RuntimeWarning,
                )
                break

            first = self._fit_at_lambda(X, y, lambda_max, verbose=verbose)
            candidates.append(first)
            best_under = self._update_best_under(best_under, first, d)
            n_fits += 1

            if first["nnz"] <= d:
                break

            if not self.adaptive_lambda_max:
                warnings.warn(
                    "The model at lambda_max has more than d nonzero coefficients, "
                    "and adaptive_lambda_max=False. The selected model may exceed d.",
                    RuntimeWarning,
                )
                break

            if verbose:
                print(
                    f"lambda_max={lambda_max:g} gave nnz={first['nnz']} > d={d}; "
                    "doubling lambda_max."
                )

            lambda_max *= 2.0

        self.validated_lambda_max_ = lambda_max
        self.lambda_max_nnz_ = first["nnz"]

        # ------------------------------------------------------------
        # 1. If lambda_max already gives exactly d, stop.
        # ------------------------------------------------------------
        if first["nnz"] == d:
            chosen = first
            search_status = "exact_at_lambda_max"

        # ------------------------------------------------------------
        # 2. If lambda_max still gives too many features and adaptive
        #    enlargement was disabled or exhausted, fallback.
        # ------------------------------------------------------------
        elif first["nnz"] > d:
            chosen = self._choose_fallback(candidates, d)
            search_status = "lambda_max_too_small"

        # ------------------------------------------------------------
        # 3. Otherwise begin coarse downward search by halving lambda.
        # ------------------------------------------------------------
        else:
            lambda_hi = lambda_max
            lambda_lo = None

            previous = first
            current_lambda = lambda_max / 2.0

            bracket_found = False
            exact_found = False
            chosen = None

            while current_lambda >= self.lambda_min:
                if n_fits >= self.max_path_fits:
                    warnings.warn(
                        "Maximum number of path fits reached during coarse "
                        "Logistic Elastic Net search.",
                        RuntimeWarning,
                    )
                    break

                current = self._fit_at_lambda(X, y, current_lambda, verbose=verbose)
                candidates.append(current)
                best_under = self._update_best_under(best_under, current, d)
                n_fits += 1

                if current["nnz"] == d:
                    chosen = current
                    exact_found = True
                    search_status = "exact_during_halving"
                    break

                if current["nnz"] > d:
                    # We have crossed from acceptable model size to too-large.
                    #
                    # lambda_lo: lower penalty, too weak, too many features.
                    # lambda_hi: higher penalty, acceptable feature count.
                    lambda_lo = current_lambda
                    lambda_hi = previous["lambda1"]
                    bracket_found = True
                    break

                previous = current
                lambda_hi = current_lambda
                current_lambda = current_lambda / 2.0

            # --------------------------------------------------------
            # 4. If no bracket was found before lambda_min, fit/use lambda_min.
            # --------------------------------------------------------
            if not exact_found and not bracket_found:
                if n_fits < self.max_path_fits:
                    if not self._already_fit_lambda(candidates, self.lambda_min):
                        current = self._fit_at_lambda(
                            X,
                            y,
                            self.lambda_min,
                            verbose=verbose,
                        )
                        candidates.append(current)
                        best_under = self._update_best_under(best_under, current, d)
                        n_fits += 1

                        if current["nnz"] == d:
                            chosen = current
                            exact_found = True
                            search_status = "exact_at_lambda_min"

                if not exact_found:
                    chosen = self._choose_fallback(candidates, d)
                    search_status = "lambda_min_reached"

            # --------------------------------------------------------
            # 5. Geometric binary search within multiplicative bracket.
            # --------------------------------------------------------
            if not exact_found and bracket_found:
                if verbose:
                    print(
                        "Starting binary search with bracket "
                        f"[lambda_lo={lambda_lo:g}, lambda_hi={lambda_hi:g}]"
                    )

                while lambda_hi / lambda_lo > self.r:
                    if n_fits >= self.max_path_fits:
                        warnings.warn(
                            "Maximum number of path fits reached during binary "
                            "Logistic Elastic Net search.",
                            RuntimeWarning,
                        )
                        break

                    lambda_mid = np.sqrt(lambda_lo * lambda_hi)

                    current = self._fit_at_lambda(X, y, lambda_mid, verbose=verbose)
                    candidates.append(current)
                    best_under = self._update_best_under(best_under, current, d)
                    n_fits += 1

                    if current["nnz"] == d:
                        chosen = current
                        exact_found = True
                        search_status = "exact_during_binary_search"
                        break

                    if current["nnz"] > d:
                        # Penalty is too weak; model is too large.
                        # Raise the lower end of the penalty bracket.
                        lambda_lo = lambda_mid
                    else:
                        # Penalty is strong enough; model is acceptable.
                        # Lower the upper end of the penalty bracket.
                        lambda_hi = lambda_mid

                if not exact_found:
                    chosen = self._choose_fallback(candidates, d)
                    search_status = "resolution_reached"

        # ------------------------------------------------------------
        # 6. Store final model in full feature space.
        # ------------------------------------------------------------
        self.coef_ = chosen["coef"].copy()
        self.intercept_ = (
            float(chosen["model"].intercept_[0])
            if self.fit_intercept
            else 0.0
        )

        self.coef_idx_ = np.where(np.abs(self.coef_) > self.eps_nnz)[0]
        self.selected_features_ = self.coef_idx_
        self.feature_names = self.feature_names_all_[self.selected_features_]

        self.best_lambda1_ = chosen["lambda1"]
        self.best_lambda2_ = chosen["lambda2"]
        self.best_C_ = chosen["C"]
        self.best_l1_ratio_ = chosen["l1_ratio"]
        self.best_nnz_ = chosen["nnz"]
        self.best_loss_ = chosen["loss"]
        self.search_status_ = search_status
        self.candidates_ = candidates
        self.n_path_fits_ = n_fits
        self.selected_model_ = chosen["model"]

        # Make self.model work on the full original feature space.
        self.model = self

        if verbose:
            print(
                f"Selected lambda1={self.best_lambda1_:g}, "
                f"lambda2={self.best_lambda2_:g}, "
                f"C={self.best_C_:g}, "
                f"l1_ratio={self.best_l1_ratio_:g}, "
                f"nnz={self.best_nnz_}, "
                f"logloss={self.best_loss_:g}, "
                f"status={self.search_status_}"
            )
            print(
                f"initial_lambda_max={self.initial_lambda_max_:g}, "
                f"validated_lambda_max={self.validated_lambda_max_:g}, "
                f"lambda_max_nnz={self.lambda_max_nnz_}"
            )
            print(f"Selected features: {self.feature_names}")

        return self

    def decision_function(self, X):
        X = np.asarray(X)
        return np.dot(X, self.coef_.ravel()) + self.intercept_

    def predict_proba(self, X):
        z = self.decision_function(X)
        z = np.clip(z, -self.clp, self.clp)

        p1 = 1.0 / (1.0 + np.exp(-z))
        p0 = 1.0 - p1

        return np.column_stack([p0, p1])

    def predict(self, X, threshold=0.5):
        p1 = self.predict_proba(X)[:, 1]
        return (p1 >= threshold).astype(int)

class ADAPTIVE_LOGISTIC_ELASTIC_NET(LOGISTIC_ELASTIC_NET):
    def __init__(
        self,
        lambda_min=1e-4,
        lambda_max=None,
        r=1.1,
        lambda2=1e-6,
        class_weight=None,
        max_iter=5000,
        epsilon=1e-4,
        eps_nnz=1e-12,
        clp=30.0,
        random_state=None,
        fit_intercept=False,
        max_path_fits=200,
        scale_penalty_by_n=True,
        adaptive_lambda_max=True,
        require_standardized=True,
        standardization_mean_tol=1e-6,
        standardization_sd_tol=1e-3,
        gamma=1.0,
        C0=1e6,
        epsilon0=1e-8,
        eps_weight=1e-8,
        scale_clp=1e6,
    ):
        if gamma < 0:
            raise ValueError(
                "gamma must be nonnegative. Use gamma=0 for standard Logistic "
                "Elastic Net and gamma>0 for Adaptive Logistic Elastic Net."
            )

        super().__init__(
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            r=r,
            lambda2=lambda2,
            class_weight=class_weight,
            max_iter=max_iter,
            epsilon=epsilon,
            eps_nnz=eps_nnz,
            clp=clp,
            random_state=random_state,
            fit_intercept=fit_intercept,
            max_path_fits=max_path_fits,
            scale_penalty_by_n=scale_penalty_by_n,
            adaptive_lambda_max=adaptive_lambda_max,
            require_standardized=require_standardized,
            standardization_mean_tol=standardization_mean_tol,
            standardization_sd_tol=standardization_sd_tol,
        )

        self.gamma = gamma
        self.C0 = C0
        self.epsilon0 = epsilon0
        self.eps_weight = eps_weight
        self.scale_clp = scale_clp

        if C0 != 1e6:
            print(
                "Warning: non-default C0 supplied for "
                "ADAPTIVE_LOGISTIC_ELASTIC_NET. The written method assumes a "
                "large fixed C0 corresponding to a small ridge stabilizer in "
                "the preliminary logistic fit."
            )

    def __str__(self):
        if self.gamma == 0:
            return "LOGISTIC_ELASTIC_NET"
        return f"ADAPTIVE_LOGISTIC_ELASTIC_NET(gamma={self.gamma:g})"

    def __repr__(self, prec=3):
        return self.__str__()

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)

        params.update(
            {
                "gamma": self.gamma,
                "C0": self.C0,
                "epsilon0": self.epsilon0,
                "eps_weight": self.eps_weight,
                "scale_clp": self.scale_clp,
            }
        )

        return params

    def _new_preliminary_model(self):
        return LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            C=self.C0,
            class_weight=self.class_weight,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
            tol=self.epsilon0,
            random_state=self.random_state,
        )

    def fit(self, X, y, d, feature_names=None, verbose=False):
        X = np.asarray(X)
        y = np.asarray(y)

        if self.gamma == 0:
            if verbose:
                print(
                    "gamma=0: using standard Logistic Elastic Net without "
                    "adaptive rescaling."
                )

            fitted = super().fit(
                X=X,
                y=y,
                d=d,
                feature_names=feature_names,
                verbose=verbose,
            )

            self.adaptive_scale_ = np.ones(X.shape[1])
            self.adaptive_weights_ = np.ones(X.shape[1])
            self.preliminary_model_ = None
            self.preliminary_coef_ = None

            return fitted

        if verbose:
            print("Fitting preliminary fixed-ridge logistic model for adaptive scaling.")

        preliminary_model = self._new_preliminary_model()
        preliminary_model.fit(X, y)

        preliminary_coef = preliminary_model.coef_.ravel()

        adaptive_scale = np.power(
            np.abs(preliminary_coef) + self.eps_weight,
            self.gamma,
        )

        adaptive_scale = np.clip(
            adaptive_scale,
            1.0 / self.scale_clp,
            self.scale_clp,
        )

        adaptive_weights = 1.0 / adaptive_scale

        if verbose:
            print(
                "Adaptive scale summary: "
                f"min={adaptive_scale.min():g}, "
                f"median={np.median(adaptive_scale):g}, "
                f"max={adaptive_scale.max():g}"
            )

        X_adapt = X * adaptive_scale

        # Fit the standard Logistic EN path search on the adaptively rescaled design.
        super().fit(
            X=X_adapt,
            y=y,
            d=d,
            feature_names=feature_names,
            verbose=verbose,
        )

        # The parent fit returned coefficients for X_adapt.
        coef_adapt = self.coef_.copy()
        intercept_adapt = self.intercept_

        # Since X_adapt_j = X_j * adaptive_scale_j,
        # beta_original_j = beta_adapt_j * adaptive_scale_j.
        coef_original = coef_adapt * adaptive_scale

        self.coef_adapt_ = coef_adapt
        self.intercept_adapt_ = intercept_adapt

        self.coef_ = coef_original
        self.intercept_ = intercept_adapt

        self.coef_idx_ = np.where(np.abs(self.coef_) > self.eps_nnz)[0]
        self.selected_features_ = self.coef_idx_

        if feature_names is None:
            self.feature_names_all_ = np.array([f"X{i}" for i in range(X.shape[1])])
        else:
            self.feature_names_all_ = np.asarray(feature_names)

        self.feature_names = self.feature_names_all_[self.selected_features_]

        self.adaptive_scale_ = adaptive_scale
        self.adaptive_weights_ = adaptive_weights
        self.preliminary_model_ = preliminary_model
        self.preliminary_coef_ = preliminary_coef

        # Keep the selected SAGA model in the adaptive scale, but make the
        # public model interface work on the original pre-adaptive feature scale.
        self.selected_model_adapt_ = self.selected_model_
        self.model = self

        if verbose:
            print("Transformed adaptive coefficients back to original feature scale.")
            print(f"Selected features: {self.feature_names}")

        return self

sci = lambda x, sig=3: f"{float(x):.{sig}e}"

# class LogisticICL:
#     def __init__(self, sis, so, k, 
#                  fit_intercept=True, normalize=True, pool_reset=False, optimize_k=True,
#                  track_intermediates=False, clp=30, random_state=None):
#         self.sis = sis
#         self.so = so
#         self.k = int(k)

#         self.fit_intercept = bool(fit_intercept)
#         self.normalize = bool(normalize)
#         self.pool_reset = bool(pool_reset)
#         self.optimize_k = bool(optimize_k)
#         self.track_intermediates = True if self.optimize_k else bool(track_intermediates)
#         self.clp = int(clp)
#         self.random_state=random_state

#         # learned
#         self.bad_col_ = None
#         self.p_filtered_ = None
#         self.feature_names_ = None

#         self.a_x_ = None
#         self.b_x_ = None

#         self.beta_idx_ = []
#         self.beta_scaled_ = np.zeros(0, dtype=float)
#         self.coef_ = np.zeros((1, 0), dtype=float)
#         self.intercept_ = 0.0

#         self.intermediates_ = np.empty((self.k, 5), dtype=object) # idx, coef, inter, names, repr

#     @staticmethod
#     def _sigmoid(z):
#         # stable sigmoid
#         out = np.empty_like(z, dtype=float)
#         pos = z >= 0
#         out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
#         ez = np.exp(z[~pos])
#         out[~pos] = ez / (1.0 + ez)
#         return out

#     @staticmethod
#     def _filter_invalid_cols(X):
#         X = np.asarray(X)
#         bad = np.any(~np.isfinite(X), axis=0)
#         return np.where(bad)[0]

#     def _maybe_filter_X(self, X):
#         X = np.asarray(X)
#         if self.bad_col_ is None:
#             return X
#         if hasattr(self, "p_filtered_") and X.shape[1] == self.p_filtered_:
#             return X
#         return np.delete(X, self.bad_col_, axis=1)


#     def _fit_transform_X(self, X):
#         X = np.asarray(X)
#         p = X.shape[1]

#         if not self.normalize:
#             self.a_x_ = np.zeros(p)
#             self.b_x_ = np.ones(p)
#             return X

#         self.a_x_ = X.mean(axis=0)
#         self.b_x_ = X.std(axis=0)
#         self.b_x_ = np.where(self.b_x_ == 0, 1.0, self.b_x_)
#         return (X - self.a_x_) / self.b_x_

#     # def _fit_transform_X(self, X):
#     #     X = np.asarray(X)
#     #     p = X.shape[1]

#     #     if not self.normalize:
#     #         self.a_x_ = np.zeros(p)
#     #         self.b_x_ = np.ones(p)
#     #         return X

#     #     self.a_x_ = X.mean(axis=0) if self.fit_intercept else np.zeros(p)
#     #     self.b_x_ = X.std(axis=0)
#     #     self.b_x_ = np.where(self.b_x_ == 0, 1.0, self.b_x_)
#     #     return (X - self.a_x_) / self.b_x_

#     def _unscale_coef(self, idx, beta_scaled, intercept_scaled):
#         idx = np.asarray(idx, dtype=int)
#         beta_scaled = np.asarray(beta_scaled, dtype=float).ravel()

#         if idx.size == 0:
#             self.coef_ = np.zeros((1, 0))
#             self.intercept_ = float(intercept_scaled) if self.fit_intercept else 0.0
#             return

#         if self.normalize:
#             coef_raw = beta_scaled / self.b_x_[idx]
#             inter_raw = float(intercept_scaled)
#             if self.fit_intercept:
#                 inter_raw = float(inter_raw - self.a_x_[idx] @ coef_raw)
#             else:
#                 inter_raw = 0.0
#         else:
#             coef_raw = beta_scaled
#             inter_raw = float(intercept_scaled) if self.fit_intercept else 0.0

#         self.coef_ = coef_raw.reshape(1, -1)
#         self.intercept_ = inter_raw

#     def _fit_fixed_k(self, Xn, y, feature_names, stopping, verbose=False, track_pool=False):
#         n, p = Xn.shape
#         pool = set()

#         if self.fit_intercept:
#             pbar = float(np.mean(y))
#             pbar = min(max(pbar, 1e-12), 1 - 1e-12)
#             intercept_s = float(np.log(pbar / (1 - pbar)))
#         else:
#             intercept_s = 0.0

#         beta = np.zeros(p, dtype=float)

#         for i in range(stopping):
#             if verbose:
#                 print(".", end="")

#             _, sis_i = self.sis(X=Xn, y=y, model=self, pool=list(pool))
#             pool_old = deepcopy(pool)
#             pool.update(sis_i)
#             pool_lst = list(pool)

#             self.so.fit(X=Xn[:, pool_lst], y=y, d=i+1,
#                         feature_names=feature_names[pool_lst],
#                         verbose=verbose)
#             beta_pool = np.asarray(self.so.coef_).ravel()
#             intercept_s = float(getattr(self.so, "intercept_", intercept_s))

#             beta[:] = 0.0
#             beta[pool_lst] = beta_pool

#             if self.pool_reset:
#                 keep = np.abs(beta_pool) > 0
#                 pool_lst = np.asarray(pool_lst)[keep].ravel().tolist()
#                 pool = set(pool_lst)
#                 beta[:] = 0.0
#                 beta[pool_lst] = beta_pool[keep]

#             idx = np.nonzero(beta)[0].tolist()
#             beta_sparse = beta[idx]
#             self.beta_idx_ = idx
#             self.beta_scaled_ = beta_sparse
#             self._unscale_coef(idx, beta_sparse, intercept_s)

#             if self.track_intermediates:
#                 self.intermediates_[i, 0] = np.array(idx, dtype=int)
#                 self.intermediates_[i, 1] = beta_sparse.copy()
#                 self.intermediates_[i, 2] = float(self.intercept_)
#                 self.intermediates_[i, 3] = feature_names[idx]
#                 self.intermediates_[i, 4] = None

#         if verbose:
#             print()

#         self.beta_idx_ = np.nonzero(beta)[0].tolist()
#         self.beta_scaled_ = beta[self.beta_idx_]

#     def fit(self, X, y, feature_names=None, val_size=0.1, random_state=None, verbose=False):
#         X = np.asarray(X)
#         y = np.asarray(y).ravel().astype(int)

#         self.bad_col_ = self._filter_invalid_cols(X)
#         Xf = np.delete(X, self.bad_col_, axis=1)
#         self.p_filtered_ = Xf.shape[1]

#         if feature_names is None or len(feature_names) != X.shape[1]:
#             fn = np.array([f"X_{j}" for j in range(X.shape[1])])
#         else:
#             fn = np.asarray(feature_names)
#         self.feature_names_ = np.delete(fn, self.bad_col_)

#         Xn = self._fit_transform_X(Xf)
#         if not self.optimize_k:
#             self._fit_fixed_k(Xn, y, self.feature_names_, stopping=self.k, verbose=verbose)
#         else:
#             X_tr, X_va, y_tr, y_va = train_test_split(
#                 Xn, y, test_size=val_size, random_state=random_state
#             )
#             self._fit_fixed_k(X_tr, y_tr, self.feature_names_, stopping=self.k, verbose=verbose)

#             best_k, best_loss = 0, np.inf
#             for kk in range(self.k):
#                 idx = self.intermediates_[kk, 0]
#                 coef = self.intermediates_[kk, 1]
#                 inter = self.intermediates_[kk, 2]
#                 eta = X_va[:, idx] @ coef + inter
#                 p1 = self._sigmoid(np.clip(eta, -30, 30))
#                 loss = log_loss(y_va, p1)

#                 if loss < best_loss:
#                     best_k, best_loss = kk + 1, loss

#             if verbose:
#                 print(f"refitting with k={best_k} (val logloss={best_loss:.6g})")

#             self._fit_fixed_k(Xn, y, self.feature_names_, stopping=best_k, verbose=verbose)

#         if len(self.beta_idx_) > 0:
#             lr = LogisticRegression(penalty=None, solver="lbfgs", fit_intercept=True)
#             lr.fit(Xn[:, self.beta_idx_], y)
#             beta_s = lr.coef_.ravel()
#             intercept_s = float(lr.intercept_[0])

#             self.beta_scaled_ = beta_s
#             self._unscale_coef(self.beta_idx_, beta_s, intercept_s)
#         else:
#             self.coef_ = np.zeros((1, 0), dtype=float)

#         return self

#     def decision_function(self, X):
#         Xf = self._maybe_filter_X(X)
#         if len(self.beta_idx_) == 0:
#             return np.full(Xf.shape[0], self.intercept_, dtype=float)
#         return Xf[:, self.beta_idx_] @ self.coef_.ravel() + self.intercept_

#     def predict_proba(self, X):
#         eta = self.decision_function(X)
#         p1 = self._sigmoid(np.clip(eta, -30, 30))
#         return np.column_stack([1.0 - p1, p1])

#     def predict(self, X, threshold=0.5):
#         return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

#     def negative_gradient(self, X, y):
#         eta = self.decision_function(X)
#         p = self._sigmoid(np.clip(eta, -self.clp, self.clp))
#         return np.asarray(y).ravel() - p

#     def get_params(self, deep=True):
#         return {
#             "sis": self.sis,
#             "so": self.so,
#             "k": self.k,
#             "fit_intercept": self.fit_intercept,
#             "normalize": self.normalize,
#             "pool_reset": self.pool_reset,
#             "optimize_k": self.optimize_k,
#             "track_intermediates": self.track_intermediates,
#             "clp": self.clp,
#         }

#     def __str__(self):
#         return f"LogisticICL({self.get_params()})"

#     def __repr__(self, prec=3):
#         coef = getattr(self, "coef_", None)
#         intercept = float(getattr(self, "intercept_", 0.0))
#         idx = getattr(self, "beta_idx_", None)

#         # Intercept-only or not yet fitted
#         if coef is None or idx is None or len(idx) == 0:
#             return (
#                 ("+" if intercept > 0 else "")
#                 + np.format_float_scientific(intercept, precision=prec, unique=False)
#             )

#         coef = np.asarray(coef).ravel()
#         idx = np.asarray(idx, dtype=int)

#         if getattr(self, "feature_names_", None) is not None:
#             names = np.asarray(self.feature_names_)[idx]
#         else:
#             names = np.array([f"X_{j}" for j in idx], dtype=object)

#         out = []
#         for c, name in zip(coef, names):
#             out.append(
#                 ("+" if float(c) > 0 else "")
#                 + np.format_float_scientific(float(c), precision=prec, unique=False)
#                 + " ("
#                 + str(name)
#                 + ")\n"
#             )

#         out.append(
#             ("+" if intercept > 0 else "")
#             + np.format_float_scientific(intercept, precision=prec, unique=False)
#         )

#         return "".join(out)


class _ScaledLogisticState:
    """
    Lightweight internal model used only during the Logistic ICL path.

    Its coefficients and intercept live on the standardized feature scale.
    This avoids passing the public LogisticICL object to generalized_SIS,
    because the public object stores raw-scale coefficients for prediction.
    """

    def __init__(self, beta, intercept, clp=30.0):
        self.beta = np.asarray(beta, dtype=float).ravel()
        self.intercept = float(intercept)
        self.clp = float(clp)

    def decision_function(self, X):
        X = np.asarray(X)
        return X @ self.beta + self.intercept

    def predict_proba(self, X):
        eta = self.decision_function(X)
        eta = np.clip(eta, -self.clp, self.clp)
        p1 = 1.0 / (1.0 + np.exp(-eta))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


class LogisticICL:
    def __init__(
        self,
        sis,
        so,
        k,
        fit_intercept=True,
        normalize=True,
        optimize_k=True,
        track_intermediates=False,
        clp=30.0,
        random_state=None,
        final_refit=False,
        final_refit_C=1e6,
        final_refit_tol=1e-8,
        final_refit_max_iter=1000,
    ):
        self.sis = sis
        self.so = so
        self.k = int(k)

        self.fit_intercept = bool(fit_intercept)
        self.normalize = bool(normalize)
        self.optimize_k = bool(optimize_k)
        self.track_intermediates = True if self.optimize_k else bool(track_intermediates)
        self.clp = float(clp)
        self.random_state = random_state

        self.final_refit = bool(final_refit)
        self.final_refit_C = final_refit_C
        self.final_refit_tol = final_refit_tol
        self.final_refit_max_iter = final_refit_max_iter

        if self.k < 1:
            raise ValueError("k must be at least 1.")

        if self.final_refit and self.final_refit_C != 1e6:
            print(
                "Warning: non-default final_refit_C supplied for LogisticICL. "
                "The intended final refit uses a large fixed C corresponding "
                "to a small ridge stabilizer."
            )

        # Learned attributes.
        self.bad_col_ = None
        self.p_filtered_ = None
        self.feature_names_ = None

        self.a_x_ = None
        self.b_x_ = None

        self.beta_idx_ = []
        self.beta_scaled_ = np.zeros(0, dtype=float)
        self.intercept_scaled_ = 0.0

        self.coef_ = np.zeros((1, 0), dtype=float)
        self.intercept_ = 0.0

        self.intermediates_ = None
        self.validation_intermediates_ = None
        self.best_k_ = None
        self.best_validation_loss_ = None
        self.final_base_estimator_ = None
        self.final_refit_model_ = None

    @staticmethod
    def _sigmoid(z):
        z = np.asarray(z, dtype=float)
        out = np.empty_like(z, dtype=float)

        pos = z >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))

        ez = np.exp(z[~pos])
        out[~pos] = ez / (1.0 + ez)

        return out

    @staticmethod
    def _filter_invalid_cols(X):
        X = np.asarray(X)
        bad = np.any(~np.isfinite(X), axis=0)
        return np.where(bad)[0]

    @staticmethod
    def _initial_intercept(y):
        pbar = float(np.mean(y))
        pbar = min(max(pbar, 1e-12), 1.0 - 1e-12)
        return float(np.log(pbar / (1.0 - pbar)))

    def _coef_eps(self):
        return float(getattr(self.so, "eps_nnz", 1e-12))

    def _maybe_filter_X(self, X):
        X = np.asarray(X)

        if self.bad_col_ is None:
            return X

        if hasattr(self, "p_filtered_") and X.shape[1] == self.p_filtered_:
            return X

        return np.delete(X, self.bad_col_, axis=1)

    def _fit_transform_X(self, X):
        X = np.asarray(X)
        p = X.shape[1]

        if not self.normalize:
            self.a_x_ = np.zeros(p)
            self.b_x_ = np.ones(p)
            return X

        self.a_x_ = X.mean(axis=0)
        self.b_x_ = X.std(axis=0)
        self.b_x_ = np.where(self.b_x_ == 0, 1.0, self.b_x_)

        return (X - self.a_x_) / self.b_x_

    def _transform_X(self, X):
        X = np.asarray(X)

        if self.a_x_ is None or self.b_x_ is None:
            raise RuntimeError("Scaling parameters have not been fitted.")

        return (X - self.a_x_) / self.b_x_

    def _unscale_coef(self, idx, beta_scaled, intercept_scaled):
        idx = np.asarray(idx, dtype=int)
        beta_scaled = np.asarray(beta_scaled, dtype=float).ravel()

        if idx.size == 0:
            self.coef_ = np.zeros((1, 0), dtype=float)
            self.intercept_ = float(intercept_scaled) if self.fit_intercept else 0.0
            return

        if self.normalize:
            coef_raw = beta_scaled / self.b_x_[idx]
            intercept_raw = float(intercept_scaled)

            if self.fit_intercept:
                intercept_raw = float(intercept_raw - self.a_x_[idx] @ coef_raw)
            else:
                intercept_raw = 0.0
        else:
            coef_raw = beta_scaled
            intercept_raw = float(intercept_scaled) if self.fit_intercept else 0.0

        self.coef_ = coef_raw.reshape(1, -1)
        self.intercept_ = intercept_raw

    def _new_final_refit_model(self):
        return LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            C=self.final_refit_C,
            fit_intercept=self.fit_intercept,
            max_iter=self.final_refit_max_iter,
            tol=self.final_refit_tol,
            random_state=self.random_state,
        )

    def _fixed_ridge_refit_scaled(self, Xn, y, idx):
        idx = np.asarray(idx, dtype=int)

        if idx.size == 0:
            intercept_s = self._initial_intercept(y) if self.fit_intercept else 0.0
            return np.zeros(0, dtype=float), intercept_s, None

        refit_model = self._new_final_refit_model()
        refit_model.fit(Xn[:, idx], y)

        beta_s = refit_model.coef_.ravel()
        intercept_s = float(refit_model.intercept_[0]) if self.fit_intercept else 0.0

        return beta_s, intercept_s, refit_model

    def _fit_fixed_k(self, Xn, y, feature_names, stopping, verbose=False):
        Xn = np.asarray(Xn)
        y = np.asarray(y).ravel().astype(int)

        n, p = Xn.shape
        stopping = int(min(stopping, p))

        pool = set()
        beta = np.zeros(p, dtype=float)

        if self.fit_intercept:
            intercept_s = self._initial_intercept(y)
        else:
            intercept_s = 0.0

        if self.track_intermediates:
            self.intermediates_ = np.empty((stopping, 6), dtype=object)
        else:
            self.intermediates_ = None

        base_estimators = []
        coef_eps = self._coef_eps()

        actual_steps = 0

        for i in range(stopping):
            if verbose:
                print(".", end="")

            screen_model = _ScaledLogisticState(
                beta=beta,
                intercept=intercept_s,
                clp=self.clp,
            )

            _, new_idxs = self.sis(
                X=Xn,
                y=y,
                model=screen_model,
                pool=list(pool),
            )

            new_idxs = np.asarray(new_idxs, dtype=int)

            if new_idxs.size == 0 and len(pool) == 0:
                warnings.warn(
                    "SIS returned no features and the candidate pool is empty. "
                    "Stopping the LogisticICL path early.",
                    RuntimeWarning,
                )
                break

            pool.update(new_idxs.tolist())
            pool_lst = sorted(pool)

            if len(pool_lst) == 0:
                break

            d_i = min(i + 1, len(pool_lst))

            so_i = clone(self.so)
            so_i.fit(
                X=Xn[:, pool_lst],
                y=y,
                d=d_i,
                feature_names=feature_names[pool_lst],
                verbose=verbose,
            )

            beta_pool = np.asarray(so_i.coef_).ravel()

            if beta_pool.shape[0] != len(pool_lst):
                raise ValueError(
                    "Base estimator returned a coefficient vector of length "
                    f"{beta_pool.shape[0]}, but the candidate pool has length "
                    f"{len(pool_lst)}."
                )

            # The preferred convention is that base estimators are called with
            # fit_intercept=False and the wrapper handles the intercept.
            # Only use the base-estimator intercept if it explicitly fitted one.
            if self.fit_intercept and bool(getattr(so_i, "fit_intercept", False)):
                intercept_s = float(getattr(so_i, "intercept_", intercept_s))

            beta[:] = 0.0
            beta[pool_lst] = beta_pool

            idx = np.where(np.abs(beta) > coef_eps)[0]
            beta_sparse = beta[idx]

            self.beta_idx_ = idx.tolist()
            self.beta_scaled_ = beta_sparse.copy()
            self.intercept_scaled_ = float(intercept_s)
            self._unscale_coef(idx, beta_sparse, intercept_s)

            base_estimators.append(so_i)
            actual_steps = i + 1

            if self.track_intermediates:
                self.intermediates_[i, 0] = np.array(idx, dtype=int)
                self.intermediates_[i, 1] = beta_sparse.copy()
                self.intermediates_[i, 2] = float(intercept_s)
                self.intermediates_[i, 3] = feature_names[idx]
                self.intermediates_[i, 4] = repr(so_i)
                self.intermediates_[i, 5] = np.array(pool_lst, dtype=int)

        if verbose:
            print()

        if actual_steps == 0:
            self.beta_idx_ = []
            self.beta_scaled_ = np.zeros(0, dtype=float)
            self.intercept_scaled_ = float(intercept_s)
            self._unscale_coef([], [], intercept_s)
            self.final_base_estimator_ = None
        else:
            self.final_base_estimator_ = base_estimators[-1]

        if self.track_intermediates and actual_steps < stopping:
            self.intermediates_ = self.intermediates_[:actual_steps, :]

        return actual_steps

    def _validation_loss_for_intermediate(self, X_tr, y_tr, X_va, y_va, row):
        idx = np.asarray(row[0], dtype=int)

        if self.final_refit:
            coef_s, intercept_s, _ = self._fixed_ridge_refit_scaled(X_tr, y_tr, idx)
        else:
            coef_s = np.asarray(row[1], dtype=float).ravel()
            intercept_s = float(row[2])

        if idx.size == 0:
            eta = np.full(X_va.shape[0], intercept_s, dtype=float)
        else:
            eta = X_va[:, idx] @ coef_s + intercept_s

        p1 = self._sigmoid(np.clip(eta, -self.clp, self.clp))
        return log_loss(y_va, p1, labels=[0, 1])

    def fit(self, X, y, feature_names=None, val_size=0.1, verbose=False):
        X = np.asarray(X)
        y = np.asarray(y).ravel().astype(int)

        self.bad_col_ = self._filter_invalid_cols(X)
        Xf = np.delete(X, self.bad_col_, axis=1)
        self.p_filtered_ = Xf.shape[1]

        if feature_names is None or len(feature_names) != X.shape[1]:
            fn = np.array([f"X_{j}" for j in range(X.shape[1])], dtype=object)
        else:
            fn = np.asarray(feature_names, dtype=object)

        self.feature_names_ = np.delete(fn, self.bad_col_)

        if self.p_filtered_ == 0:
            raise ValueError("All columns were removed because they contained NaN or infinite values.")

        if not self.optimize_k:
            Xn = self._fit_transform_X(Xf)

            self.best_k_ = self.k
            self._fit_fixed_k(
                Xn=Xn,
                y=y,
                feature_names=self.feature_names_,
                stopping=self.k,
                verbose=verbose,
            )

        else:
            Xf_tr, Xf_va, y_tr, y_va = train_test_split(
                Xf,
                y,
                test_size=val_size,
                random_state=self.random_state,
                stratify=y if len(np.unique(y)) == 2 else None,
            )

            # Fit scaling on the fitting subset only.
            X_tr = self._fit_transform_X(Xf_tr)
            X_va = self._transform_X(Xf_va)

            self._fit_fixed_k(
                Xn=X_tr,
                y=y_tr,
                feature_names=self.feature_names_,
                stopping=self.k,
                verbose=verbose,
            )

            self.validation_intermediates_ = None if self.intermediates_ is None else self.intermediates_.copy()

            if self.validation_intermediates_ is None or self.validation_intermediates_.shape[0] == 0:
                raise RuntimeError("No intermediate models were fitted during validation.")

            best_k = 1
            best_loss = np.inf

            for kk in range(self.validation_intermediates_.shape[0]):
                loss = self._validation_loss_for_intermediate(
                    X_tr=X_tr,
                    y_tr=y_tr,
                    X_va=X_va,
                    y_va=y_va,
                    row=self.validation_intermediates_[kk],
                )

                if loss < best_loss:
                    best_k = kk + 1
                    best_loss = loss

            self.best_k_ = best_k
            self.best_validation_loss_ = best_loss

            if verbose:
                print(f"refitting with k={best_k} (val logloss={best_loss:.6g})")

            # Final refit on all supplied training data, with scaling refit on all data.
            Xn = self._fit_transform_X(Xf)

            self._fit_fixed_k(
                Xn=Xn,
                y=y,
                feature_names=self.feature_names_,
                stopping=best_k,
                verbose=verbose,
            )

        # Optional fixed-ridge post-selection refit on the final selected support.
        if self.final_refit:
            idx = np.asarray(self.beta_idx_, dtype=int)

            beta_s, intercept_s, refit_model = self._fixed_ridge_refit_scaled(
                Xn=Xn,
                y=y,
                idx=idx,
            )

            self.beta_scaled_ = beta_s.copy()
            self.intercept_scaled_ = float(intercept_s)
            self.final_refit_model_ = refit_model

            self._unscale_coef(idx, beta_s, intercept_s)

        self.selected_features_ = np.asarray(self.beta_idx_, dtype=int)
        self.feature_names = self.feature_names_[self.selected_features_]

        return self

    def decision_function(self, X):
        Xf = self._maybe_filter_X(X)

        if len(self.beta_idx_) == 0:
            return np.full(Xf.shape[0], self.intercept_, dtype=float)

        return Xf[:, self.beta_idx_] @ self.coef_.ravel() + self.intercept_

    def predict_proba(self, X):
        eta = self.decision_function(X)
        p1 = self._sigmoid(np.clip(eta, -self.clp, self.clp))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def negative_gradient(self, X, y):
        eta = self.decision_function(X)
        p = self._sigmoid(np.clip(eta, -self.clp, self.clp))
        return np.asarray(y).ravel() - p

    def get_params(self, deep=True):
        return {
            "sis": self.sis,
            "so": self.so,
            "k": self.k,
            "fit_intercept": self.fit_intercept,
            "normalize": self.normalize,
            "optimize_k": self.optimize_k,
            "track_intermediates": self.track_intermediates,
            "clp": self.clp,
            "random_state": self.random_state,
            "final_refit": self.final_refit,
            "final_refit_C": self.final_refit_C,
            "final_refit_tol": self.final_refit_tol,
            "final_refit_max_iter": self.final_refit_max_iter,
        }

    def __str__(self):
        return f"LogisticICL({self.get_params()})"

    def __repr__(self, prec=3):
        coef = getattr(self, "coef_", None)
        intercept = float(getattr(self, "intercept_", 0.0))
        idx = getattr(self, "beta_idx_", None)

        if coef is None or idx is None or len(idx) == 0:
            return (
                ("+" if intercept > 0 else "")
                + np.format_float_scientific(intercept, precision=prec, unique=False)
            )

        coef = np.asarray(coef).ravel()
        idx = np.asarray(idx, dtype=int)

        if getattr(self, "feature_names_", None) is not None:
            names = np.asarray(self.feature_names_)[idx]
        else:
            names = np.array([f"X_{j}" for j in idx], dtype=object)

        out = []

        for c, name in zip(coef, names):
            out.append(
                ("+" if float(c) > 0 else "")
                + np.format_float_scientific(float(c), precision=prec, unique=False)
                + " ("
                + str(name)
                + ")\n"
            )

        out.append(
            ("+" if intercept > 0 else "")
            + np.format_float_scientific(intercept, precision=prec, unique=False)
        )

        return "".join(out)

if __name__ == "__main__":
    pass
