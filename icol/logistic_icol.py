import warnings
warnings.filterwarnings('ignore')

from time import time
from copy import deepcopy

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
    sigma_X = np.std(X, axis=0)
    sigma_Y = np.std(g)

    XY = X*g.reshape(-1, 1)
    E_XY = np.mean(XY, axis=0)
    E_X = np.mean(X, axis=0)
    E_Y = np.mean(g)
    cov = E_XY - E_X*E_Y
    sigma = sigma_X*sigma_Y
    pearsons = cov/sigma
    absolute_pearsons = np.abs(pearsons)
    absolute_pearsons[np.isnan(absolute_pearsons)] = 0 # setting all rows of constants to have 0 correlation
    absolute_pearsons[np.isinf(absolute_pearsons)] = 0 # setting all rows of constants to have 0 correlation
    absolute_pearsons[np.isneginf(absolute_pearsons)] = 0

    return absolute_pearsons 

def squared(X, y, model):
    y_hat = model.predict(X).ravel()
    res = y - y_hat
    return corr(X, res)

def df_log_loss(X, y, model, clp=np.infty):
    eta = model.decision_function(X).ravel()      # real-valued score
    p = 1.0 / (1.0 + np.exp(-np.clip(eta, -clp, clp)))
    g = y - p
    return corr(X, g)

OBJECTIVE_DICT = {
    'squared': squared,
    'logistic': df_log_loss
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

class generalised_SIS:
    def __init__(self, s, obj='squared'):
        self.s=s
        self.obj=obj

    def __str__(self):
        return 'SIS(s={0}, obj={1})'.format(self.s, self.obj)
    
    def __repr__(self):
        return self.__str__()
    
    def __call__(self, X, y, model, pool):
        scores = OBJECTIVE_DICT[self.obj](X=X, y=y, model=model)
        idxs = np.argsort(scores)[::-1]

        pool_set = set(pool)
        chosen = []
        for j in idxs:
            if j not in pool_set:
                chosen.append(j)
                if len(chosen) == self.s:
                    break

        chosen = np.array(chosen, dtype=int)
        return scores[chosen], chosen

class THRESHOLDED_LOGISTIC_REGRESSION:
    def __init__(self, random_state):
        self.random_state = random_state

    def __str__(self):
        return "THRESHOLDED_LOGISTIC_REGRESSION"
    
    def get_params(self, deep=True):
        return {'random_state': self.random_state}
    
    def fit(self, X, y, d, feature_names=None, verbose=False):
        if feature_names is None: feature_names = np.array(['X{0}'.format(i) for i in range(X.shape[1])])

        LR = LogisticRegression(penalty=None, fit_intercept=False, random_state=self.random_state)
        if verbose: print('Fitting on full candidate pool')
        LR.fit(X, y)
        if verbose: print('Selecting top {0} candidates'.format(d))
        beta_hat_all = np.abs(LR.coef_.squeeze())
        idxs = np.argsort(beta_hat_all)[::-1]
        use_idxs = idxs[:d]
        if verbose: print('Fitting on selected candidates')
        LR.fit(X[:, use_idxs], y)
        if verbose: print('Saving sparse base estimator')
        beta_hat_sparse = LR.coef_.squeeze()
        beta_hat = np.zeros(X.shape[1])
        beta_hat[use_idxs] = beta_hat_sparse
        self.coef_ = beta_hat
        self.feature_names = feature_names[use_idxs]
        self.model = LR
        return self

    def __repr__(self, prec=3):
        return self.__str__()

    def decision_function(self, X):
        return np.dot(X, self.coef_.ravel())
    
    def predict_proba(self, X):
        z = self.decision_function(X)
        z = np.clip(z, -self.clp, self.clp)  # numerical stability
        p1 = 1.0 / (1.0 + np.exp(-z))
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])
    
    def predict(self, X , threshold=0.5):
        proba = self.predict_proba(X)
        p1 = proba[:, 1]
        return (p1 >= threshold).astype(int)

class BSS_LOGISTIC:
    def __init__(self, random_state, metric='logloss'):
        self.random_state = random_state
        self.metric = metric
        self.loss = LOSS_DICT[metric]

    def __str__(self):
        return "THRESHOLDED_LOGISTIC_REGRESSION"
    
    
    def fit(self, X, y, d, feature_names=None, verbose=False):
        if feature_names is None: feature_names = np.array(['X{0}'.format(i) for i in range(X.shape[1])])
        n, p  = X.shape
        all_comb = combinations(range(p), d)
        base = LogisticRegression(penalty=None, fit_intercept=False, random_state=self.random_state)
        best_i, best_comb, best_loss, best_coef = None, None, None, None
        for i, comb in enumerate(all_comb):
            if verbose: print('Attempting Combination: {0}'.format(comb))
            LR = clone(base)
            LR.fit(X[:, comb], y)
            y_hat = LR.predict(X[:, comb])
            loss = self.loss(y, y_hat)
            if (best_i is None) or (loss < best_loss):
                best_i, best_comb, best_loss, best_coef = i, comb, loss, LR.coef_

        if verbose: print('The best combination was {0} with a loss of {1}'.format(best_comb, best_loss))
        self.use_idxs = list(best_comb)
        beta_hat_sparse = best_coef.ravel()
        beta_hat = np.zeros(X.shape[1])
        beta_hat[self.use_idxs] = beta_hat_sparse
        self.coef_ = beta_hat
        self.feature_names = feature_names[self.use_idxs]
        if verbose: print('Corresponding to the features: {0}'.format(str(self.feature_names)))
        return self
    
    def __repr__(self, prec=3):
        return self.__str__()
    
    def get_params(self, deep=True):
        return {'random_state': self.random_state,
                'metric': self.metric}

    def decision_function(self, X):
        return np.dot(X, self.coef_.ravel())
    
    def predict_proba(self, X):
        z = self.decision_function(X)
        z = np.clip(z, -self.clp, self.clp)  # numerical stability
        p1 = 1.0 / (1.0 + np.exp(-z))
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])
    
    def predict(self, X , threshold=0.5):
        proba = self.predict_proba(X)
        p1 = proba[:, 1]
        return (p1 >= threshold).astype(int)

class LOGISTIC_ADALASSO:
    def __init__(self, log_c_lo=-4, log_c_hi=3, c_num=100,  solver="saga",
                 class_weight=None, max_iter=5000, tol=1e-4, eps_nnz=1e-12, 
                 clp=np.infty, random_state=None, gamma=1, search_C = 'linear'):
        self.log_c_lo = log_c_lo
        self.log_c_hi = log_c_hi
        self.c_num= c_num
        self.C_grid = np.sort(np.asarray(np.logspace(self.log_c_lo, self.log_c_hi, self.c_num), dtype=float))
        self.solver = solver
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.tol = tol
        self.eps_nnz = eps_nnz
        self.random_state = random_state
        self.clp = clp
        self.gamma = gamma
        self.search_C = search_C

        self.models = np.array([LogisticRegression(C=c, 
                           solver=self.solver, class_weight=self.class_weight, 
                           max_iter=self.max_iter, tol=self.tol, random_state=random_state,
                           penalty='l1', l1_ratio=1, fit_intercept=False,  
                           ) for c in self.C_grid], dtype=object)
        
    def get_params(self, deep=True):
        return {
            'log_c_lo': self.log_c_lo,
            'log_c_hi': self.log_c_hi,
            'c_num': self.c_num,
            "solver": self.solver,
            "class_weight": self.class_weight,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "eps_nnz": self.eps_nnz,
            "random_state": self.random_state,
            'clp': self.clp,
            'search_C': self.search_C
        }

    def fit(self, X, y, d, feature_names=None, verbose=False):
        self.feature_names = ['X_{0}'.format(i) for i in range(X.shape[1])] if feature_names is None else feature_names

        nonancols = np.isnan(X).sum(axis=0)==0
        noinfcols = np.isinf(X).sum(axis=0)==0
        valcols = np.logical_and(nonancols, noinfcols)
        if np.abs(self.gamma) <= 1e-10:
            beta_hat = np.ones(X.shape[1])
            w_hat = np.ones(X.shape[1])
            X_star_star = X.copy()
        else:
            X_valcols = X[:, valcols]
            LR = LogisticRegression(penalty=None, fit_intercept=False, random_state=self.random_state)
            LR.fit(X_valcols, y)
            beta_hat = LR.coef_.squeeze()
            if beta_hat.shape == ():
                beta_hat = beta_hat.reshape(-1, 1)
            
            w_hat = 1/np.power(np.abs(beta_hat)+self.eps_nnz, self.gamma).ravel()
            X_star_star = X_valcols / w_hat

        # best_idx = 0
        # for i, model in enumerate(self.models):
        #     if verbose: print('Fitting model {0} of {1} with C={2} and has '.format(i, len(self.models), model.C), end='')
        #     model.fit(X_star_star, y)
        #     nnz = self._count_nnz(model.coef_)
        #     if verbose: print('{0} nonzero terms'.format(nnz))
        #     if nnz<=d:
        #         best_idx = i
        #     else:
        #         break

        best_idx, fitted_models = SEARCH_DICT[self.search_C](models=self.models, X=X_star_star, y=y, d=d, eps_nnz=self.eps_nnz, verbose=verbose)

        self.model_idx = best_idx
        beta_hat_star_star = self.models[self.model_idx].coef_.ravel()

        self.coef_idx_ = np.arange(len(beta_hat_star_star))[np.abs(np.ravel(beta_hat_star_star)) > self.eps_nnz]

        beta_hat_star_n_valcol = np.array([beta_hat_star_star[j]/w_hat[j] for j in range(len(beta_hat_star_star))])

        beta_hat_star_n = np.zeros(X.shape[1])
        beta_hat_star_n[valcols] = beta_hat_star_n_valcol
        
        self.coef_ = beta_hat_star_n
        self.model = self.models[self.model_idx]
        self.model.coef_ = self.coef_

        return self
    
    def _count_nnz(self, coef):
        return int(np.sum(
            np.abs(np.ravel(coef)) > self.eps_nnz
            ))
    
    def __str__(self):
        return "Logistic{0}Lasso".format("Ada" if np.abs(self.gamma)>=1e-10 else '')
    
    def __repr__(self, prec=3):
        return self.__str__()

    def decision_function(self, X):
        return np.dot(X, self.model.coef_.ravel())
    
    def predict_proba(self, X):
        z = self.decision_function(X)
        z = np.clip(z, -self.clp, self.clp)  # numerical stability
        p1 = 1.0 / (1.0 + np.exp(-z))
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])
    
    def predict(self, X , threshold=0.5):
        proba = self.predict_proba(X)
        p1 = proba[:, 1]
        return (p1 >= threshold).astype(int)
        
sci = lambda x, sig=3: f"{float(x):.{sig}e}"

class LogisticICL:
    def __init__(self, sis, so, k, 
                 fit_intercept=True, normalize=True, pool_reset=False, optimize_k=True,
                 track_intermediates=False, clp=30):
        self.sis = sis
        self.so = so
        self.k = int(k)

        self.fit_intercept = bool(fit_intercept)
        self.normalize = bool(normalize)
        self.pool_reset = bool(pool_reset)
        self.optimize_k = bool(optimize_k)
        self.track_intermediates = True if self.optimize_k else bool(track_intermediates)
        self.clp = int(clp)

        # learned
        self.bad_col_ = None
        self.p_filtered_ = None
        self.feature_names_ = None

        self.a_x_ = None
        self.b_x_ = None

        self.beta_idx_ = []
        self.beta_scaled_ = np.zeros(0, dtype=float)
        self.coef_ = np.zeros((1, 0), dtype=float)
        self.intercept_ = 0.0

        self.intermediates_ = np.empty((self.k, 5), dtype=object) # idx, coef, inter, names, repr

    @staticmethod
    def _sigmoid(z):
        # stable sigmoid
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

        self.a_x_ = X.mean(axis=0) if self.fit_intercept else np.zeros(p)
        self.b_x_ = X.std(axis=0)
        self.b_x_ = np.where(self.b_x_ == 0, 1.0, self.b_x_)
        return (X - self.a_x_) / self.b_x_

    def _unscale_coef(self, idx, beta_scaled, intercept_scaled):
        idx = np.asarray(idx, dtype=int)
        beta_scaled = np.asarray(beta_scaled, dtype=float).ravel()

        if idx.size == 0:
            self.coef_ = np.zeros((1, 0))
            self.intercept_ = float(intercept_scaled) if self.fit_intercept else 0.0
            return

        if self.normalize:
            coef_raw = beta_scaled / self.b_x_[idx]
            inter_raw = float(intercept_scaled)
            if self.fit_intercept:
                inter_raw = float(inter_raw - self.a_x_[idx] @ coef_raw)
            else:
                inter_raw = 0.0
        else:
            coef_raw = beta_scaled
            inter_raw = float(intercept_scaled) if self.fit_intercept else 0.0

        self.coef_ = coef_raw.reshape(1, -1)
        self.intercept_ = inter_raw

    def _fit_fixed_k(self, Xn, y, feature_names, stopping, verbose=False, track_pool=False):
        n, p = Xn.shape
        pool = set()

        if self.fit_intercept:
            pbar = float(np.mean(y))
            pbar = min(max(pbar, 1e-12), 1 - 1e-12)
            intercept_s = float(np.log(pbar / (1 - pbar)))
        else:
            intercept_s = 0.0

        beta = np.zeros(p, dtype=float)

        for i in range(stopping):
            if verbose:
                print(".", end="")

            _, sis_i = self.sis(X=Xn, y=y, model=self, pool=list(pool))
            pool_old = deepcopy(pool)
            pool.update(sis_i)
            pool_lst = list(pool)

            self.so.fit(X=Xn[:, pool_lst], y=y, d=i+1,
                        feature_names=feature_names[pool_lst],
                        verbose=verbose)
            beta_pool = np.asarray(self.so.coef_).ravel()
            intercept_s = float(getattr(self.so, "intercept_", intercept_s))

            beta[:] = 0.0
            beta[pool_lst] = beta_pool

            if self.pool_reset:
                keep = np.abs(beta_pool) > 0
                pool_lst = np.asarray(pool_lst)[keep].ravel().tolist()
                pool = set(pool_lst)
                beta[:] = 0.0
                beta[pool_lst] = beta_pool[keep]

            idx = np.nonzero(beta)[0].tolist()
            beta_sparse = beta[idx]
            self.beta_idx_ = idx
            self.beta_scaled_ = beta_sparse
            self._unscale_coef(idx, beta_sparse, intercept_s)

            if self.track_intermediates:
                self.intermediates_[i, 0] = np.array(idx, dtype=int)
                self.intermediates_[i, 1] = beta_sparse.copy()
                self.intermediates_[i, 2] = float(self.intercept_)
                self.intermediates_[i, 3] = feature_names[idx]
                self.intermediates_[i, 4] = None

        if verbose:
            print()

        self.beta_idx_ = np.nonzero(beta)[0].tolist()
        self.beta_scaled_ = beta[self.beta_idx_]

    def fit(self, X, y, feature_names=None, val_size=0.1, random_state=None, verbose=False):
        X = np.asarray(X)
        y = np.asarray(y).ravel().astype(int)

        self.bad_col_ = self._filter_invalid_cols(X)
        Xf = np.delete(X, self.bad_col_, axis=1)
        self.p_filtered_ = Xf.shape[1]

        if feature_names is None or len(feature_names) != X.shape[1]:
            fn = np.array([f"X_{j}" for j in range(X.shape[1])])
        else:
            fn = np.asarray(feature_names)
        self.feature_names_ = np.delete(fn, self.bad_col_)

        Xn = self._fit_transform_X(Xf)
        if not self.optimize_k:
            self._fit_fixed_k(Xn, y, self.feature_names_, stopping=self.k, verbose=verbose)
        else:
            X_tr, X_va, y_tr, y_va = train_test_split(
                Xn, y, test_size=val_size, random_state=random_state
            )
            self._fit_fixed_k(X_tr, y_tr, self.feature_names_, stopping=self.k, verbose=verbose)

            best_k, best_loss = 0, np.inf
            for kk in range(self.k):
                idx = self.intermediates_[kk, 0]
                coef = self.intermediates_[kk, 1]
                inter = self.intermediates_[kk, 2]
                eta = X_va[:, idx] @ coef + inter
                p1 = self._sigmoid(np.clip(eta, -30, 30))
                loss = log_loss(y_va, p1)

                if loss < best_loss:
                    best_k, best_loss = kk + 1, loss

            if verbose:
                print(f"refitting with k={best_k} (val logloss={best_loss:.6g})")

            self._fit_fixed_k(Xn, y, self.feature_names_, stopping=best_k, verbose=verbose)

        if len(self.beta_idx_) > 0:
            lr = LogisticRegression(penalty=None, solver="lbfgs", fit_intercept=True)
            lr.fit(Xn[:, self.beta_idx_], y)
            beta_s = lr.coef_.ravel()
            intercept_s = float(lr.intercept_[0])

            self.beta_scaled_ = beta_s
            self._unscale_coef(self.beta_idx_, beta_s, intercept_s)
        else:
            self.coef_ = np.zeros((1, 0), dtype=float)

        return self

    def decision_function(self, X):
        Xf = self._maybe_filter_X(X)
        if len(self.beta_idx_) == 0:
            return np.full(Xf.shape[0], self.intercept_, dtype=float)
        return Xf[:, self.beta_idx_] @ self.coef_.ravel() + self.intercept_

    def predict_proba(self, X):
        eta = self.decision_function(X)
        p1 = self._sigmoid(np.clip(eta, -30, 30))
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
            "pool_reset": self.pool_reset,
            "optimize_k": self.optimize_k,
            "track_intermediates": self.track_intermediates,
            "clp": self.clp,
        }

    def __str__(self):
        return f"LogisticICL({self.get_params()})"

    def __repr__(self, prec=3):
        coef = getattr(self, "coef_", None)
        intercept = float(getattr(self, "intercept_", 0.0))
        idx = getattr(self, "beta_idx_", None)

        # Intercept-only or not yet fitted
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
