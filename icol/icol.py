import warnings
warnings.filterwarnings('ignore')

from time import time
from copy import deepcopy
from itertools import combinations, permutations

import numpy as np
import sympy as sp

from sklearn.linear_model import lars_path, Ridge, Lars
from sklearn.preprocessing import PolynomialFeatures
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegressionCV

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)

def LL(res):
    n = len(res)
    return n*np.log(np.sum(res**2)/n)

def initialize_ols(D, y, init_idx):
    """
    Fit initial OLS solution on selected columns of D.
    
    Parameters
    ----------
    D : (n, d) ndarray
        Full dictionary matrix.
    y : (n,) ndarray
        Response vector.
    init_idx : list[int]
        Indices of columns from D to use initially.
    
    Returns
    -------
    beta : (p,) ndarray
        OLS coefficients for selected columns.
    A_inv : (p, p) ndarray
        Inverse Gram matrix for selected columns.
    XT : (p, n) ndarray
        Transposed design matrix of selected columns.
    active_idx : list[int]
        Current indices of D included in the model.
    """
    X = D[:, init_idx]
    A = X.T @ X
    try: 
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        A_inv = np.linalg.pinv(A)
    beta = A_inv @ (X.T @ y)
    XT = X.T
    return beta, A_inv, XT, list(init_idx)

def sweep_update_from_D(beta, A_inv, XT, active_idx, D, y, new_idx):
    # Generated with ChatGPT using the commands;
    # 1. write me a function which takes in an n by p dimension matrix X, for which we already have an OLS solution, beta.
    #  Additionally, a second input is a data matrix Z with n rows and q columns. 
    # Add the Z matrix of columns to the OLS solution using SWEEP
    # 2. Are we also able to efficiently update the gram and its inverse with this procedure for X augmented with Z
    # 3. Ok, imagine that I need to update my SWEEP solution multiple times.
    #  Adjust the inputs and return values so that everything can be used again in the next SWEEP update.
    #  Then update the function to make use of these previous computations
    # 4. Lets make some changes for the sake of indexing. Imagine that we have a large matrix D, with d columns.
    # Through some selection procedure we select p of those columns to form an initial OLS solution.
    # We then iteratively select p new columns and incorporate those into the ols solution using sweep. 
    # Update the code to reflect this change while also tracking the indices of columns in the original D matrix 
    # and their mapping to the respective betas.

    """
    Update OLS solution by adding new columns from D.
    
    Parameters
    ----------
    beta : (p,) ndarray
        Current OLS coefficients.
    A_inv : (p, p) ndarray
        Inverse Gram matrix for current features.
    XT : (p, n) ndarray
        Transposed design matrix for current features.
    active_idx : list[int]
        Current indices of columns in D that are in the model.
    D : (n, d) ndarray
        Full dictionary matrix.
    y : (n,) ndarray
        Response vector.
    new_idx : list[int]
        Indices of new columns in D to add.
    
    Returns
    -------
    beta_new : (p+q,) ndarray
        Updated OLS coefficients.
    A_tilde_inv : (p+q, p+q) ndarray
        Updated inverse Gram matrix.
    XT_new : (p+q, n) ndarray
        Updated design matrix transpose.
    active_idx_new : list[int]
        Updated indices of active columns in D.
    """
    p = beta.shape[0]
    Z = D[:, new_idx]    # n x q
    q = Z.shape[1]

    # Cross products
    B = XT @ Z                # p x q
    C = Z.T @ Z               # q x q
    yZ = Z.T @ y              # q x 1

    # Schur complement
    S = C - B.T @ (A_inv @ B)

    # Solve for new coefficients (numerically stable)
    rhs = yZ - B.T @ beta
    try:
        beta_Z = np.linalg.solve(S, rhs)
    except np.linalg.LinAlgError:
        beta_Z = np.linalg.pinv(S) @ rhs

    # Update old coefficients
    beta_X_new = beta - A_inv @ (B @ beta_Z)
    beta_new = np.concatenate([beta_X_new, beta_Z])

    # Update Gram inverse
    try: 
        S_inv = np.linalg.inv(S)  # small q x q
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)

    top_left = A_inv + A_inv @ B @ S_inv @ B.T @ A_inv
    top_right = -A_inv @ B @ S_inv
    bottom_left = -S_inv @ B.T @ A_inv
    bottom_right = S_inv

    A_tilde_inv = np.block([
        [top_left, top_right],
        [bottom_left, bottom_right]
    ])

    # Update XT and active indices
    XT_new = np.vstack([XT, Z.T])
    active_idx_new = active_idx + list(new_idx)

    return beta_new, A_tilde_inv, XT_new, active_idx_new

IC_DICT = {
    'AIC': lambda res, k: LL(res) + 2*k,
    'HQIC': lambda res, k: LL(res) + np.log(np.log(len(res)))*k,
    'BIC': lambda res, k, n: LL(res) + 2*k*np.log(n),
    'CAIC': lambda res, k: LL(res) + (np.log(len(res))+1)*k,
    'AICc': lambda res, k: LL(res) + 2*k + 2*k*(k+1)/(len(res)-k-1)
}

OP_DICT = {
    'sin': {
        'op': sp.sin,
        'op_np': np.sin,
        'inputs': 1,
        'commutative': True,
        'cares_units': False
        },
    'cos': {
        'op': sp.cos,
        'op_np': np.cos,
        'inputs': 1,
        'commutative': True,
        'cares_units': False
        },
    'log': {
        'op': sp.log,
        'op_np': np.log,
        'inputs': 1,
        'commutative': True,
        'cares_units': False
        },
    'exp': {
        'op': sp.exp,
        'op_np': np.exp,
        'inputs': 1,
        'commutative': True,
        'cares_units': False
        },
    'abs': {
        'op': sp.Abs,
        'op_np': np.abs,
        'inputs': 1,
        'commutative': True,
        'cares_units': False
        },
    'sqrt': {
        'op': sp.sqrt,
        'op_np': np.sqrt,
        'inputs': 1,
        'commutative': True,
        'cares_units': False
        },
    'cbrt': {
        'op': lambda x: sp.Pow(x, sp.Rational(1, 3)),
        'op_np': lambda x: np.power(x, 1/3),
        'inputs': 1,
        'commutative': True,
        'cares_units': False
        },
    'sq': {
        'op': lambda x: sp.Pow(x, 2),
        'op_np': lambda x: np.power(x, 2),
        'inputs': 1,
        'commutative': True,
        'cares_units': False
        },
    'cb': {
        'op': lambda x: sp.Pow(x, 3),
        'op_np': lambda x: np.power(x, 3),
        'inputs': 1,
        'commutative': True,
        'cares_units': False
        },
    'six_pow': {
        'op': lambda x: sp.Pow(x, 6),
        'op_np': lambda x: np.power(x, 6),
        'inputs': 1,
        'commutative': True,
        'cares_units': False
        },
    'inv': {
        'op': lambda x: 1/x,
        'op_np': lambda x: 1/x,
        'inputs': 1,
        'commutative': True,
        'cares_units': False
        },
    'mul': {
        'op': sp.Mul,
        'op_np': np.multiply,
        'inputs': 2,
        'commutative': True,
        'cares_units': False
        },
    'div': {
        'op': lambda x, y: sp.Mul(x, 1/y),
        'op_np': lambda x, y: np.multiply(x, 1/y),
        'inputs': 2,
        'commutative': False,
        'cares_units': False
        },
    'add': {
        'op': sp.Add,
        'op_np': lambda x, y: x+y,
        'inputs': 2,
        'commutative': True,
        'cares_units': False
        },
    'sub': {
        'op': lambda x, y: sp.Add(x, -y),
        'op_np': lambda x, y: x-y,
        'inputs': 2,
        'commutative': False,
        'cares_units': False
        },
    'abs_diff': {
        'op': lambda x, y: sp.Abs(sp.Add(x, -y)),
        'op_np': lambda x, y: np.abs(x-y),
        'inputs': 2,
        'commutative': True,
        'cares_units': False
        },
    }

class PolynomialFeaturesICL:
    def __init__(self, rung, include_bias=False):
        self.rung = rung
        self.include_bias = include_bias
        self.PolynomialFeatures = PolynomialFeatures(degree=self.rung, include_bias=self.include_bias)

    def __str__(self):
        return 'PolynomialFeatures(degree={0}, include_bias={1})'.format(self.rung, self.include_bias)

    def __repr__(self):
        return self.__str__()

    def fit(self, X, y=None):
        self.PolynomialFeatures.fit(X, y)
        return self
    
    def transform(self, X):
        return self.PolynomialFeatures.transform(X)

    def fit_transform(self, X, y=None):
        return self.PolynomialFeatures.fit_transform(X, y)
    
    def get_feature_names_out(self):
        return self.PolynomialFeatures.get_feature_names_out()

class BSS:
    def __init__(self):
        pass

    def get_params(self, deep=False):
        return {}

    def __str__(self):
        return 'BSS'

    def __repr__(self):
        return 'BSS'
    
    def gen_V(self, X, y):
        n, p = X.shape
        XtX = np.dot(X.T, X)
        Xty = np.dot(X.T, y).reshape(p, 1)
        yty = np.dot(y.T, y)
        V = np.hstack([XtX, Xty])
        V = np.vstack([V, np.vstack([Xty, yty]).T])
        return V

    def s_max(self, k, n, p, c0=0, c1=1):
        return c1*np.power(p, 1/k) + c0
    
    def add_remove(self, V, k):
        n, p = V.shape
        td = V[k, k]
        V[k, :] = V[k, :]/td
        I = np.arange(start=0, stop=n, dtype=int)
        I = np.delete(I, k)
        ct = V[I, k].reshape(-1, 1)
        z = np.dot(ct, V[k, :].reshape(1, -1))
        V[I, :] = V[I, :] - z
        V[I, k] = -ct.squeeze()/td
        V[k, k] = 1/td

    def sweep(self, V, K):
        for k in K:
            self.add_remove(V, k)

    def __call__(self, X, y, d, verbose=False):
        n, p = X.shape
        combs = combinations(range(p), d)
        comb_curr = set([])
        V = self.gen_V(X, y)
        best_comb, best_rss = None, None
        for i, comb in enumerate(combs):
            if verbose: print(comb)
            comb = set(comb)
            new = comb - comb_curr
            rem = comb_curr - comb
            comb_curr = comb
            changes = list(new.union(rem))
            self.sweep(V, changes)
            rss = V[-1, -1]
            if (best_rss is None) or (best_rss > rss):
                best_comb = comb
                best_rss = rss
        beta, _, _, _ = np.linalg.lstsq(a=X[:, list(best_comb)], b=y)
        beta_ret = np.zeros(p)
        beta_ret[list(best_comb)] = beta.reshape(1, -1)
        return beta_ret

class EfficientAdaptiveLASSO:
    def __init__(self, gamma=1, fit_intercept=False, default_d=5, rcond=-1, alpha=0):
        self.gamma = gamma
        self.fit_intercept = fit_intercept
        self.default_d = default_d
        self.rcond=rcond
        self.alpha=alpha
        self.A_inv = None
        self.XT = None
        self.beta_ols = None
        self.active_idx = None

    def __str__(self):
        return ('EffAda' if self.gamma != 0 else '') + ('LASSO') + ('(gamma={0})'.format(self.gamma) if self.gamma != 0 else '')
    
    def __repr__(self):
        return self.__str__()
    
    def get_params(self, deep=False):
        return {'gamma': self.gamma,
                'fit_intercept': self.fit_intercept,
                'default_d': self.default_d,
                'rcond': self.rcond}
    
    def set_default_d(self, d):
        self.default_d = d

    def __call__(self, X, y, d, idx_old = None, idx_new=None, verbose=False):

        self.set_default_d(d)
        nonancols = np.isnan(X).sum(axis=0)==0
        noinfcols = np.isinf(X).sum(axis=0)==0
        valcols = np.logical_and(nonancols, noinfcols)
        idx_ala = list(idx_new) + list(idx_old)

        if np.abs(self.gamma)<1e-10:
            beta_ols = np.ones(X.shape[1])
            w_hat = np.ones(X.shape[1])
            X_star_star = X.copy()
        else:
            X_valcols = X[:, valcols]
            if not idx_old:
                self.beta_ols, self.A_inv, self.XT, self.active_idx = initialize_ols(X_valcols, y, init_idx=idx_new)
            else:
                self.beta_ols, self.A_inv, self.XT, self.active_idx = sweep_update_from_D(beta = self.beta_ols, A_inv=self.A_inv,
                                                                                          XT=self.XT, active_idx=self.active_idx, D=X, y=y, 
                                                                                          new_idx=idx_new)

            w_hat = 1/np.power(np.abs(self.beta_ols), self.gamma)
            X_star_star = np.zeros_like(X_valcols[:, idx_ala])
            for j in range(X_star_star.shape[1]): # vectorise
                X_j = X_valcols[:, j]/w_hat[j]
                X_star_star[:, j] = X_j

        _, _, coefs, _ = lars_path(X_star_star, y.ravel(), return_n_iter=True, max_iter=d, method='lasso')
        # alphas, active, coefs = lars_path(X_star_star, y.ravel(), method='lasso')
        try:           
            beta_hat_star_star = coefs[:, d]
        except IndexError: # in the event that a solution with d components cant be found, use the next largest. 
            beta_hat_star_star = coefs[:, -1]

        beta_hat_star_n_old_new = np.array([beta_hat_star_star[j]/w_hat[j] for j in range(len(beta_hat_star_star))])
#        beta_hat_star_n = np.zeros(X.shape[1])
#        beta_hat_star_n[idx_ala] = beta_hat_star_n_old_new

#        beta_hat_star_n[valcols] = beta_hat_star_n_valcol
#        ret = beta_hat_star_n.reshape(1, -1).squeeze()
        return beta_hat_star_n_old_new.squeeze()
    
    def fit(self, X, y, verbose=False):
        self.mu = y.mean() if self.fit_intercept else 0            
        beta = self.__call__(X=X, y=y-self.mu, d=self.default_d, verbose=verbose)
        self.beta = beta.reshape(-1, 1)

    def predict(self, X):
        return np.dot(X, self.beta) + self.mu
    
    def s_max(self, k, n, p, c1=1, c0=0):
        if self.gamma==0:
            return c1*(p/(k**2)) + c0
        else:
            return c1*min(np.power(p, 1/2)/k, np.power(p*n, 1/3)/k) + c0

class AdaptiveLASSO:
    def __init__(self, gamma=1, fit_intercept=False, default_d=5, rcond=-1, alpha=0):
        self.gamma = gamma
        self.fit_intercept = fit_intercept
        self.default_d = default_d
        self.rcond=rcond
        self.alpha=0

    def __str__(self):
        return ('Ada' if self.gamma != 0 else '') + ('LASSO') + ('(gamma={0})'.format(self.gamma) if self.gamma != 0 else '')
    
    def __repr__(self):
        return self.__str__()
    
    def get_params(self, deep=False):
        return {'gamma': self.gamma,
                'fit_intercept': self.fit_intercept,
                'default_d': self.default_d,
                'rcond': self.rcond}
    
    def set_default_d(self, d):
        self.default_d = d

    def __call__(self, X, y, d, verbose=False):

        self.set_default_d(d)

        nonancols = np.isnan(X).sum(axis=0)==0
        noinfcols = np.isinf(X).sum(axis=0)==0
        valcols = np.logical_and(nonancols, noinfcols)
        if np.abs(self.gamma)<1e-10:
            beta_hat = np.ones(X.shape[1])
            w_hat = np.ones(X.shape[1])
            X_star_star = X.copy()
        else:

            X_valcols = X[:, valcols]
            beta_hat, _, _, _ = np.linalg.lstsq(X_valcols, y, rcond=self.rcond)

            w_hat = 1/np.power(np.abs(beta_hat), self.gamma)
            X_star_star = np.zeros_like(X_valcols)
            for j in range(X_star_star.shape[1]): # vectorise
                X_j = X_valcols[:, j]/w_hat[j]
                X_star_star[:, j] = X_j

        _, _, coefs, _ = lars_path(X_star_star, y.ravel(), return_n_iter=True, max_iter=d, method='lasso')
        # alphas, active, coefs = lars_path(X_star_star, y.ravel(), method='lasso')
        try:           
            beta_hat_star_star = coefs[:, d]
        except IndexError:
            beta_hat_star_star = coefs[:, -1]

        beta_hat_star_n_valcol = np.array([beta_hat_star_star[j]/w_hat[j] for j in range(len(beta_hat_star_star))])
        beta_hat_star_n = np.zeros(X.shape[1])
        beta_hat_star_n[valcols] = beta_hat_star_n_valcol
        return beta_hat_star_n.reshape(1, -1).squeeze()
    
    def fit(self, X, y, verbose=False):
        self.mu = y.mean() if self.fit_intercept else 0            
        beta = self.__call__(X=X, y=y-self.mu, d=self.default_d, verbose=verbose)
        self.beta = beta.reshape(-1, 1)

    def predict(self, X):
        return np.dot(X, self.beta) + self.mu
    
    def s_max(self, k, n, p, c1=1, c0=0):
        if self.gamma==0:
            return c1*(p/(k**2)) + c0
        else:
            return c1*min(np.power(p, 1/2)/k, np.power(p*n, 1/3)/k) + c0

class LARS:
    def __init__(self, default_d=None):
        self.default_d=default_d
    
    def __repr__(self):
        return 'Lars'

    def __str__(self):
        return 'Lars'

    def set_default_d(self, default_d):
        self.default_d = default_d

    def get_params(self, deep=False):
        return {'default_d': self.default_d}

    def __call__(self, X, y, d, verbose=False):
        self.lars = Lars(fit_intercept=False, fit_path=False, verbose=verbose, n_nonzero_coefs=d, copy_X=True)
        self.lars.fit(X, y)
        return self.lars.coef_

class ThresholdedLeastSquares:
    def __init__(self, default_d=None):
        self.default_d=default_d

    def __repr__(self):
        return 'TLS'

    def __str__(self):
        return 'TLS'

    def set_default_d(self, d):
        self.set_default_d=d
    
    def get_params(self, deep=False):
        return {
            'default_d': self.default_d
        }

    def __call__(self, X, y, d, verbose=False):
        if verbose: print('Full OLS')
        beta_ols, _, _, _ = np.linalg.lstsq(X, y)
        idx = np.argsort(beta_ols)[-d:]
        if verbose: print('Thresholded OLS')
        beta_tls, _, _, _ = np.linalg.lstsq(X[:, idx], y)
        beta = np.zeros_like(beta_ols)
        beta[idx] = beta_tls
        if verbose: print(idx, beta_tls)
        return beta

class SIS:
    def __init__(self, n_sis):
        self.n_sis = n_sis
    
    def get_params(self, deep=False):
        return {'n_sis': self.n_sis,
                }
    
    def __str__(self):
        return 'OSIS(n_sis={0})'.format(self.n_sis)
    
    def __repr__(self):
        return self.__str__()
    
    def __call__(self, X, pool, res, verbose=False):
        sigma_X = np.std(X, axis=0)
        sigma_Y = np.std(res)

        XY = X*res.reshape(-1, 1)
        E_XY = np.mean(XY, axis=0)
        E_X = np.mean(X, axis=0)
        E_Y = np.mean(res)
        cov = E_XY - E_X*E_Y
        sigma = sigma_X*sigma_Y
        pearsons = cov/sigma
        absolute_pearsons = np.abs(pearsons)
        absolute_pearsons[np.isnan(absolute_pearsons)] = 0 # setting all rows of constants to have 0 correlation
        absolute_pearsons[np.isinf(absolute_pearsons)] = 0 # setting all rows of constants to have 0 correlation
        absolute_pearsons[np.isneginf(absolute_pearsons)] = 0 # setting all rows of constants to have 0 correlation
        if verbose: print('Selecting top {0} features'.format(self.n_sis))
        idxs = np.argsort(absolute_pearsons)
        
        idxs = idxs[::-1]
        max_size = len(pool) + self.n_sis
        only_options = idxs[:min(max_size, len(idxs))]
        mask = list(map(lambda x: not(x in pool), only_options))
        only_relevant_options = only_options[mask]
        best_idxs = only_relevant_options[:min(self.n_sis, len(only_relevant_options))]

        best_corr = absolute_pearsons[best_idxs]

        return best_corr, best_idxs

class ICL:
    def __init__(self, s, so, k, fit_intercept=True, normalize=True, pool_reset=False, optimize_k=False, track_intermediates=False):
        self.s = s
        self.sis = SIS(n_sis=s)
        self.so = so
        self.k = k
        self.fit_intercept = fit_intercept
        self.normalize=normalize
        self.pool_reset = pool_reset
        self.optimize_k = optimize_k
        self.track_intermediates = track_intermediates

    def get_params(self, deep=False):
        return {'s': self.s,
                'so': self.so,
                'k': self.k,
                'fit_intercept': self.fit_intercept,
                'normalize': self.normalize,
                'pool_reset': self.pool_reset,
                'optimize_k': self.optimize_k,
                'track_intermediates': self.track_intermediates
                }

    def __str__(self):
        return 'ICL(n_sis={0}, SO={1}, k={2})'.format(self.s, str(self.so), self.k)

    def __repr__(self, prec=3):
        ret = []
        for i, name in enumerate(self.feature_names_sparse_):
            ret += [('+' if self.coef_[0, i] > 0 else '') + 
                    str(np.format_float_scientific(self.coef_[0, i], precision=prec, unique=False))
                      + ' (' + str(name) + ')' + '\n']
        ret += [('+' if self.intercept_>0 else '') + str(float(np.round(self.intercept_, prec)))]
        return ''.join(ret)
     
    def solve_norm_coef(self, X, y):
        n, p = X.shape
        a_x, a_y = (X.mean(axis=0), y.mean()) if self.fit_intercept else (np.zeros(p), 0.0)
        b_x, b_y = (X.std(axis=0), y.std()) if self.normalize else (np.ones(p), 1.0)

        self.a_x = a_x
        self.a_y = a_y
        self.b_x = b_x
        self.b_y = b_y

        return self
    
    def normalize_Xy(self, X, y):
        X = (X - self.a_x)/self.b_x
        y = (y - self.a_y)/self.b_y
        return X, y

    def coef(self):
        if self.normalize:
            self.coef_ = self.beta_.reshape(1, -1) * self.b_y / self.b_x[self.beta_idx_].reshape(1, -1)
            self.intercept_ = self.a_y - self.coef_.dot(self.a_x[self.beta_idx_])
        else:
            self.coef_ = self.beta_
            self.intercept_ = self.intercept_
            
    def filter_invalid_cols(self, X):
        nans = np.isnan(X).sum(axis=0) > 0
        infs = np.isinf(X).sum(axis=0) > 0
        ninfs = np.isneginf(X).sum(axis=0) > 0

        nanidx = np.where(nans==True)[0]
        infidx = np.where(infs==True)[0]
        ninfidx = np.where(ninfs==True)[0]

        bad_cols = np.hstack([nanidx, infidx, ninfidx])
        bad_cols = np.unique(bad_cols)

        return bad_cols

    def fitting(self, X, y, feature_names=None, verbose=False, track_pool=False, opt_k = None):
        self.feature_names_ = feature_names
        n,p = X.shape
        stopping = self.k if opt_k is None else opt_k
        if verbose: print('Stopping after {0} iterations'.format(stopping))

        pool_ = set()
        if track_pool: self.pool = []
        if self.optimize_k or self.track_intermediates: self.intermediates = np.empty(shape=(self.k, 5), dtype=object)

        res = y
        i = 0
        IC = np.infty
        while i < stopping:
            if self.fit_intercept: 
                self.intercept_ = np.mean(res).squeeze()
            else:
                self.intercept_ = 0

            if verbose: print('.', end='')

            p, sis_i = self.sis(X=X, res=res, pool=list(pool_), verbose=verbose)
            pool_old = deepcopy(pool_)
            pool_.update(sis_i)
            pool_lst = list(pool_)
            if track_pool: self.pool = pool_lst
            if str(self.so) == 'EffAdaLASSO(gamma=1)':
                beta_i = self.so(X=X, y=y, d=i+1, idx_old = list(pool_old), idx_new=sis_i, verbose=verbose)
            else:
                beta_i = self.so(X=X[:, pool_lst], y=y, d=i+1, verbose=verbose)

            beta = np.zeros(shape=(X.shape[1]))
            beta[pool_lst] = beta_i

            if self.optimize_k or self.track_intermediates:
                idx = np.nonzero(beta)[0]
                if self.normalize:
                    coef = (beta[idx].reshape(1, -1)*self.b_y/self.b_x[idx].reshape(1, -1))
                    if self.fit_intercept:
                        intercept_ = self.a_y - coef.dot(self.a_x[idx])
                else:
                    coef = beta[idx]
                    if self.fit_intercept:
                        intercept_ = self.intercept_
                if len(coef.shape) > 1:
                    coef = coef[0]
                expr = ''.join([('+' if float(c) >= 0 else '') + str(np.round(float(c), 3)) + str(self.feature_names_[idx][q]) for q, c in enumerate(coef)])
                if verbose: print('Model after {0} iterations: {1}'.format(i, expr))

                self.intermediates[i, 0] = deepcopy(idx)
                self.intermediates[i, 1] = coef # deepcopy(beta[idx])
                self.intermediates[i, 2] = intercept_ if self.fit_intercept else 0
                self.intermediates[i, 3] = self.feature_names_[idx]
                self.intermediates[i, 4] = expr

            if self.pool_reset:
                idx = np.abs(beta_i) > 0 
                beta_i = beta_i[idx] 
                pool_lst = np.array(pool_lst)[idx]
                pool_lst = pool_lst.ravel().tolist()
                pool_ = set(pool_lst)

            res = (y.reshape(1, -1) - (np.dot(X, beta).reshape(1, -1)+self.intercept_) ).T

            i += 1
        if self.optimize_k or self.track_intermediates: self.intermediates = self.intermediates[:, :i]
            
        if verbose: print()
        
        self.beta_ = beta
        self.intercept_ = np.mean(res).squeeze() if self.fit_intercept else 0

        self.beta_idx_ = list(np.nonzero(self.beta_)[0])
        self.beta_sparse_ = self.beta_[self.beta_idx_]
        self.feature_names_sparse_ = np.array(self.feature_names_)[self.beta_idx_]

        return self

    def fit(self, X, y, val_size=0.1, feature_names=None, timer=False, verbose=False, track_pool=False, random_state=None):
        if verbose: print('removing invalid features')
        self.bad_col = self.filter_invalid_cols(X)
        X_ = np.delete(X, self.bad_col, axis=1)
        have_valid_names = not(feature_names is None) and X.shape[1] == len(feature_names)
        feature_names_ = np.delete(np.array(feature_names), self.bad_col) if have_valid_names else ['X_{0}'.format(i) for i in range(X_.shape[1])]
      
        if verbose: print('Feature normalisation')
        self.solve_norm_coef(X_, y)
        X_, y_ = self.normalize_Xy(X_, y)

        if verbose: print('Fitting ICL model')
        if timer: start=time()
        if self.optimize_k == False:
            self.fitting(X=X_, y=y_, feature_names=feature_names_, verbose=verbose, track_pool = track_pool)
        else:
            if verbose: print('Finding optimal model size')
            X_train, X_val, y_train, y_val = train_test_split(X_, y_, test_size=val_size, random_state=random_state)
            self.fitting(X=X_train, y=y_train, feature_names=feature_names_, verbose=verbose, track_pool = track_pool)
            best_k, best_e2 = 0, np.infty
            for k in range(self.k):
                idx = self.intermediates[k, 0]
                coef = self.intermediates[k, 1]
                inter = self.intermediates[k, 2]
                X_pred = np.delete(X_val, self.bad_col, axis=1)
                y_hat = (np.dot(X_pred[:, idx], coef.squeeze()) + inter).reshape(-1, 1)
                e2_val = rmse(y_hat, y_val)
                if e2_val < best_e2:
                    best_k, best_e2 = k+1, e2_val
            if verbose: print('refitting with k={0}'.format(best_k))
            self.fitting(X=X_, y=y_, feature_names=feature_names_, verbose=verbose, track_pool = track_pool, opt_k = best_k)

        if timer: self.fit_time=time()-start
        if timer and verbose: print(self.fit_time)

        self.beta_so_ = self.beta_sparse_
        self.feature_names = self.feature_names_sparse_

        self.beta_, _, _, _ = np.linalg.lstsq(a=X_[:, self.beta_idx_], b=y_)
        
        if verbose: print('Inverse Transform of Feature Space')
        self.coef()

        if verbose: print('Fitting complete')

        return self
    
    def predict(self, X):
        X_ = np.delete(X, self.bad_col, axis=1)
        return (np.dot(X_[:, self.beta_idx_], self.coef_.squeeze()) + self.intercept_).reshape(-1, 1)

    def predict_ensemble(self, X):
        y_hat = np.zeros(shape=(X.shape[0], self.k))
        for k in range(self.k):
            idx = self.intermediates[k, 0]
            coef = self.intermediates[k, 1]
            inter = self.intermediates[k, 2]
            X_pred = np.delete(X, self.bad_col, axis=1)
            y_hat[:, k]=(np.dot(X_pred[:, idx], coef) + inter).reshape(-1, 1).squeeze()
        return y_hat
    
    def repr_ensemble(self, prec=3):
        ret = []
        for k in range(self.k):
            idx = self.intermediates[k, 0]
            coef = self.intermediates[k, 1]
            inter = self.intermediates[k, 2]
            feat = self.intermediates[k, 3]
            model_k = []
            for i, name in enumerate(feat):
                model_k += [('+' if coef[i] > 0 else '') + 
                        str(np.format_float_scientific(coef[i], precision=prec, unique=False))
                        + ' (' + str(name) + ')' + '\n']
            model_k += [('+' if inter > 0 else '')  + str(float(np.round(inter, prec)))]
            model_k = ''.join(model_k)
            ret += [model_k]
        return ';\n\n'.join(ret)

    def score(self, X, y, scorer=rmse):
        return scorer(self.predict(X), y)

    def score_ensemble(self, X, y):
        y_hat_ens = self.predict_ensemble(X)
        return np.mean((y_hat_ens - y.reshape(-1,1))**2, axis=0)

class BOOTSTRAP:
    def __init__(self, X, y=None, random_state=None):
        self.X = X
        self.y = y
        self.random_state = random_state
        np.random.seed(random_state)

    def sample(self, n, ret_idx=False):
        in_idx = np.random.randint(low=0, high=self.X.shape[0], size=n)
        out_idx = list(set(range(self.X.shape[0])) - set(in_idx))
        if ret_idx:
            return in_idx, out_idx
        else:
            return self.X[in_idx], self.X[out_idx], self.y[in_idx], self.y[out_idx]

class ICL_ensemble:
    def __init__(self, n_estimators, s, so, d, fit_intercept=True, normalize=True, pool_reset=False, information_criteria=None, random_state = None): #, track_intermediates=False):
        self.n_estimators = n_estimators
        self.s = s
        self.sis = SIS(n_sis=s)
        self.so = so
        self.d = d
        self.fit_intercept = fit_intercept
        self.normalize=normalize
        self.pool_reset = pool_reset
        self.information_criteria = information_criteria if information_criteria in IC_DICT.keys() else None
        self.random_state = random_state
        self.base = ICL(s=s, so=so, d=d,
                         fit_intercept=fit_intercept, normalize=normalize,
                           pool_reset=pool_reset, information_criteria=information_criteria)
    
    def get_params(self, deep=False):
        return {
                'n_estimators': self.n_estimators,
                's': self.s,
                'so': self.so,
                'd': self.d,
                'fit_intercept': self.fit_intercept,
                'normalize': self.normalize,
                'pool_reset': self.pool_reset,
                'information_criteria': self.information_criteria,
                'random_state': self.random_state
        }
    
    def __str__(self):
        return 'ICL(s={0}, so={1}, d={2}, fit_intercept={3}, normalize={4}, pool_reset={5}, information_criteria={6}, random_state={7})'.format(self.s, self.so, self.d, self.fit_intercept, self.normalize, self.pool_reset, self.information_criteria, self.random_state)

    def __repr__(self):
        return '\n'.join([self.ensemble_[i].__repr__() for i in range(self.n_estimators)])
               
    def fit(self, X, y, feature_names=None, verbose=False):
        sampler = BOOTSTRAP(X=X, y=y, random_state=self.random_state)
        self.ensemble_ = np.empty(shape=self.n_estimators, dtype=object)
        for i in range(self.n_estimators):
            if verbose: print('fitting model {0}'.format(i+1))
            X_train, X_test, y_train, y_test = sampler.sample(n=len(X))
            self.ensemble_[i] = clone(self.base)
            self.ensemble_[i].fit(X=X_train, y=y_train, feature_names=feature_names, verbose=verbose)

    def get_rvs(self, X):
        rvs = np.empty(shape=(X.shape[0], self.n_estimators))
        for i in range(self.n_estimators):
            rvs[:, i] = self.ensemble_[i].predict(X).squeeze()
        return rvs
    
    def mean(self, X):
        return self.get_rvs(X=X).mean(axis=1)

    def std(self, X):
        return self.get_rvs(X=X).std(axis=1)

    def predict(self, X, std=False):
        rvs = self.get_rvs(X=X)
        if std:
            return rvs.mean(axis=1), rvs.std(axis=1)
        else:
            return rvs.mean(axis=1)

class FeatureExpansion:
    def __init__(self, ops, rung, printrate=1000):
        self.ops = ops
        self.rung = rung
        self.printrate = printrate
        self.prev_print = 0
        for i, op in enumerate(self.ops):
            if type(op) == str:
                self.ops[i] = (op, range(rung))
        
    def remove_redundant_features(self, symbols, names, X):
        sorted_idxs = np.argsort(names)
        for i, idx in enumerate(sorted_idxs):
            if i == 0:
                unique = [idx]
            elif names[idx] != names[sorted_idxs[i-1]]:
                unique += [idx]
        unique_original_order = np.sort(unique)
        
        return symbols[unique_original_order], names[unique_original_order], X[:, unique_original_order]
    
    def expand(self, X, names=None, verbose=False, f=None, check_pos=False):
        n, p = X.shape
        if (names is None) or (len(names) != p):
            names = ['x_{0}'.format(i) for i in range(X.shape[1])]
        
        if check_pos == False:
            symbols = sp.symbols(' '.join(name.replace(' ', '.') for name in names))
        else:
            symbols = []
            for i, name in enumerate(names):
                name = name.replace(' ', '.')
                if np.all(X[:, i] > 0):
                    sym = sp.symbols(name, real=True, positive=True)
                else:
                    sym = sp.symbols(name, real=True)               
                symbols.append(sym)

        symbols = np.array(symbols)
        names = np.array(names)
        
        if verbose: print('Estimating the creation of around {0} features'.format(self.estimate_workload(p=p, max_rung=self.rung, verbose=verbose>2)))
        
        names, symbols, X = self.expand_aux(X=X, names=names, symbols=symbols, crung=0, prev_p=0, verbose=verbose)

        if not(f is None):
            import pandas as pd
            df = pd.DataFrame(data=X, columns=names)
            df['y'] = y
            df.to_csv(f)

        return names, symbols, X
        
    def estimate_workload(self, p, max_rung,verbose=False):
        p0 = 0
        p1 = p
        for rung in range(max_rung):
            if verbose: print('Applying rung {0} expansion'.format(rung))
            new_u, new_bc, new_bn = 0, 0, 0
            for (op, rung_range) in self.ops:
                if rung in rung_range:
                    if verbose: print('Applying {0} to {1} features will result in approximately '.format(op, p1-p0))
                    if OP_DICT[op]['inputs'] == 1:
                        new_u += p1
                        if verbose: print('{0} new features'.format(p1))
                    elif OP_DICT[op]['commutative'] == True:
                        new_bc += (1/2)*(p1 - p0 + 1)*(p0 + p1 + 2)
                        if verbose: print('{0} new features'.format((1/2)*(p1 - p0 + 1)*(p0 + p1 + 2)))
                    else:
                        new_bn += (p1 - p0 + 1)*(p0 + p1 + 2)
                        if verbose: print('{0} new features'.format((p1 - p0 + 1)*(p0 + p1 + 2)))
            p0 = p1
            p1 = p1 + new_u + new_bc + new_bn
            if verbose: print('For a total of {0} features by rung {1}'.format(p1, rung))
        return p1
        
    def add_new(self, new_names, new_symbols, new_X, new_name, new_symbol, new_X_i, verbose=False):
        valid = (np.isnan(new_X_i).sum(axis=0) + np.isposinf(new_X_i).sum(axis=0) + np.isneginf(new_X_i).sum(axis=0)) == 0
        if new_names is None:
            new_names = np.array(new_name[valid])
            new_symbols = np.array(new_symbol[valid])
            new_X = np.array(new_X_i[:, valid])
        else:
            new_names = np.concatenate((new_names, new_name[valid]))
            new_symbols = np.concatenate((new_symbols, new_symbol[valid]))
            new_X = np.hstack([new_X, new_X_i[:, valid]])
#        if (verbose > 1) and not(new_names is None) and (len(new_names) % self.printrate == 0): print('Created {0} features so far'.format(len(new_names)))
        if (verbose > 1) and not(new_names is None) and (len(new_names) - self.prev_print >= self.printrate):
            self.prev_print = len(new_names)
            elapsed = np.round(time() - self.start_time, 2)
            print('Created {0} features so far in {1} seconds'.format(len(new_names),elapsed))
        return new_names, new_symbols, new_X

    def expand_aux(self, X, names, symbols, crung, prev_p, verbose=False):
        
        str_vectorize = np.vectorize(str)

        def simplify_nested_powers(expr):
            # Replace (x**n)**(1/n) with x
            def flatten_pow_chain(e):
                if isinstance(e, sp.Pow) and isinstance(e.base, sp.Pow):
                    base, inner_exp = e.base.args
                    outer_exp = e.exp
                    combined_exp = inner_exp * outer_exp
                    if sp.simplify(combined_exp) == 1:
                        return base
                    return sp.Pow(base, combined_exp)
                elif isinstance(e, sp.Pow) and sp.simplify(e.exp) == 1:
                    return e.base
                return e
            # Apply recursively
            return expr.replace(
                lambda e: isinstance(e, sp.Pow),
                flatten_pow_chain
            )
        
        if crung == 0:
            self.start_time = time()
            symbols, names, X = self.remove_redundant_features(X=X, names=names, symbols=symbols)
        if crung==self.rung:
            if verbose: print('Completed {0} rounds of feature transformations'.format(self.rung))
            return symbols, names, X
        else:
            if verbose: print('Applying round {0} of feature transformations'.format(crung+1))
#            if verbose: print('Estimating the creation of {0} features this iteration'.format(self.estimate_workload(p=X.shape[1], max_rung=1)))
                
            new_names, new_symbols, new_X = None, None, None
            
            for (op_key, rung_range) in self.ops:
                if crung in rung_range:
                    if verbose>1: print('Applying operator {0} to {1} features'.format(op_key, X.shape[1]))
                    op_params = OP_DICT[op_key]
                    op_sym, op_np, inputs, comm = op_params['op'], op_params['op_np'], op_params['inputs'], op_params['commutative']
                    if inputs == 1:
                        sym_vect = np.vectorize(op_sym)
                        new_op_symbols = sym_vect(symbols[prev_p:])
                        new_op_X = op_np(X[:, prev_p:])
                        new_op_names = str_vectorize(new_op_symbols)
                        new_names, new_symbols, new_X = self.add_new(new_names=new_names, new_symbols=new_symbols, new_X=new_X, 
                                                                    new_name=new_op_names, new_symbol=new_op_symbols, new_X_i=new_op_X, verbose=verbose)
                    elif inputs == 2:
                        for idx1 in range(prev_p, X.shape[1]):
                            sym_vect = np.vectorize(lambda idx2: op_sym(symbols[idx1], symbols[idx2]))
                            idx2 = range(idx1 if comm else X.shape[1])
                            if len(idx2) > 0:
                                new_op_symbols = sym_vect(idx2)
                                new_op_names = str_vectorize(new_op_symbols)
                                X_i = X[:, idx1]
                                new_op_X = op_np(X_i[:, np.newaxis], X[:, idx2]) #X_i[:, np.newaxis]*X[:, idx2]                                                
                                new_names, new_symbols, new_X = self.add_new(new_names=new_names, new_symbols=new_symbols, new_X=new_X, 
                                                                        new_name=new_op_names, new_symbol=new_op_symbols, new_X_i=new_op_X, verbose=verbose)
            if not(new_names is None):                
                names = np.concatenate((names, new_names))
                symbols = np.concatenate((symbols, new_symbols))
                prev_p = X.shape[1]
                X = np.hstack([X, new_X])
            else:
                prev_p = X.shape[1]
                
            if verbose: print('After applying rounds {0} of feature transformations there are {1} features'.format(crung+1, X.shape[1]))
            if verbose: print('Removing redundant features leaves... ', end='')            
            symbols, names, X = self.remove_redundant_features(X=X, names=names, symbols=symbols)
            if verbose: print('{0} features'.format(X.shape[1]))

            return self.expand_aux(X=X, names=names, symbols=symbols, crung=crung+1, prev_p=prev_p, verbose=verbose)
    
class LOGISTIC_ICL:
    def __init__(self, s, so, k, pool_reset=False, track_intermediates=False, max_iter=100, tol=1e-6, eps=1e-12, damping=0.5, prec=3):
        self.s = s
        self.so = so
        self.k = k
        self.pool_reset=pool_reset
        self.track_intermediates = track_intermediates
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.eps = float(eps)
        self.damping = float(damping)
        self.prec = prec

        self.icl = ICL(s=s, so=so, k=k, normalize=False, fit_intercept=False, optimize_k=True, track_intermediates=self.track_intermediates)
        self.coef_ = None          # (p,)
        self.intercept_ = 0.0      # scalar

    def get_params(self, deep=False):
        params = {
            "s": self.s,
            "so": self.so,
            "k": self.k,
            "pool_reset": self.pool_reset,
            "track_intermediates": self.track_intermediates,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "eps": self.eps,
            "damping": self.damping,
            "prec": self.prec
        }
        if deep:
            # expose inner ICL params using sklearn convention
            for key, value in self.icl.get_params(deep=True).items():
                params[f"icl__{key}"] = value

        return params 

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        if self.coef_ is None:
            raise ValueError("Model is not fitted yet.")
        return X @ self.coef_ + self.intercept_

    def predict_proba(self, X):
        eta = self.decision_function(X)
        p1 = self._sigmoid(eta)
        p1 = np.clip(p1, self.eps, 1.0 - self.eps)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)
    
    def __repr__(self):
        return '\n'.join([('+' if beta>=0 else '') +
                 sci(beta, self.prec) + 
                 '('+
                 str(self.feature_names[i]) +
                 ')' 
                 for i, beta in enumerate(self.coef_) if abs(beta)>0]) + '\n' + ('+' if self.intercept_ >= 0 else '') + sci(self.intercept_, self.prec)        

    def __str__(self):
        return 'LOGISTIC_ICL({0})'.format(str(self.get_params()))
    
    def fit(self, X, y, feature_names=None, verbose=False):
        n, p = X.shape

        beta = np.zeros(p, dtype=float)
        b = 0.0

        for it in range(self.max_iter):
            if verbose: print('iteration {0}'.format(it))
            # predicting with current model
            eta = np.dot(X, beta) + b

            # converting to probabilties 
            p_hat = self._sigmoid(eta)
            p_hat = np.clip(p_hat, self.eps, 1.0 - self.eps)
            
            # row weights
            w = p_hat * (1.0 - p_hat)
            w = np.clip(w, self.eps, 1.0)

            # reweighted x and y
            z = eta + (y - p_hat) / w
            
            w_sum = w.sum()
            xbar = (w[:, None] * X).sum(axis=0) / w_sum
            zbar = (w * z).sum() / w_sum

            Xc = X - xbar
            zc = z - zbar

            s = np.sqrt(w)
            X_star = Xc * s[:, None]
            z_star = zc * s
            
            # fitting icl model to reweighted data
            icl_iter = clone(self.icl)
            icl_iter.fit(X_star, z_star, feature_names=feature_names, verbose=verbose>1)

            beta_new = np.zeros(p, dtype=float)
            beta_new[icl_iter.beta_idx_] = icl_iter.beta_

            b_new = zbar - xbar @ beta_new

            # updating previous solution as linear combination of past and current
            beta_next = beta + self.damping * (beta_new - beta)
            b_next = b + self.damping * (b_new - b)

            # stopping criteria
            delta = np.linalg.norm(beta_next - beta, 2) + abs(b_next - b)
            if verbose:
                print(f"iter {it+1}: delta={delta:.3e}")

            beta, b = beta_next, b_next

            if delta <= self.tol:
                if verbose: print('converged')
                break
        
        if (verbose) and (it == self.max_iter): print('did not converge')

        tol = 1e-10

        # number of nonzeros in the fitted model
        nz_idx = np.flatnonzero(np.abs(beta) > tol)
        nz = int(np.sum(np.abs(beta) > tol))

        std = X.std(axis=0, ddof=0)
        scores = np.abs(beta) * std

        kk = min(self.k, nz)

        # choose among nonzero coefficients only (avoids picking zero-weight features)
        if kk == 0:
            # degenerate: no features selected
            self.coef_ = np.zeros_like(beta)
            self.idx = np.array([], dtype=int)
            self.intercept_ = float(b)
        else:
            order = nz_idx[np.argsort(scores[nz_idx])[-kk:][::-1]]
            XS = X[:, order]

            clf = LogisticRegressionCV(
                Cs=20,                 # or a list/array like np.logspace(-4, 4, 30)
                cv=5,
                penalty="l2",
                solver="lbfgs",
                scoring="neg_log_loss",
                fit_intercept=True,
                max_iter=5000,
                n_jobs=-1,
                refit=True,
            )
            clf.fit(XS, y)

            self.coef_ = np.zeros_like(beta)
            self.coef_[order] = clf.coef_.ravel()
            self.idx = order
            self.intercept_ = float(clf.intercept_[0])

        self.feature_names_sparse_ = feature_names[self.idx]
        self.feature_names = feature_names
        return self
    
    @staticmethod
    def _sigmoid(eta):
        # stable sigmoid
        eta = np.asarray(eta, dtype=float)
        out = np.empty_like(eta, dtype=float)
        pos = eta >= 0
        neg = ~pos
        out[pos] = 1.0 / (1.0 + np.exp(-eta[pos]))
        exp_eta = np.exp(eta[neg])
        out[neg] = exp_eta / (1.0 + exp_eta)
        return out

def zero_one_loss(X, y, model):
    y = np.asarray(y)
    y_hat = model.predict(X)
    return np.mean(y_hat != y)

def hinge_loss(X, y, model):
    y = np.asarray(y)
    y_pm = 2*y - 1          # {0,1} -> {-1,+1}
    eta = model.decision_function(X)
    return np.mean(np.maximum(0.0, 1.0 - y_pm * eta))
    
def log_loss(X, y, model):
    y = np.asarray(y)
    eta = model.decision_function(X)
    return np.mean(np.logaddexp(0.0, eta) - y*eta)


sci = lambda x, sig=3: f"{float(x):.{sig}e}"

if __name__ == "__main__":   
    test = "bandgap" 
    random_state = 0
    np.random.seed(random_state)
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score as r2

    import pandas as pd
    import os

    if test == "DIABETES":
        df = pd.read_csv(os.path.join(os.getcwd(), "Input", "pima.csv"))
        df["DIABETES"] = df["DIABETES"].map({"Y":1, "N": 0})
        y = df['DIABETES'].values
        X = df.drop(columns=['DIABETES'])
        feature_names = X.columns
        X = X.values

        rung = 2
        small = ['sin', 'cos', 'log', 'abs', 'sqrt', 'cbrt', 'sq', 'cb', 'inv']
        big = ['six_pow', 'exp', 'add', 'mul', 'div', 'abs_diff']
        small  = [(op, range(rung)) for op in small]
        big = [(op, range(1)) for op in big]
        ops = small+big

        FE = FeatureExpansion(rung=rung, ops=ops)
        Phi_names, Phi_symbols, Phi_ = FE.expand(X=X, names=feature_names, check_pos=True, verbose=True)
        X_train, X_test, y_train, y_test = train_test_split(Phi_, y, test_size=0.2, random_state=random_state)

        logistic_icl_params = {
            "s": 10,
            "so": AdaptiveLASSO(gamma=1, fit_intercept=False),
            "k": 5,
            "pool_reset": False,
            "track_intermediates": False,
            "max_iter": 1000,
            "tol": 1e-1,
            "eps": 1e-3,
            "damping": 0.5,
            "prec": 3
        }

        icl_log = LOGISTIC_ICL(**logistic_icl_params)
        icl_log.fit(X=X_train, y=y_train, feature_names=Phi_names, verbose=1)

        print(icl_log.__repr__())
        print('zero_one: {0}'.format(zero_one_loss(X_test, y_test, icl_log)))
        print('hinge: {0}'.format(hinge_loss(X_test, y_test, icl_log)))
        print('logloss: {0}'.format(log_loss(X_test, y_test, icl_log))) 
    elif test=="Synthetic":
        k,n,p=3,10000,1000
        rng = np.random.default_rng(random_state)
        X = rng.standard_normal((n,p))
        feature_names = np.array(['X_{0}'.format(i) for i in range(p)])
        support = range(k)
        beta = np.zeros(p, dtype=float)
        signs = rng.choice([-1.0, 1.0], size=k)
        mags = rng.uniform(0.5, 1.5, size=k)
        beta[support] = signs * mags
        eta_no_b = X @ beta
        b = float(-np.mean(eta_no_b))
        eta = eta_no_b + b
        p1 = 1.0 / (1.0 + np.exp(-np.clip(eta, -50, 50)))
        y = rng.binomial(1, p1, size=n).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

        logistic_icl_params = {
            "s": 10,
            "so": AdaptiveLASSO(gamma=1, fit_intercept=False),
            "k": k,
            "pool_reset": False,
            "track_intermediates": False,
            "max_iter": 1000,
            "tol": 1e-3,
            "eps": 1e-6,
            "damping": 0.8,
            "prec": 3
        }

        icl_log = LOGISTIC_ICL(**logistic_icl_params)
        icl_log.fit(X=X_train, y=y_train, feature_names=feature_names, verbose=1)

        print(icl_log.__repr__())
        print('zero_one: {0}'.format(zero_one_loss(X_test, y_test, icl_log)))
        print('hinge: {0}'.format(hinge_loss(X_test, y_test, icl_log)))
        print('logloss: {0}'.format(log_loss(X_test, y_test, icl_log)))
    
        print('True Coef: {0}'.format(beta[:k]))
        print('True intercept: {0}'.format(b))
        eta_test = icl_log.decision_function(X_test)    # log-odds
        p_test = 1.0 / (1.0 + np.exp(-eta_test))
        print('Bayes error: {0}'.format(np.mean(np.minimum(p_test, 1-p_test))))
    elif test=='bandgap':
        path = os.path.join('/'.join(os.getcwd().split('/')[:-1]), 'icol_exp', 'Input', 'data_HTE.csv')
        df = pd.read_csv(path)
        y = df['Y_oxygenate'].values
        X = df.drop(columns=['material_and_condition', 'Y_oxygenate'])
        feature_names = X.columns
        X = X.values

        rung = 2
        small = ['sin', 'cos', 'log', 'abs', 'sqrt', 'cbrt', 'sq', 'cb', 'inv']
        big = ['six_pow', 'exp', 'add', 'mul', 'div', 'abs_diff']
        small  = [(op, range(rung)) for op in small]
        big = [(op, range(1)) for op in big]
        ops = small+big

        FE = FeatureExpansion(rung=rung, ops=ops)
        Phi_names, Phi_symbols, Phi_ = FE.expand(X=X, names=feature_names, check_pos=True, verbose=True)

        X_train, X_test, y_train, y_test = train_test_split(Phi_, y, test_size=0.2, random_state=random_state)
        for i, s in enumerate([1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,200,300,400]):
            icl = ICL(s=s, so=AdaptiveLASSO(gamma=1, fit_intercept=False), k=5, fit_intercept=True,
                    normalize=True, optimize_k=True, track_intermediates=False)

            icl.fit(X_train, y_train, feature_names=Phi_names, verbose=0)
            y_test_hat = icl.predict(X_test)
            score = r2(y_test, y_test_hat)
            print('model={0}, s={2}, r2={1}'.format(icl.__repr__(), score, s))
