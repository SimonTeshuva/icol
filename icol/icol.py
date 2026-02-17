import warnings
warnings.filterwarnings('ignore')

from time import time
from copy import deepcopy
from itertools import combinations, permutations

import numpy as np
import sympy as sp

from sklearn.linear_model import lars_path, Ridge, Lars
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

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
            X_star_star = X_valcols / w_hat

            # X_star_star = np.zeros_like(X_valcols[:, idx_ala])
            # for j in range(X_star_star.shape[1]): # vectorise
            #     X_j = X_valcols[:, j]/w_hat[j]
            #     X_star_star[:, j] = X_j

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
            X_star_star = X_valcols / w_hat
            # X_star_star = np.zeros_like(X_valcols)
            # for j in range(X_star_star.shape[1]): # vectorise
            #     X_j = X_valcols[:, j]/w_hat[j]
            #     X_star_star[:, j] = X_j

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
        ret += [('+' if self.intercept_>0 else '') +  str(np.format_float_scientific(self.intercept_, precision=prec, unique=False))]
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

sci = lambda x, sig=3: f"{float(x):.{sig}e}"

if __name__ == "__main__":   
    pass