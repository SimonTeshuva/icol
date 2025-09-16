import warnings
warnings.filterwarnings('ignore')

from time import time
from copy import deepcopy
from itertools import combinations, permutations

import numpy as np
import sympy as sp

from sklearn.linear_model import lars_path, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)

def LL(res):
    n = len(res)
    return n*np.log(np.sum(res**2)/n)

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
    def __init__(self, s, so, k, fit_intercept=True, normalize=True, pool_reset=False, optimize_k=False):
        self.s = s
        self.sis = SIS(n_sis=s)
        self.so = so
        self.k = k
        self.fit_intercept = fit_intercept
        self.normalize=normalize
        self.pool_reset = pool_reset
        self.optimize_k = optimize_k
    
    def get_params(self, deep=False):
        return {'s': self.s,
                'so': self.so,
                'k': self.k,
                'fit_intercept': self.fit_intercept,
                'normalize': self.normalize,
                'pool_reset': self.pool_reset,
                'self.optimize_k': self.optimize_k
                }

    def __str__(self):
        return 'ICL(n_sis={0}, SO={1}, k={2})'.format(self.s, str(self.so), self.k)

    def __repr__(self, prec=3):
        ret = []
        for i, name in enumerate(self.feature_names_sparse_):
            ret += [('+' if self.coef_[0, i] > 0 else '') + 
                    str(np.format_float_scientific(self.coef_[0, i], precision=prec, unique=False))
                      + ' (' + str(name) + ')' + '\n']
        ret += ['+' + str(float(np.round(self.intercept_, prec)))]
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
        if self.optimize_k: self.intermediates = np.empty(shape=(self.k, 5), dtype=object)

        res = y
        i = 0
        IC = np.infty
        while i < stopping:
            self.intercept_ = np.mean(res).squeeze()
            if verbose: print('.', end='')

            p, sis_i = self.sis(X=X, res=res, pool=list(pool_), verbose=verbose)
            pool_.update(sis_i)
            pool_lst = list(pool_)
            
            if track_pool: self.pool = pool_lst
            beta_i = self.so(X=X[:, pool_lst], y=y, d=i+1, verbose=verbose)

            beta = np.zeros(shape=(X.shape[1]))
            beta[pool_lst] = beta_i

            if self.optimize_k:
                idx = np.nonzero(beta)[0]
                if self.normalize:
                    coef = (beta[idx].reshape(1, -1)*self.b_y/self.b_x[idx].reshape(1, -1))
                    intercept_ = self.a_y - coef.dot(self.a_x[idx])
                else:
                    coef = beta[idx]
                    intercept_ = self.intercept_
                coef = coef[0]
                expr = ''.join([('+' if float(c) >= 0 else '') + str(np.round(float(c), 3)) + self.feature_names_[idx][q] for q, c in enumerate(coef)])
                if verbose: print('Model after {0} iterations: {1}'.format(i, expr))

                self.intermediates[i, 0] = deepcopy(idx)
                self.intermediates[i, 1] = coef # deepcopy(beta[idx])
                self.intermediates[i, 2] = intercept_
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
        if self.optimize_k: self.intermediates = self.intermediates[:, :i]
            
        if verbose: print()
        
        self.beta_ = beta
        self.intercept_ = np.mean(res).squeeze()

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

    def score(self, X, y, scorer=rmse):
        return scorer(self.predict(X), y)

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
    
    def expand(self, X, names=None, verbose=False, f=None):
        n, p = X.shape
        if (names is None) or (len(names) != p):
            names = ['x_{0}'.format(i) for i in range(X.shape[1])]
        symbols = np.array(sp.symbols(' '.join(name for name in names)))
        names = np.array(names)
        
        if verbose: print('Estimating the creation of around {0} features'.format(self.estimate_workload(p=p, max_rung=self.rung, verbose=verbose>2)))
        
        names, symbols, X = self.expand_aux(X=X, names=names, symbols=symbols, crung=0, prev_p=0, verbose=verbose)
        if not(f is None):
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
                                new_op_X = X_i[:, np.newaxis]*X[:, idx2]                                                
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
        
if __name__ == "__main__":
    # n = 1000
    # X = np.eye(n)
    # X[-1, -1] = 1e-300  # Tiny singular value
    # y=np.ones(n)

    # ala = AdaptiveLASSO(gamma=1, fit_intercept=False)
    # coef = ala(X, y, d=X.shape[1], verbose=True)
    # print(coef)
    # # print(X @ coef)

    # #######
    # testing feature expansion here

    import os
    import pandas as pd

    root = '/'.join(os.getcwd().split('/')[:-1])
    	
    f = os.path.join(root, 'ExperimentCode', 'Input', 'data_bandgap.csv')
    df = pd.read_csv(f)
    target = 'bg_hse06 (eV)'
    drop = ['material']
    y = df[target].values
    X = df.drop(columns=drop+[target])   
    feature_names = X.columns
    X = X.values

    rung = 2
    size=0.05


    unary = ['sin', 'cos', 'log', 'exp', 'sqrt', 'cbrt', 'sq', 'cb', 'six_pow', 'inv']
    binary = ['mul', 'div', 'add', 'sub', 'abs_diff']
    binary = ['mul', 'div', 'add', 'sub', 'abs_diff']
    unary = [(op, range(rung)) for op in unary]
    binary = [(op, range(1)) for op in binary]
    ops = unary + binary    

    fe = FeatureExpansion(rung=rung, ops=ops)
    spnames, names, X_ = fe.expand(X=X, names=feature_names, verbose=True)

    # n,p_ = X_.shape
    n,p = X.shape
    sampler = BOOTSTRAP(X=X, y=y, random_state=0)
    idx, out = sampler.sample(int(size*n), ret_idx=True)

    X_train, y_train = X_[idx], y[idx]
    X_test, y_test = X_[out], y[out]

    s_max = 3000
    s = 1

    while s < s_max:
        icol = ICL(s=s, so=AdaptiveLASSO(gamma=1, fit_intercept=False), k=5, optimize_k=False)
        icol.fit(X=X_train, y=y_train, feature_names=names, verbose=False, val_size=0.2)
        y_hat_test = icol.predict(X_test)
        print('Not optimizing k')
        print(icol.__str__())
        print(icol.__repr__())
        print(rmse(y_test, y_hat_test))


        icol = ICL(s=s, so=AdaptiveLASSO(gamma=1, fit_intercept=False), k=5, optimize_k=True)
        icol.fit(X=X_train, y=y_train, feature_names=names, verbose=False, val_size=0.2)
        y_hat_test = icol.predict(X_test)
        print('with optimizing k')
        print(icol.__str__())
        print(icol.__repr__())
        print(rmse(y_test, y_hat_test))

        s *= 2

        input()

    # idx1 = np.array([8, 3081, 3084, 530, 1049, 1052, 1054, 1059, 3108, 3110, 2088, 555, 1068, 558, 559, 1073, 3121, 568, 2108, 1092, 75, 78, 2647, 1113, 1114, 3674, 604, 3675, 3166, 2658, 1122, 3685, 1638, 616, 2153, 3694, 3182, 113, 1142, 634, 126, 644, 1156, 1159, 1161, 1163, 3212, 1165, 2701, 3213, 3214, 1169, 659, 662, 663, 3224, 665, 1179, 156, 1691, 1184, 3233, 675, 3753, 681, 3246, 3763, 3256, 2750, 191, 1215, 2753, 1218, 1220, 3781, 198, 3782, 3784, 1225, 3785, 721, 1234, 3794, 724, 725, 3282, 1239, 3804, 3292, 734, 3302, 2793, 1258, 1259, 234, 1265, 247, 3319, 3320, 3321, 3323, 3324, 3839, 3840, 770, 1283, 1285, 1286, 3333, 2823, 3850, 3343, 3859, 3347, 283, 800, 2857, 810, 2860, 1325, 1328, 3377, 1330, 3378, 3379, 1335, 825, 828, 829, 3389, 831, 1344, 1856, 2369, 1349, 3398, 841, 1353, 3918, 846, 1361, 338, 3411, 344, 3928, 3421, 354, 2915, 2918, 3946, 3947, 3949, 3950, 886, 3959, 3447, 889, 890, 3969, 3457, 899, 3467, 2958, 399, 922, 412, 3484, 3485, 3486, 3488, 3489, 4004, 933, 4005, 3498, 2988, 4015, 435, 3508, 439, 4024, 3512, 4029, 4037, 4045, 3023, 465, 3026, 3542, 3543, 984, 473, 3544, 990, 3554, 995, 3563, 1006, 3568, 3573])
    # idx2 = np.array([8, 2088, 1637, 2108, 75, 78, 2153, 113, 126, 156, 191, 198, 234, 247, 283, 2369, 4418, 338, 344, 354, 399, 412, 435, 439, 465, 473, 530, 555, 558, 559, 568, 2647, 604, 2658, 616, 634, 644, 2701, 659, 662, 663, 665, 675, 681, 2750, 2753, 721, 724, 725, 734, 2793, 770, 2823, 800, 2857, 810, 2860, 825, 828, 829, 831, 841, 846, 2915, 2918, 886, 889, 890, 899, 2958, 922, 933, 2988, 3023, 3026, 984, 1417, 990, 1418, 995, 1419, 1006, 3081, 3084, 1049, 1052, 1054, 1059, 3108, 3110, 1068, 1073, 3121, 1436, 1437, 1438, 1092, 1440, 1104, 1441, 1113, 1114, 3166, 1122, 3182, 1142, 1156, 1159, 1161, 1163, 3212, 1165, 3213, 3214, 1169, 3224, 1179, 1184, 3233, 3246, 3256, 1215, 1218, 1220, 1225, 1234, 3282, 1239, 3292, 3302, 1258, 1259, 1265, 3319, 3320, 3321, 3323, 3324, 1283, 1285, 1286, 3333, 3343, 3347, 1482, 1325, 1328, 3377, 1330, 3378, 3379, 1335, 3389, 1344, 1349, 3398, 1353, 1361, 3411, 3421, 1382, 1383, 1384, 1385, 1386, 1387, 1388, 1389, 1390, 1391, 1392, 1393, 1394, 1395, 1396, 1397, 1398, 3447, 1399, 1400, 1401, 1402, 1403, 1405, 1404, 1406, 1407, 3457, 1408, 1409, 1410, 1413, 1414, 1411, 1412, 1415, 1416, 3467, 1420, 1421, 1422, 1423, 1424, 1425, 1426, 1427, 1428, 1429, 1430, 1431, 1432, 1433, 1434, 1435, 3484, 3485, 3486, 1439, 3488, 3489, 1442, 1443, 1444, 1445, 1446, 1447, 1448, 1449, 1450, 3498, 1452, 1451, 1453, 1455, 1454, 1456, 1457, 1459, 1460, 1458, 3508, 1463, 1464, 1465, 1461, 1462, 3512, 1466, 1467, 1468, 1469, 1470, 1471, 1472, 1474, 1473, 1475, 1476, 1477, 1478, 1479, 1480, 1483, 1481, 1486, 1487, 1488, 1489, 1490, 1491, 1484, 1485, 1494, 1495, 1496, 1497, 1498, 1492, 1493, 3542, 1502, 3543, 3544, 1499, 1500, 1501, 1503, 1504, 1505, 3554, 1506, 1509, 1507, 1508, 1510, 1511, 1512, 3563, 1513, 1514, 1515, 3568, 1516, 3573, 1517, 1518, 1519, 1588, 1590, 1591, 1592, 1593, 1594, 1595, 1596, 1597, 1598, 1599, 1600, 1601, 1602, 1603, 1604, 1605, 1606, 1607, 1608, 1609, 1610, 1611, 1612, 1613, 1614, 1615, 1616, 1617, 1618, 1619, 1620, 1621, 1622, 1623, 1624, 3674, 3675, 1630, 1631, 1632, 1633, 1634, 1635, 1636, 3685, 1638, 1639, 1640, 1641, 1642, 1643, 1644, 1645, 3694, 1647, 1648, 1649, 1650, 1651, 1652, 1653, 1654, 1646, 1691, 3753, 3763, 3781, 3782, 3784, 3785, 3794, 3804, 3839, 3840, 3850, 3859, 1856, 3918, 3928, 3946, 3947, 3949, 3950, 3959, 3969, 4004, 4005, 4015, 4024, 4029, 4037, 4045])
    # idx3 = np.array([8, 1141, 2088, 1637, 2108, 75, 78, 2153, 113, 126, 156, 191, 198, 234, 247, 283, 2369, 4418, 338, 344, 354, 399, 412, 435, 439, 465, 473, 530, 555, 558, 559, 568, 2647, 604, 2658, 616, 634, 644, 2701, 659, 662, 663, 665, 675, 681, 2750, 2753, 721, 724, 725, 734, 2793, 770, 2823, 800, 2857, 810, 2860, 825, 828, 829, 831, 841, 846, 2915, 2918, 1521, 886, 889, 890, 899, 2958, 922, 933, 2988, 3023, 3026, 984, 1417, 990, 1418, 995, 1419, 1522, 1006, 3081, 3084, 1049, 1052, 1054, 1059, 3108, 3110, 1068, 1073, 3121, 1436, 1437, 1438, 1092, 1440, 1104, 1106, 1441, 1108, 1109, 1110, 1111, 1112, 1113, 1114, 1115, 1116, 1117, 3166, 1119, 1120, 1121, 1122, 1523, 1123, 1124, 1125, 1126, 1127, 1129, 1128, 1130, 1131, 1132, 3182, 1133, 1134, 1135, 1136, 1137, 1138, 1139, 1142, 1140, 1144, 1145, 1146, 1147, 1148, 1149, 1150, 1151, 1152, 1153, 1154, 1155, 1156, 1157, 1158, 1159, 1160, 1161, 1162, 1163, 3212, 1165, 3213, 3214, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 3224, 1177, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 3233, 1186, 1187, 1188, 1189, 1190, 1191, 1192, 1193, 1194, 1195, 1196, 1197, 3246, 1198, 1199, 1200, 1201, 1202, 1203, 1204, 1205, 1206, 3256, 1207, 1208, 1209, 1212, 1213, 1214, 1215, 1216, 1217, 1218, 1219, 1220, 1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228, 1229, 1230, 1231, 1232, 1233, 1234, 3282, 1236, 1237, 1238, 1239, 1235, 1241, 1240, 1242, 3292, 1524, 3302, 1258, 1259, 1265, 3319, 3320, 3321, 3323, 3324, 1283, 1285, 1286, 3333, 3343, 3347, 1482, 1325, 1328, 3377, 1330, 3378, 3379, 1331, 1332, 1335, 1336, 3389, 1526, 1344, 1349, 3398, 1353, 1361, 3411, 1525, 3421, 1118, 1378, 1379, 1380, 1381, 1382, 1383, 1384, 1385, 1386, 1387, 1388, 1389, 1390, 1391, 1392, 1393, 1394, 1395, 1396, 1397, 1398, 3447, 1399, 1400, 1401, 1402, 1403, 1405, 1404, 1406, 1407, 3457, 1408, 1409, 1410, 1413, 1414, 1411, 1412, 1415, 1416, 3467, 1420, 1421, 1422, 1423, 1424, 1425, 1426, 1427, 1428, 1429, 1430, 1431, 1432, 1433, 1434, 1435, 3484, 3485, 3486, 1439, 3488, 3489, 1442, 1443, 1444, 1445, 1446, 1447, 1448, 1449, 1450, 3498, 1452, 1451, 1453, 1455, 1454, 1456, 1457, 1459, 1460, 1458, 3508, 1463, 1464, 1465, 1461, 1462, 3512, 1466, 1467, 1468, 1469, 1470, 1471, 1472, 1474, 1473, 1475, 1476, 1477, 1478, 1479, 1480, 1483, 1481, 1486, 1487, 1488, 1489, 1490, 1491, 1484, 1485, 1494, 1495, 1496, 1497, 1498, 1492, 1493, 3542, 1502, 3543, 3544, 1499, 1500, 1501, 1503, 1504, 1505, 3554, 1506, 1509, 1507, 1508, 1510, 1511, 1512, 3563, 1513, 1514, 1515, 3568, 1516, 3573, 1517, 1518, 1519, 1520, 1530, 1531, 1532, 1533, 1534, 1535, 1536, 1537, 1538, 1539, 1540, 1541, 1542, 1543, 1544, 1545, 1546, 1547, 1548, 1549, 1550, 1551, 1529, 1528, 1527, 1552, 1553, 1554, 1555, 1559, 1560, 1561, 1562, 1563, 1564, 1565, 1566, 1567, 1568, 1569, 1570, 1571, 1572, 1573, 1574, 1575, 1576, 1577, 1578, 1579, 1580, 1581, 1582, 1583, 1584, 1585, 1586, 1587, 1588, 1589, 1590, 1591, 1592, 1593, 1594, 1595, 1596, 1597, 1598, 1599, 1600, 1601, 1602, 1603, 1604, 1605, 1606, 1607, 1608, 1609, 1610, 1611, 1612, 1613, 1614, 1615, 1616, 1617, 1618, 1619, 1620, 1621, 1622, 1623, 1624, 1625, 3674, 3675, 1628, 1629, 1630, 1631, 1632, 1633, 1634, 1635, 1636, 3685, 1638, 1639, 1640, 1641, 1642, 1643, 1644, 1645, 3694, 1647, 1648, 1649, 1650, 1651, 1652, 1653, 1654, 1646, 1143, 1556, 1557, 1691, 1558, 3753, 3763, 3781, 3782, 3784, 3785, 3794, 3804, 1164, 1166, 1167, 3839, 3840, 3850, 3859, 1176, 1856, 1185, 3918, 3928, 3946, 3947, 3949, 3950, 3959, 3969, 4004, 4005, 4015, 4024, 4029, 4037, 1210, 4045, 1211, 1626, 1627, 1107])
    # idx4 = np.array([2048, 2049, 2050, 2051, 2052, 2053, 2054, 2055, 8, 2057, 2058, 2059, 2060, 2061, 2062, 2063, 2064, 2065, 2066, 2067, 2068, 2039, 2070, 2042, 1141, 2088, 1637, 2045, 2108, 75, 78, 2153, 2056, 113, 1244, 1245, 126, 1247, 156, 1254, 191, 198, 234, 247, 283, 1279, 1280, 1281, 1282, 1284, 2369, 4418, 1287, 338, 344, 354, 1295, 1299, 399, 412, 435, 439, 1313, 465, 473, 530, 1329, 555, 558, 559, 568, 2647, 604, 2658, 616, 634, 1348, 644, 1350, 2701, 1351, 659, 1352, 662, 663, 665, 1354, 675, 681, 2750, 2753, 1364, 721, 724, 725, 734, 2793, 1372, 1373, 1374, 770, 2823, 800, 2857, 810, 2860, 825, 828, 829, 831, 841, 846, 2915, 2918, 1521, 886, 889, 890, 899, 2958, 922, 933, 2988, 3023, 3026, 984, 1417, 990, 1418, 995, 1419, 1522, 1006, 3081, 3084, 1049, 1052, 1054, 1059, 3108, 3110, 1068, 1073, 3121, 1436, 1437, 1438, 1092, 1440, 1104, 1106, 1441, 1108, 1109, 1110, 1111, 1112, 1113, 1114, 1115, 1116, 1117, 3166, 1119, 1120, 1121, 1122, 1523, 1123, 1124, 1125, 1126, 1127, 1129, 1128, 1130, 1131, 1132, 3182, 1133, 1134, 1135, 1136, 1137, 1138, 1139, 1142, 1140, 1144, 1145, 1146, 1147, 1148, 1149, 1150, 1151, 1152, 1153, 1154, 1155, 1156, 1157, 1158, 1159, 1160, 1161, 1162, 1163, 3212, 1165, 3213, 3214, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 3224, 1177, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 3233, 1186, 1187, 1188, 1189, 1190, 1191, 1192, 1193, 1194, 1195, 1196, 1197, 3246, 1198, 1199, 1200, 1201, 1202, 1203, 1204, 1205, 1206, 3256, 1207, 1208, 1209, 1212, 1213, 1214, 1215, 1216, 1217, 1218, 1219, 1220, 1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228, 1229, 1230, 1231, 1232, 1233, 1234, 3282, 1236, 1237, 1238, 1239, 1235, 1241, 1240, 1242, 3292, 1243, 1246, 1524, 1248, 1249, 1250, 1251, 1252, 1253, 3302, 1255, 1256, 1257, 1258, 1259, 1260, 1261, 1262, 1263, 1264, 1265, 1266, 1267, 1268, 1269, 1270, 3319, 3320, 3321, 1271, 3323, 3324, 1272, 1273, 1274, 1275, 1276, 1277, 1283, 1278, 1285, 1286, 3333, 1288, 1289, 1290, 1291, 1292, 1293, 1294, 3343, 1296, 1297, 1298, 3347, 1300, 1301, 1302, 1303, 1304, 1305, 1306, 1307, 1308, 1309, 1310, 1311, 1312, 1482, 1314, 1315, 1316, 1317, 1318, 1319, 1320, 1321, 1322, 1323, 1324, 1325, 1326, 1327, 1328, 3377, 1330, 3378, 3379, 1331, 1332, 1335, 1336, 1333, 1334, 1337, 1338, 3389, 1526, 1339, 1344, 1340, 1341, 1342, 1343, 1349, 3398, 1345, 1346, 1353, 1347, 1355, 1356, 1357, 1358, 1359, 1360, 1361, 1362, 3411, 1363, 1365, 1366, 1367, 1368, 1369, 1370, 1371, 1525, 3421, 1118, 1375, 1376, 1377, 1378, 1379, 1380, 1381, 1382, 1383, 1384, 1385, 1386, 1387, 1388, 1389, 1390, 1391, 1392, 1393, 1394, 1395, 1396, 1397, 1398, 3447, 1399, 1400, 1401, 1402, 1403, 1405, 1404, 1406, 1407, 3457, 1408, 1409, 1410, 1413, 1414, 1411, 1412, 1415, 1416, 3467, 1420, 1421, 1422, 1423, 1424, 1425, 1426, 1427, 1428, 1429, 1430, 1431, 1432, 1433, 1434, 1435, 3484, 3485, 3486, 1439, 3488, 3489, 1442, 1443, 1444, 1445, 1446, 1447, 1448, 1449, 1450, 3498, 1452, 1451, 1453, 1455, 1454, 1456, 1457, 1459, 1460, 1458, 3508, 1463, 1464, 1465, 1461, 1462, 3512, 1466, 1467, 1468, 1469, 1470, 1471, 1472, 1474, 1473, 1475, 1476, 1477, 1478, 1479, 1480, 1483, 1481, 1486, 1487, 1488, 1489, 1490, 1491, 1484, 1485, 1494, 1495, 1496, 1497, 1498, 1492, 1493, 3542, 1502, 3543, 3544, 1499, 1500, 1501, 1503, 1504, 1505, 3554, 1506, 1509, 1507, 1508, 1510, 1511, 1512, 3563, 1513, 1514, 1515, 3568, 1516, 3573, 1517, 1518, 1519, 1520, 1530, 1531, 1532, 1533, 1534, 1535, 1536, 1537, 1538, 1539, 1540, 1541, 1542, 1543, 1544, 1545, 1546, 1547, 1548, 1549, 1550, 1551, 1529, 1528, 1527, 1552, 1553, 1554, 1555, 1559, 1560, 1561, 1562, 1563, 1564, 1565, 1566, 1567, 1568, 1569, 1570, 1571, 1572, 1573, 1574, 1575, 1576, 1577, 1578, 1579, 1580, 1581, 1582, 1583, 1584, 1585, 1586, 1587, 1588, 1589, 1590, 1591, 1592, 1593, 1594, 1595, 1596, 1597, 1598, 1599, 1600, 1601, 1602, 1603, 1604, 1605, 1606, 1607, 1608, 1609, 1610, 1611, 1612, 1613, 1614, 1615, 1616, 1617, 1618, 1619, 1620, 1621, 1622, 1623, 1624, 1625, 3674, 3675, 1628, 1629, 1630, 1631, 1632, 1633, 1634, 1635, 1636, 3685, 1638, 1639, 1640, 1641, 1642, 1643, 1644, 1645, 3694, 1647, 1648, 1649, 1650, 1651, 1652, 1653, 1654, 1646, 1143, 1655, 1656, 1657, 1556, 1557, 1691, 1558, 3753, 3763, 3781, 3782, 3784, 3785, 3794, 3804, 1164, 1166, 1167, 3839, 3840, 3850, 3859, 1176, 1856, 1185, 3918, 3928, 3946, 3947, 3949, 3950, 3959, 3969, 1952, 1953, 1954, 1955, 4004, 4005, 1958, 1956, 1957, 1959, 1960, 4015, 4024, 4029, 4037, 1210, 4045, 1211, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035, 2036, 1626, 2037, 2038, 2040, 2041, 1627, 2043, 2044, 1107, 2046, 2047])
    # idx_all = [idx1, idx2, idx3, idx4]
    # for i, idx in enumerate(idx_all):
    #     X_train[:, idx]
    #     print(np.isnan(X).sum(axis=0).sum(), np.isinf(X).sum(axis=0).sum())
