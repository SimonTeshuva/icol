import warnings
warnings.filterwarnings('ignore')

from time import time
from copy import deepcopy
from itertools import combinations, permutations

import numpy as np
import sympy as sp

from sklearn.linear_model import lars_path
from sklearn.preprocessing import PolynomialFeatures
from sklearn.base import clone

from sklearn.metrics import mean_squared_error

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)

def LL(res):
    n = len(res)
    return n*np.log(np.sum(res**2)/n)

IC_DICT = {
    'AIC': lambda res, k: LL(res) + 2*k,
    'HQIC': lambda res, k: LL(res) + np.log(np.log(len(res)))*k,
    'AIC': lambda res, k: LL(res) + 2*k,
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
    def __init__(self, gamma=1, fit_intercept=False, default_d=5, rcond=-1):
        self.gamma = gamma
        self.fit_intercept = fit_intercept
        self.default_d = default_d
        self.rcond=rcond

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

    def __call__(self, X, y, d, rcond=None, verbose=False):

        self.set_default_d(d)

        if np.abs(self.gamma)<1e-10:
            beta_hat = np.ones(X.shape[1])
            w_hat = np.ones(X.shape[1])
            X_star_star = X.copy()
        else:
            beta_hat, _, _, _ = np.linalg.lstsq(X, y, rcond=self.rcond)

            w_hat = 1/np.power(np.abs(beta_hat), self.gamma)
            X_star_star = np.zeros_like(X)
            for j in range(X_star_star.shape[1]): # vectorise
                X_j = X[:, j]/w_hat[j]
                X_star_star[:, j] = X_j

        _, _, coefs, _ = lars_path(X_star_star, y.ravel(), return_n_iter=True, max_iter=d, method='lasso')
        # alphas, active, coefs = lars_path(X_star_star, y.ravel(), method='lasso')
        try:           
            beta_hat_star_star = coefs[:, d]
        except IndexError:
            beta_hat_star_star = coefs[:, -1]
        beta_hat_star_n = np.array([beta_hat_star_star[j]/w_hat[j] for j in range(len(beta_hat_star_star))])
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
    def __init__(self, s, so, d, fit_intercept=True, normalize=True, pool_reset=False, information_criteria=None): #, track_intermediates=False):
        self.s = s
        self.sis = SIS(n_sis=s)
        self.so = so
        self.d = d
        self.fit_intercept = fit_intercept
        self.normalize=normalize
        self.pool_reset = pool_reset
        self.information_criteria = information_criteria if information_criteria in IC_DICT.keys() else None
        # self.track_intermediates = track_intermediates
    
    def get_params(self, deep=False):
        return {'s': self.s,
                'so': self.so,
                'd': self.d,
                'fit_intercept': self.fit_intercept,
                'normalize': self.normalize,
                'pool_reset': self.pool_reset,
                'information_criteria': self.information_criteria
                }

    def __str__(self):
        return 'SISSO(n_sis={0}, SO={1}, d={2})'.format(self.s, str(self.so), self.d)

    def __repr__(self, prec=3):
        ret = []
        for i, name in enumerate(self.feature_names_sparse_):
            ret += [('+' if self.coef_[0, i] > 0 else '') + str(np.round(self.coef_[0, i], prec)) + str(name)]
        ret += ['+' + str(float(np.round(self.intercept_, prec)))]
        return ''.join(ret)
    
        # return '+'.join(['{0}({1})'.format(str(np.round(b, 3)), self.feature_names_sparse_[i]) for i, b in enumerate(self.coef_) if np.abs(b) > 0]+[str(self.intercept_)])
     
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

    def fitting(self, X, y, feature_names=None, verbose=False, track_pool=False, track_intermediates=False):
        self.feature_names_ = feature_names
        n,p = X.shape

        pool_ = set()
        if track_pool: self.pool = []
        if track_intermediates: self.intermediates = np.empty(shape=(self.d, 5), dtype=object)
        res = y
        i = 0
        IC = np.infty
        cont = True
        while i < self.d and cont:
            self.intercept_ = np.mean(res).squeeze()
            if verbose: print('.', end='')

            p, sis_i = self.sis(X=X, res=res, pool=list(pool_), verbose=verbose)
            pool_.update(sis_i)
            pool_lst = list(pool_)
            
            if track_pool: self.pool = pool_lst
            beta_i = self.so(X=X[:, pool_lst], y=y, d=i+1, verbose=verbose)

            beta = np.zeros(shape=(X.shape[1]))
            beta[pool_lst] = beta_i

            if track_intermediates:
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
            if not(self.information_criteria is None):
                IC_old = IC
                IC = IC_DICT[self.information_criteria](res=res, k=i+1)
                if verbose: print('{0}={1}'.format(self.information_criteria, IC))
                cont = IC < IC_old

            i += 1
        if track_intermediates: self.intermediates = self.intermediates[:, :i]
            
        if verbose: print()
        
        self.beta_ = beta
        self.intercept_ = np.mean(res).squeeze()

        self.beta_idx_ = list(np.nonzero(self.beta_)[0])
        self.beta_sparse_ = self.beta_[self.beta_idx_]
        self.feature_names_sparse_ = np.array(self.feature_names_)[self.beta_idx_]

        return self

    def fit(self, X, y, feature_names=None, timer=False, verbose=False, track_pool=False, track_intermediates=False):
        if verbose: print('removing invalid features')
        self.bad_col = self.filter_invalid_cols(X)
        X_ = np.delete(X, self.bad_col, axis=1)
        have_valid_names = not(feature_names is None) and X.shape[1] == len(feature_names)
        feature_names_ = np.delete(np.array(feature_names), self.bad_col) if have_valid_names else ['X_{0}'.format(i) for i in range(X_.shape[1])]
      
        if verbose: print('Feature normalisation')
        self.solve_norm_coef(X_, y)
        X_, y_ = self.normalize_Xy(X_, y)

        if verbose: print('Fitting SISSO model')
        if timer: start=time()
        self.fitting(X=X_, y=y_, feature_names=feature_names_, verbose=verbose, track_pool = track_pool, track_intermediates=track_intermediates)
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

    def __call__(self, X, feature_names=None, verbose=False, f=None):
        if verbose: print('Prepping Symbols')
        if feature_names is None: feature_names = sp.symbols(' '.join(['x_{0}'.format(i) for i in range(X.shape[1])]))
        if verbose: print('Performing Feature Expansion')
        if verbose: print('Estimating the creation of {0} features with duplicates'.format(self.extimate_workload(X=X, max_rung=self.rung)))
        spnames, names, X_ = self.FE_aux(X=X, feature_names=feature_names, rung=self.rung, max_rung=self.rung, prev_start = -1, verbose=verbose)
        if verbose: print('Created {0} features, now removing duplicate features'.format(X_.shape[1]))
        spnames, names, X_ = self.remove_redundant_features(spnames, names, X_)
        if f:
            pass
        return spnames, names, X_

    def remove_redundant_features(self, spnames, names, X_):
        sorted_idxs = np.argsort(names)
        for i, idx in enumerate(sorted_idxs):
            if i == 0:
                unique = [idx]
            elif names[idx] != names[sorted_idxs[i-1]]:
                unique += [idx]
        unique_original_order = np.sort(unique)
        return spnames[unique_original_order], names[unique_original_order], X_[:, unique_original_order]
            
    def extimate_workload(self, X, max_rung):
        rung = max_rung
        p = X.shape[1]
        p_prev = X.shape[1]
        unary = 0
        binary = 0
        for op in self.ops:
            if OP_DICT[op]['inputs'] == 1:
                unary += 1
            elif OP_DICT[op]['inputs'] == 2:
                binary += 1
        while rung > 0:
            new_unary = unary*(p-p_prev) if rung != max_rung else unary*p
            new_binary = int(binary*(p-p_prev)*(p-1)) if rung != max_rung else int(binary*p*(p-1)/2)
            p_prev = p
            p = p + new_unary + new_binary
            rung -= 1
        return p

    def FE_aux(self, X, feature_names, prev_start, rung=0, max_rung=0, verbose=False):

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
        
        # if rung == max_rung:
        #     feature_names = np.array(feature_names)
        #     sympy_names = np.array([str(name) for name in feature_names])

        if rung <= 0:
            return (np.array(feature_names), 
                    np.array(
                        [str(sp.simplify(simplify_nested_powers(name)))for name in feature_names]), 
                    X)
        else:
            if verbose: print('Creating rung {0} features'.format(max_rung - rung+1)) 
            new_names = ()
            for op_key in self.ops:
                if OP_DICT[op_key]['inputs'] == 1:
                    for i in range(prev_start, len(feature_names)):
                        if verbose and ((len(feature_names) + len(new_names)) % self.printrate == 0): print('Created {0} Features'.format(len(feature_names) + len(new_names)))
                        if len(new_names) == 0:
                            new_X = OP_DICT[op_key]['op_np'](X[:, i]).reshape(X.shape[0], 1)
                        else:
                            new_X = np.hstack([new_X, OP_DICT[op_key]['op_np'](X[:, i]).reshape(X.shape[0], 1)])
                        new_names += (OP_DICT[op_key]['op'](feature_names[i]), )
                        if verbose>1: print(new_names[-1])
                elif OP_DICT[op_key]['inputs'] == 2:
                    pairings = combinations if OP_DICT[op_key]['commutative'] else permutations
                    for idx1, idx2 in pairings(range(len(feature_names)), 2): 
                        if verbose and ((len(feature_names) + len(new_names)) % self.printrate == 0): print('Created {0} Features'.format(len(feature_names) + len(new_names)))
                        # make sure at least one of the features if from the new features
                        if idx1 >= prev_start or idx2 >= prev_start: 
                            new_col = OP_DICT[op_key]['op_np'](X[:, idx1], X[:, idx2]).reshape(X.shape[0], 1).reshape(X.shape[0], 1)
                            new_X = new_col if len(new_names) == 0 else np.hstack([new_X,new_col])
                            new_name = OP_DICT[op_key]['op'](feature_names[idx1],feature_names[idx2])
                            new_names += (new_name, )
                            if verbose > 1: print(new_name)
            if new_names == ():
                return self.FE_aux(X = X, feature_names=feature_names, rung=rung-1, prev_start=len(feature_names), max_rung=max_rung, verbose=verbose)
            else:
                return self.FE_aux(X = np.hstack([X, new_X]), feature_names=feature_names+new_names, rung=rung-1, prev_start=len(feature_names), max_rung=max_rung, verbose=verbose)

if __name__ == "__main__":
    random_state = 0
    n = 100
    p = 10
    rung = 3
    s = 5
    d = 4

    np.random.seed(random_state)
    X_train = np.random.normal(size=(n, p))

    y = lambda X: X[:, 0] + 2*X[:, 1]**2 - X[:, 0]*X[:, 1] + 3*X[:, 2]**3
    y_train = y(X_train)

    # Initialise and fit the ICL model
    FE = PolynomialFeaturesICL(rung=rung, include_bias=False)
    so = AdaptiveLASSO(gamma=1, fit_intercept=False)
    information_criteria='BIC'

    X_train_transformed = FE.fit_transform(X_train, y)
    feature_names = FE.get_feature_names_out()

    icl = ICL(s=s, so=so, d=d, fit_intercept=True, normalize=True, pool_reset=False, information_criteria=information_criteria)
    icl.fit(X_train_transformed, y_train, feature_names=feature_names, verbose=True, track_intermediates=True)

    # Compute the train and test error and print the model to verify that we have reproduced the data generating function
    print(icl)
    print(icl.__repr__())

    y_hat_train = icl.predict(X_train_transformed)

    print("Train rmse: " + str(rmse(y_hat_train, y_train)))

    X_test = np.random.normal(size=(100*n, p))
    X_test_transformed = FE.transform(X_test)
    y_test = y(X_test)
    y_hat_test = icl.predict(X_test_transformed)
    print("Test rmse: " + str(rmse(y_hat_test, y_test)))
    print("k={0}".format(len(icl.coef_[0])))

    # print(icl.intermediates)