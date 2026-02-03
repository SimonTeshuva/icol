import warnings

from time import time
from copy import deepcopy
from itertools import combinations, permutations

import numpy as np
import sympy as sp

from sklearn.preprocessing import PolynomialFeatures
from sklearn.base import clone


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
    
    def expand(self, X, y=None, names=None, verbose=False, f=None, check_pos=False):
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
    