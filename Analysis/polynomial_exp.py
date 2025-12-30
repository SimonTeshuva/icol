import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt

from icol.icol import *
from time import time
from sklearn.base import clone
from sklearn.model_selection import KFold

from problem_samplers.polynomial_generator import *

# for i, f in enumerate(os.listdir(outdir)):
#     if i==0:
#         df = pd.read_csv(os.path.join(outdir, f))
#     else:
#         df_i = pd.read_csv(os.path.join(outdir, f))
#         df = df.append(df_i, ignore_index=True)
#     print(i, f)
out_dir = os.path.join(os.getcwd(), 'Output')
n_test = 5000
k = 5

choice = 'testing'
## Oracle work
if choice == 'Oracle':
    First = False
    SO = [AdaptiveLASSO(gamma=1, fit_intercept=False), AdaptiveLASSO(gamma=0, fit_intercept=False),
        BSS(), ThresholdedLeastSquares(), LARS()]
    D = [1, 2, 3, 4, 5, 6, 7]
    RS = range(10)
    N = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]    
    REPS = range(10)
    S = np.array([1,2,3,4,5,6,7,8,9,
        10,20,30,40,50,60,70,80,90,
        100,200,300,400,500,600,700,800,900,
        1000,2000,3000])
    S_MAX_DICT = {
        'AdaLASSO(gamma=1)': lambda k, n, p: 500,
        'LASSO': lambda k, n, p: 2000,
        'BSS': lambda k, n, p: 7,
        'TLS': lambda k, n, p: 600,
        'Lars': lambda k, n, p: 2000
    }
    done_d = []
    done_RS = []
    done_rep = []
    FOLDS = [1]
elif choice == 'CV':
    # CV
    First = False
    SO = [AdaptiveLASSO(gamma=1, fit_intercept=False), AdaptiveLASSO(gamma=0, fit_intercept=False),
        BSS(), ThresholdedLeastSquares(), LARS()]
    S_MAX_DICT = {
        'AdaLASSO(gamma=1)': lambda k, n, p: 90,
        'LASSO': lambda k, n, p: 200,
        'BSS': lambda k, n, p: 4,
        'TLS': lambda k, n, p: 200,
        'Lars': lambda k, n, p: 200
    }
    D = [1,2,3,4,5,6,7]
    RS = range(10)
    N = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]    
    REPS = range(10)
    S = np.array([1,2,3,4,5,6,7,8,9,
        10,20,30,40,50,60,70,80,90,
        100,200,300,400,500,600,700,800,900,
        1000,2000,3000])
    done_d = [1,2,3,5,6,7]
    done_RS = []
    done_rep = []
    FOLDS = [10]
elif choice == 'testing':
    First = True
    SO = [AdaptiveLASSO(gamma=1, fit_intercept=False), AdaptiveLASSO(gamma=0, fit_intercept=False),
        BSS(), ThresholdedLeastSquares(), LARS()]
    D = [1]
    RS = range(10)
    N = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]    
    REPS = range(10)
    S = np.array([1,2,3,4,5,6,7,8,9,
        10,20,30,40,50,60,70,80,90,
        100,200,300,400,500,600,700,800,900,
        1000,2000,3000])
    S_MAX_DICT = {
        'AdaLASSO(gamma=1)': lambda k, n, p: 500,
        'LASSO': lambda k, n, p: 2000,
        'BSS': lambda k, n, p: 7,
        'TLS': lambda k, n, p: 600,
        'Lars': lambda k, n, p: 2000
    }
    done_d = [2,3,4,5,6,7]
    done_RS = [2,3,4,5,6,7,8,9]
    done_rep = [2,3,4,5,6,7,8,9]
    FOLDS = [1]

p0 = 10
r = 5
num_terms = 5

SAMPLERS = np.empty(shape=(len(D), len(RS)), dtype=object)
for i, d in enumerate(D):
    for j, random_state in enumerate(RS):
        print(d, random_state)
        sampler = function_sampler(p=p0, r=r, num_terms=num_terms, degree=d, random_state=random_state) 
        sampler.genearte_rho_matrix()
        sampler.generate_formula()
        sampler.generate_r_prime()
        sampler.generate_sampler_idxs()
        SAMPLERS[i, j] = sampler

res_df = pd.DataFrame()
total_fit_times = 0

for i, d in enumerate(D):
    if d not in done_d:
        FE = PolynomialFeatures(degree=d, include_bias=False)
        for random_state in RS:
            if random_state not in done_RS:
                sampler = SAMPLERS[d-1, random_state]
                X_test, y_test = sampler.sample(n_test)
                Phi_test = FE.fit_transform(X=X_test, y=y_test)
                p = Phi_test.shape[1]
                feature_names = FE.get_feature_names_out()
                for rep in REPS:
                    for j, n in enumerate(N):
                        X_train, y_train = sampler.sample(n)
                        if rep not in done_rep:
                            Phi_train = FE.transform(X_train)
                            for o, so in enumerate(SO):
                                s_max = S_MAX_DICT[str(so)](k=k, n=n, p=p)
                                for l, s in enumerate(S[S<=s_max]):
                                    s_prev = 1 if l == 0 else S[l-1]
                                    if s_prev*k < Phi_train.shape[1]:
                                        for m, f in enumerate(FOLDS):
                                            if f == 1:
                                                print('d={0}, random_state={1}, rep={2}, n={3}, so={6}, s={4}, f={5}'.format(d, random_state, rep, n, s, f, so))
                                                icl = ICL(s=s, so=clone(so), k=k)
                                                start = time()
                                                icl.fit(X=Phi_train, y=y_train, feature_names=feature_names, verbose=False, random_state=random_state)                    
                                                fit_time = time() - start
                                                total_fit_times += fit_time
                                                y_hat_train = icl.predict(Phi_train)
                                                y_hat_test = icl.predict(Phi_test)
                                                rmse_train = rmse(y_train, y_hat_train)
                                                rmse_test = rmse(y_test, y_hat_test)
                                                row = {
                                                    'p0': int(p0), 'p': int(Phi_train.shape[1]), 'n': int(Phi_train.shape[0]), 'k': int(icl.k), 'r': int(sampler.r),
                                                    'd': int(d), 'random_state': int(random_state), 'rep': int(rep), 's': int(s), 'so': str(icl.so), 'f': int(f),
                                                    'model': -1, 
                                                    'model_coef': -1, 
                                                    'formula': str(sampler.monomials_used_idxs).replace(',', ';'), 
                                                    'formula_coef': -1,
                                                    'f_i': f, 'fit_time': fit_time, 'test_rmse': rmse_test, 'train_rmse': rmse_train, 
                                                    'model_formula': str(icl.beta_idx_).replace(',', ';'),
                                                    'inclusion': len(set(icl.beta_idx_).intersection(set(sampler.monomials_used_idxs)))/k, 
                                                    'model_stes': -1,
                                                    'sampler_stes': -1,
                                                    'beta_rmse': -1,
                                                    'beta_armse': -1,
                                                    'val_rmse': -1
                                                }
                                                print('train_rmse={0}, test_rmse={1}, inclusion={2}, fit_time={3}'.format(row['train_rmse'], row['test_rmse'], row['inclusion'], row['fit_time']))
                                                res_df = res_df.append(row, ignore_index=True)
                                            else:
                                                cv = KFold(n_splits=f)
                                                for f_i, (fold_idx, val_idx) in enumerate(cv.split(Phi_train)):
                                                    print('d={0}, random_state={1}, rep={2}, n={3},so={7}, s={4}, f={5}, f_i={6}'.format(d, random_state, rep, n, s, f, f_i, str(so)))
                                                    Phi_fold, Phi_val, y_fold, y_val = Phi_train[fold_idx], Phi_train[val_idx], y_train[fold_idx], y_train[val_idx]
                                                    icl = ICL(s=s, so=clone(so), k=k)
                                                    start = time()
                                                    icl.fit(X=Phi_fold, y=y_fold, feature_names=feature_names, verbose=False, random_state=random_state)                    
                                                    fit_time = time() - start
                                                    total_fit_times += fit_time
                                                    y_hat_fold = icl.predict(Phi_fold)
                                                    y_hat_val = icl.predict(Phi_val)
                                                    rmse_train = rmse(y_fold, y_hat_fold)
                                                    rmse_test = rmse(y_val, y_hat_val)
                                                    row = {
                                                        'p0': int(p0), 'p': int(Phi_train.shape[1]), 'n': int(Phi_train.shape[0]), 'k': int(icl.k), 'r': int(sampler.r),
                                                        'd': int(d), 'random_state': int(random_state), 'rep': int(rep), 's': int(s), 'so': str(icl.so), 'f': int(f),
                                                        'model': -1, 
                                                        'model_coef': -1, 
                                                        'formula': str(sampler.monomials_used_idxs).replace(',', ';'), 
                                                        'formula_coef': -1,
                                                        'f_i': f_i, 'fit_time': fit_time, 'test_rmse': rmse_test, 'train_rmse': rmse_train, 
                                                        'model_formula': str(icl.beta_idx_).replace(',', ';'),
                                                        'inclusion': len(set(icl.beta_idx_).intersection(set(sampler.monomials_used_idxs)))/k, 
                                                        'model_stes': -1,
                                                        'sampler_stes': -1,
                                                        'beta_rmse': -1,
                                                        'beta_armse': -1,
                                                        'val_rmse': -1
                                                    }
                                                    print('train_rmse={0}, test_rmse={1}, inclusion={2}, fit_time={3}, f_i={4}'.format(row['train_rmse'], row['test_rmse'], row['inclusion'], row['fit_time'], f_i))
                                                    res_df = res_df.append(row, ignore_index=True)
                                                    print('fit time so far: {0}'.format(total_fit_times))
        if FOLDS == [1]:                                                    
            res_dir = os.path.join(out_dir, 'Polynomial_Oracle_Exps', 'd={0}.csv'.format(d))
        else:
            res_dir = os.path.join(out_dir, 'Polynomial_CV_Exps', 'd={0}_folds={1}.csv'.format(d, str(FOLDS)))
        res_df[res_df['d']==d].to_csv(res_dir, header=First, mode='w' if First else 'a')
print(res_df.head())
print(len(res_df))