import numpy as np
import pandas as pd
from time import time
import os
from icol.icol import *
from sklearn.base import clone
from sklearn.metrics import r2_score

from fastridge import RidgeEM as fr

class PredictMean:
    def __init__(self):
        pass

    def get_params(self, deep=False):
        return {}
    
    def fit(self, X, y):
        self.intercept_ = y.mean()

    def predict(self, X):
        return (self.intercept_ * np.ones(shape=X.shape[0])).squeeze()

    def __str__(self):
        return 'PredictMean'
    
    def __repr__(self):
        return r'$\mathbb{E}[X] = {0}$'.format(self.intercept_)

def load_dataset(data_key, indir=os.path.join(os.getcwd(), 'Input')):
    f, drop_cols, target = DATA_DICT[data_key]['f'], DATA_DICT[data_key]['drop_cols'], DATA_DICT[data_key]['target'], 
    df = pd.read_csv(os.path.join(indir, f))
    y = df[target].values
    X = df.drop(columns=drop_cols+[target])
    names = X.columns
    X = X.values
    return X, y, names

def sample_data(X, y, n, ret_idx=False, random_state=None):
    if not(random_state is None): np.random.seed(random_state)
    idx = np.random.randint(low=0, high=X.shape[0], size=n)
    out = list(set(range(X.shape[0])) - set(idx))
    return (idx, out) if ret_idx else (X[idx], X[out], y[idx], y[out])

def cost_lim_fn(t_max = 30, t_min = 0.1, kmax = 5, nmax = 10000, pmax = 19477,
                S_grid = [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,2000,3000]):
    oracle_f = ['polynomial_oracle_d_1.csv',
            'polynomial_oracle_d_2.csv',
            'polynomial_oracle_d_3.csv',
            'polynomial_oracle_d_4.csv',
            'polynomial_oracle_d_5.csv',
            'polynomial_oracle_d_6.csv',
            'polynomial_oracle_d_7.csv',
            'results_polynomial_supplementary2024-11-19 18:47:32.186937.csv',
            'results_polynomial_supplementary2024-11-20 10:22:27.292263.csv',
            'results_polynomial_supplementary2024-11-20 11:10:21.219566.csv',
            'results_polynomial_supplementary2024-11-20 11:22:41.868501.csv',
            'results_polynomial_supplementary2024-11-24 17:27:49.777397.csv',
            'results_polynomial_supplementary2024-11-24 18:38:07.072042.csv',
            'results_polynomial_supplementary2024-11-24 18:47:38.323950.csv',
            'additional_so_d=1.csv',
            'additional_so_d=2.csv',
            'additional_so_d=3.csv',
            'additional_so_d=4.csv',
            'additional_so_d=5.csv',
            'additional_so_d=6.csv',
            'additional_so_d=7.csv',
            ]

    out_dir = os.path.join(os.getcwd(), 'Output', 'Cost_Aware_ICOL_Oracle_Results')

    for i, f in enumerate(oracle_f):
        path = os.path.join(out_dir, f)
        df_i = pd.read_csv(path)
        if i == 0:
            oracle_df = df_i
        else:
            oracle_df = oracle_df.append(df_i, ignore_index=True)

    N = oracle_df['n'].unique()
    P = oracle_df['p'].unique()
    D = oracle_df['d'].unique()
    SO = oracle_df['so'].unique()
    RS = oracle_df['random_state'].unique()
    REPS = oracle_df['rep'].unique()

    X = np.array([
        [1, 1],
        [1, kmax*nmax*pmax]
    ])
    y = np.array([
        [t_min],
        [t_max]
    ])

    c0, c1 = np.linalg.solve(X, y)
    c0, c1, c0 + c1*kmax*nmax*pmax, c0+c1

    from scipy.special import binom

    base_terms = lambda k, n, p, s: np.array([k*n, k*n*p, k*s + k*(k+1)/2])
    las_terms = lambda k, n, p, s: np.array([n*s*k**2, k**3])
    ala_terms = lambda k, n, p, s: np.array([n*s*k**2, k**3, n*(s*k)**2, (s*k)**3])
    bss_terms = lambda k, n, p, s: binom(s*k, k)*np.array([n*k**2, k**3])

    def cost(df, base_cost, so_cost, so):
        time_arr = df[df['so']==so][['k', 'n', 'p', 's', 'fit_time']].values
        num_base = len(base_cost(0,0,0,0))
        num_so = len(so_cost(0,0,0,0))
        X = np.zeros(shape = (len(time_arr), num_base+num_so))
        y = np.zeros(shape = len(time_arr))
        for i in range(len(time_arr)):
            b = base_cost(k=time_arr[i, 0], n=time_arr[i, 1], p=time_arr[i, 2], s=time_arr[i, 3])
            s = np.zeros(num_so)
            for l in range(1, 1+int(time_arr[i, 0])):
                s += so_cost(k=l, n=time_arr[i, 1], p=time_arr[i, 2], s=time_arr[i, 3])
            X[i, :] = np.hstack([b, s])
            y[i] = time_arr[i, 4]
        d0 = y.mean()
        y -= d0
        d, res, rank, singular = np.linalg.lstsq(X, y, rcond=False)
        return d0, d
    
    d0_las, d_las = cost(df=oracle_df, base_cost=base_terms, so_cost=las_terms, so='LASSO')
    d0_ala, d_ala = cost(df=oracle_df, base_cost=base_terms, so_cost=ala_terms, so='AdaLASSO(gamma=1)')
    d0_bss, d_bss = cost(df=oracle_df, base_cost=base_terms, so_cost=bss_terms, so='L0')
    d0_tls, d_tls = cost(df=oracle_df, base_cost=base_terms, so_cost=ala_terms, so='TLS')
    d0_lar, d_lar = cost(df=oracle_df, base_cost=base_terms, so_cost=ala_terms, so='Lars')

    def complexity_gen(base, so, d0, d):
        def complexity(k, n, p, s):
            so_cost = np.zeros(len(so(0,0,0,0)))
            for l in range(1, k+1):
                so_cost += so(k=l, n=n, p=p, s=s)
            return np.dot(d, np.hstack([base(k, n, p, s), so_cost])) + d0
        return complexity

    las_comp = complexity_gen(base=base_terms, so=las_terms, d0=d0_las, d=d_las)
    ala_comp = complexity_gen(base=base_terms, so=ala_terms, d0=d0_ala, d=d_ala)
    bss_comp = complexity_gen(base=base_terms, so=bss_terms, d0=d0_bss, d=d_bss)
    tls_comp = complexity_gen(base=base_terms, so=ala_terms, d0=d0_tls, d=d_tls)
    lar_comp = complexity_gen(base=base_terms, so=ala_terms, d0=d0_lar, d=d_lar)

    def find_s_max(comp, c0, c1, k, n, p, S_grid = [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,2000,3000]):
        found = False
        s_max = 1
        i = 0
        while (not found) and (i<len(S_grid)):
            s = S_grid[-(i+1)]
            s_next = 1 if i < len(S_grid) else S_grid[-(i+2)]
            if comp(k=k, n=n, p=p, s=s) <= c0 + c1*k*n*p:
                if s*k <= p:
                    s_max = s
                    found = True
            i += 1
        return s_max
    
    s_max_las = lambda k, n, p: find_s_max(comp=las_comp, c0=c0, c1=c1, k=k, n=n, p=p, S_grid=S_grid)
    s_max_ala = lambda k, n, p: find_s_max(comp=ala_comp, c0=c0, c1=c1, k=k, n=n, p=p, S_grid=S_grid)
    s_max_bss = lambda k, n, p: find_s_max(comp=bss_comp, c0=c0, c1=c1, k=k, n=n, p=p, S_grid=S_grid)
    s_max_lar = lambda k, n, p: find_s_max(comp=lar_comp, c0=c0, c1=c1, k=k, n=n, p=p, S_grid=S_grid)
    s_max_tls = lambda k, n, p: find_s_max(comp=tls_comp, c0=c0, c1=c1, k=k, n=n, p=p, S_grid=S_grid)

    return s_max_las, s_max_ala, s_max_bss, s_max_lar, s_max_tls

DATA_DICT = {
    'bulk_modulus': {
        'f': 'data_bulk_modulus.csv',
        'drop_cols': ['material', 'A', 'B1', 'B2'],
        'target': 'bulk_modulus (eV/AA^3)'
    },
    'bandgap': {
        'f': 'data_bandgap.csv',
        'drop_cols': ['material'],
        'target': 'bg_hse06 (eV)'
    },
    'yield': {
        'f': 'data_HTE.csv',
        'drop_cols': ['material_and_condition'],
        'target': 'Y_oxygenate'
    },
    'diabetes': {
        'f': 'diabetes.csv',
        'drop_cols': [],
        'target': 'Y'
    },
    'eye': {
        'f': 'eye.csv',
        'drop_cols': [],
        'target': 'y'
    },
    'facebook': {
        'f': 'facebook.csv',
        'drop_cols': ['Type'],
        'target': 'Total.Interactions'
    },
    'realEstate': {
        'f': 'realEstate.csv',
        'drop_cols': [],
        'target': 'Y'
    },    
    'ribo': {
        'f': 'ribo.csv',
        'drop_cols': [],
        'target': 'Y'
    },    
    'concrete': {
        'f': 'concrete.csv',
        'drop_cols': [],
        'target': 'Strength'
    },    
    'yacht': {
        'f': 'yacht.csv',
        'drop_cols': [],
        'target': 'V7'
    },   
    'conditionTur': {
        'f': 'conditionTur.csv',
        'drop_cols': [],
        'target': 'gt_t_decay'
    },   
    'airfoil': {
        'f': 'airfoil.csv',
        'drop_cols': [],
        'target': 'V6'
    },   
    'autompg': {
        'f': 'autompg.csv',
        'drop_cols': [],
        'target': 'mpg'
    }, 
   'parkinson_total': {
        'f': 'parkinson_total.csv',
        'drop_cols': [],
        'target': 'total_UPDRS'
    }, 
    'parkinson_motor': {
        'f': 'parkinson_motor.csv',
        'drop_cols': [],
        'target': 'motor_UPDRS'
    }, 
}

def exp(data_keys=DATA_DICT.keys(), indir = os.path.join(os.getcwd(), 'Input'),
        N=[0.05, 0.1, 0.25, 0.5, 0.75, 1], SO=['AdaptiveLASSO', 'LASSO', 'BSS', 'LARS', 'TLS'], k=5, RS=range(10), 
        S_grid = np.array([1,2,3,4,5,6,7,8,9,
             10,20,30,40,50,60,70,80,90,
             100,200,300,400,500,600,700,800,900,
             1000,2000,3000]),
        rung=2, ops = [('mul', range(1))],
        verbose=False,
        optimize_k=False,
        check_pos=True
        ):
    
    res_df = pd.DataFrame()
    full_start = time()
    for i, data_key in enumerate(data_keys):
        print(data_key)
        X, y, names = load_dataset(data_key=data_key, indir=indir)
        FE = FeatureExpansion(ops=ops, rung=rung)
        n0, p0 = X.shape
        Phi_names, Phi_symbols, Phi = FE.expand(X=X, names=names, verbose=verbose+1, check_pos=check_pos)
        p = len(Phi_names)

        for j, n_ in enumerate(N):
            n = int(n_*n0)
            for random_state in RS:
                X_train, X_test, y_train, y_test = sample_data(X=Phi, y=y, n=n, ret_idx=False, random_state=random_state)

                for l, so in enumerate(SO):
                    s_max = SO_DICT[so]['s_max'](k=k, n=n, p=p)
                    S = S_grid[S_grid<=s_max]
                    for m, s in enumerate(S):
                        if True: #n > 4000 and s >= 500 and so == 'LASSO' and data_key == 'bulk_modulus':
                            icl = ICL(s=s, so=clone(SO_DICT[so]['so']), k=k, fit_intercept=True, optimize_k=optimize_k)
                            try: 
                                start = time()
                                icl.fit(X=X_train, y=y_train, feature_names=Phi_names, verbose = verbose>1)
                                fit_time = time() - start
                                y_hat_train = icl.predict(X_train).ravel()
                                y_hat_test = icl.predict(X_test).ravel()
                                sse_train = np.sum((y_hat_train - y_train)**2)
                                sse_test = np.sum((y_hat_test - y_test)**2)
                                m = icl.__repr__()
                            except ValueError:
                                if verbose: print('Fit terminated due to value error bug in lars_lasso')
                                sse_train = np.infty
                                sse_test = np.infty
                                fit_time = np.infty
                                m = '/'*1000

                            tss_train = np.sum((y_train.ravel() - y_train.mean())**2)
                            tss_test = np.sum((y_test.ravel() - y_test.mean())**2)

                            row = {
                                'problem': data_key, 'n': n, 'p0': p0, 'p': p, 'random_state': random_state,
                                'k': k, 'rung': rung, 'ops': str(ops).replace(',', ';'), 'so': so, 's': s, 
                                'sse_train': sse_train, 'sse_test': sse_test, 'tss_train': tss_train, 'tss_test': tss_test, 
                                'fit_time': fit_time, 'model': m
                            }
                            res_df = res_df.append(row, ignore_index=True)
                            print('problem: {0}, n: {1}, p: {7}, random_state: {2}, so: {3}, s: {4}, fit_time: {5}, time_so_far: {6}'.format(data_key, n, random_state, so, s, np.round(fit_time, 3), np.round(time() - full_start, 3), p))
    return res_df

def fastridge_exp(data_keys=DATA_DICT.keys(), indir = os.path.join(os.getcwd(), 'Input'), 
                  RS=range(10), verbose=False):
    
    res_df = pd.DataFrame()
    full_start = time()
    for i, data_key in enumerate(data_keys):
        print(data_key)
        X, y, names = load_dataset(data_key=data_key, indir=indir)
        n, p0 = X.shape
        for random_state in RS:
            X_train, X_test, y_train, y_test = sample_data(X=X, y=y, n=n, ret_idx=False, random_state=random_state)
            model = fr(trace=True, fit_intercept=True)
            start = time()
            model.fit(X_train, y_train)
            fit_time = time() - start
            y_hat_train = model.predict(X_train).ravel()
            y_hat_test = model.predict(X_test).ravel()
            sse_train = np.sum((y_hat_train - y_train)**2)
            sse_test = np.sum((y_hat_test - y_test)**2)
            m = str(model.coef_).replace('  ', ' ').replace(' ', ';') + ';' + str(model.intercept_)
            tss_train = np.sum((y_train.ravel() - y_train.mean())**2)
            tss_test = np.sum((y_test.ravel() - y_test.mean())**2)

            row = {'problem': data_key, 'n': n, 'p0': p0, 'random_state': random_state,
                   'sse_train': sse_train, 'sse_test': sse_test, 'tss_train': tss_train, 'tss_test': tss_test, 
                   'fit_time': fit_time, 'model': m
                   }
            res_df = res_df.append(row, ignore_index=True)
            print('problem: {0}, n: {1}, random_state: {2}, fit_time: {3}, time_so_far: {4}'.format(data_key, n, random_state, np.round(fit_time, 3), np.round(time() - full_start, 3)))
    return res_df

def predict_mean_exp(data_keys=DATA_DICT.keys(), indir = os.path.join(os.getcwd(), 'Input'), 
                  RS=range(10), verbose=False):
    
    res_df = pd.DataFrame()
    full_start = time()
    for i, data_key in enumerate(data_keys):
        print(data_key)
        X, y, names = load_dataset(data_key=data_key, indir=indir)
        n, p0 = X.shape
        for random_state in RS:
            if verbose: print(data_key, random_state)
            X_train, X_test, y_train, y_test = sample_data(X=X, y=y, n=n, ret_idx=False, random_state=random_state)
            model = PredictMean()
            start = time()
            model.fit(X_train, y_train)
            fit_time = time() - start
            y_hat_train = model.predict(X_train).ravel()
            y_hat_test = model.predict(X_test).ravel()
            sse_train = np.sum((y_hat_train - y_train)**2)
            sse_test = np.sum((y_hat_test - y_test)**2)
            m = str(model.intercept_)
            tss_train = np.sum((y_train.ravel() - y_train.mean())**2)
            tss_test = np.sum((y_test.ravel() - y_test.mean())**2)

            row = {'problem': data_key, 'n': n, 'p0': p0, 'random_state': random_state,
                   'sse_train': sse_train, 'sse_test': sse_test, 'tss_train': tss_train, 'tss_test': tss_test, 
                   'fit_time': fit_time, 'model': m
                   }
            res_df = res_df.append(row, ignore_index=True)
            print('problem: {0}, n: {1}, random_state: {2}, fit_time: {3}, time_so_far: {4}'.format(data_key, n, random_state, np.round(fit_time, 3), np.round(time() - full_start, 3)))
    return res_df

def run_datasets_icl(to_csv=True, First=False, optimize_k=True, verbose=True, start=0, end=25, problem_lst=['small_p0', 'big_p0', 'huge_p0', 'med_p0']):
    all_outputs = pd.DataFrame()
    for i,problems in enumerate(problem_lst): # ['small_p0', 'big_p0', 'huge_p0', 'med_p0']
        if problems == 'med_p0':
            rung = 2
            small = ['sin', 'cos', 'log', 'abs', 'sqrt', 'cbrt', 'sq', 'cb', 'inv']
            big = ['exp', 'six_pow', 'mul', 'div', 'abs_diff', 'add']
            small  = [(op, range(rung)) for op in small]
            big = [(op, range(1)) for op in big]
            data_keys = ['yield', 'conditionTur', 'facebook', 'parkinson_total', 'parkinson_motor', 'bulk_modulus', 'bandgap'] 
        elif problems == 'small_p0': 
            rung = 2
            small = ['sin', 'cos', 'log', 'abs', 'sqrt', 'cbrt', 'sq', 'cb', 'inv',  'mul']
            big = ['exp', 'six_pow', 'abs_diff', 'add', 'div']
            small  = [(op, range(rung)) for op in small]
            big = [(op, range(1)) for op in big]
            data_keys = ['diabetes', 'concrete', 'airfoil','autompg','yacht', 'realEstate']
        elif problems == 'big_p0':
            rung = 1
            small = ['sin', 'cos', 'log', 'abs', 'sqrt', 'cbrt', 'sq', 'cb', 'inv',  'mul', 'div', 'abs_diff', 'exp', 'six_pow']
            big = []
            small  = [(op, range(rung)) for op in small]
            big = [(op, range(1)) for op in big]
            data_keys = ['eye']
        elif problems == 'huge_p0':
            rung = 0
            small = ['sin', 'cos', 'log', 'abs', 'sqrt', 'cbrt', 'sq', 'cb', 'inv',  'mul', 'div', 'abs_diff', 'add', 'exp', 'six_pow']
            big = []
            small  = [(op, range(rung)) for op in small]
            big = [(op, range(1)) for op in big]
            data_keys = ['ribo']

        ops = small+big

        for i, key in enumerate(data_keys):
            res_df = exp(data_keys=[key], N=[1], RS=range(start, end), rung=rung, ops = ops, verbose=verbose, optimize_k=optimize_k, check_pos=True)
            if to_csv:
                f = 'output_{0}_opt_k_{1}.csv'.format(key, optimize_k)
                out_dir = os.path.join(os.getcwd(), 'Output', 'Dataset_Exps', f)
                res_df.to_csv(out_dir, header=First, mode='w' if First else 'a')
            else:
                all_outputs = all_outputs.append(res_df, ignore_index=True)
    if not to_csv: return all_outputs

if __name__ == '__main__':
    out_dir = os.path.join(os.getcwd(), 'Output', 'Dataset_Exps')
    models = 'predict_mean' 
    verbose = 1
    if models=='icl':
        s_max_las, s_max_ala, s_max_bss, s_max_lar, s_max_tls = cost_lim_fn(t_max = 30, t_min = 0.1, kmax = 5, nmax = 10000, pmax = 19477)
        #s_max_las, s_max_ala, s_max_bss, s_max_lar, s_max_tls = lambda k, n, p:  1000,  lambda k, n, p:  200, lambda k, n, p:  7, lambda k, n, p: 1000, lambda k, n, p:  200 

        SO_DICT = {
            'LASSO': {
                'so': AdaptiveLASSO(gamma=0, fit_intercept=False),
                's_max': s_max_las
            },
            'AdaptiveLASSO': {
                'so': AdaptiveLASSO(gamma=1, fit_intercept=False),
                's_max': s_max_ala
            },
            'BSS': {
                'so': BSS(),
                's_max': s_max_bss
            },
            'LARS': {
                'so': LARS(),
                's_max': s_max_lar
            },
            'TLS': {
                'so': ThresholdedLeastSquares(),
                's_max': s_max_tls
            }
        }

        First = False
        optimize_k = True
        verbose=verbose
        start = 0
        end = 100
        run_datasets_icl(to_csv=True, First=False, optimize_k=True, verbose=verbose, 
                        start=start, end=end, 
                        problem_lst=['small_p0', 'big_p0', 'huge_p0', 'med_p0'])
    elif models=='fastridge':
        start = 0
        end = 100
        res_df = fastridge_exp(data_keys=DATA_DICT.keys(), indir = os.path.join(os.getcwd(), 'Input'), 
                  RS=range(start, end), verbose=verbose)
        res_df.to_csv(os.path.join(out_dir, 'fastridge_output.csv'), header=True, mode='w')
    elif models=='predict_mean':
        start = 0
        end = 100
        res_df = predict_mean_exp(data_keys=DATA_DICT.keys(), indir = os.path.join(os.getcwd(), 'Input'), 
                  RS=range(start, end), verbose=verbose)
        res_df.to_csv(os.path.join(out_dir, 'predict_mean_output.csv'), header=True, mode='w')