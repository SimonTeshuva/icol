import numpy as np

import pandas as pd

from math import sqrt
from scipy.special import comb
from copy import deepcopy, copy
from scipy.optimize import minimize
from scipy.stats import multinomial 
from IPython.display import Markdown
from sklearn.preprocessing import PolynomialFeatures

from icol.icol import *

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


def truncated_multinomial_rv(n, probs, bounds, random_state=None):
    RNG = np.random.default_rng(random_state)
    # truncated_multinomial_rv(num_terms, alpha, b, random_state=RNG)
    while True:
        counts = multinomial.rvs(n, probs, random_state=RNG)
        if (counts <= bounds).sum()==len(counts):
            return counts


def enum_comb_w_replacement(n, k):
    """Show how many combinations
    n: number of variables
    k: a specific degree
    """
    def rightmost_change_incr_index(a):
        n = len(a)
        for i in range(n-2, -1, -1):
            if a[i] > 0:
                return i+1

    def rest_from_index(a, i):
        s = sum(a[i:])
        a[i] = s
        for j in range(i+1, len(a)): a[j]=0

    def next(a):
        res = copy(a)
        i = rightmost_change_incr_index(res)
        res[i-1] = res[i-1]-1
        res[i] = res[i]+1
        rest_from_index(res, i)
        return res

    current = [k]
    current.extend((n-1)*[0])
    res = [current]

    last = (n-1)*[0]
    last.append(k)
    while current!=last:
        current = next(current)
        res.append(current)
    return res

def is_pure(monomial):
    non_zero = 0 
    for i in range(len(monomial)):
        non_zero += (monomial[i] > 0)
    return non_zero == 1

def sample_polynomial(n_vars, num_terms, degree, alpha=None, beta=0.5, random_state=None):
    RNG = np.random.default_rng(random_state)
    alpha = np.ones(degree)/degree if alpha is None else alpha
    while True:
        b = comb(n_vars, np.arange(1, degree+1), repetition=True)
        c = truncated_multinomial_rv(num_terms, alpha, b, random_state=RNG)
        # print(c)
        res = set()
        for i in range(degree):
            order_i = set()
            # if c[i] <= b[i]/2:
            while len(order_i) < c[i]:
                if RNG.random()<=beta:
                    if beta==1 and i==0: # fix the bugs if we only need interaction terms
                        break
                    monomial = tuple(multinomial.rvs(i+1, np.ones(n_vars)/n_vars, random_state=RNG))
                    if not is_pure(monomial): order_i.add(monomial)
                else:
                    var_idx = RNG.choice(n_vars)
                    monomial = [0]*n_vars
                    monomial[var_idx] = i+1
                    order_i.add(tuple(monomial))
            res = res.union(order_i)
        res = np.array(list(res))
        cnt=0
        for i in range(res.shape[1]):
            if len(np.unique(res[:, i])) != 1:
                cnt += 1
        if cnt == res.shape[1]:
            return res

class dataGenerator:

    def __init__(self, n, p, r, num_terms, degree, alpha=None, beta=0.5, random_state=None, rho=0.8, SNR=1, model='binomial') -> None:
        self.n=n
        self.p=p
        self.r=r
        self.num_terms=num_terms
        self.degree=degree
        self.alpha=alpha
        self.beta=beta
        self.random_state=random_state
        self.rho=rho
        self.SNR=SNR
        self.model=model

    def generateX(self):
        """Generate original feature space X
        """
        if not(self.random_state == None):
            np.random.seed(self.random_state)
        # Generate the design matrix
        rho_matrix = np.zeros((self.p, self.p))
        for i in range(self.p):
            for j in range(self.p):
                rho_matrix[i,j] = self.rho ** abs(i-j)
        X = np.random.multivariate_normal(np.zeros(self.p), rho_matrix, self.n)
        I = np.random.permutation(self.p)
        X = X[:, I]# permut
        df_X = pd.DataFrame(dict(zip(['X' + str(i) for i in range(self.p)], X.T)))
        return df_X

    def generateBtrue(self, k):
        """Generate 'btrue'
        """
        if not(self.random_state == None):
            np.random.seed(self.random_state)
        btrue = np.random.normal(0, 1, k)
        return btrue

    def testX(self):
        """Generate normalized feature space X'
        """
        variable_lst = sample_polynomial(self.r, self.num_terms, self.degree, 
                                        self.alpha, self.beta, random_state=self.random_state)
        df_X = self.generateX()
        test_X = []
        # generate b_i
        btrue = self.generateBtrue(self.num_terms)
        formula = []
        for j in range(len(variable_lst)):
            variable = variable_lst[j]
            X_i = np.zeros(self.n)
            X_i[:] = 1
            term = ""
            for i, order in enumerate(variable):
                X_i *= deepcopy(df_X.iloc[:, i].values) ** order
                if order:
                    if order ==1:
                        term+='X_' + str(i)
                    else:
                        term+='X_' + str(i) + "^" + str(order)
            formula.append(term)
            btrue[j] = btrue[j]/sqrt(np.var(X_i)) # normalized
            test_X.append(X_i)
        return df_X, np.array(test_X), btrue, variable_lst, '+'.join(formula)

    def glmpredict(self, X, btrue, b0, model):
        n, p = X.shape
        eta = np.dot(X, btrue) + b0
        if model == 'binomial': ## currently only working on "binomial"
            mu = 1/(1+ np.exp(-eta))
            v = mu * (1 - mu)
            y = np.array([each[0] for each in np.random.rand(n, 1)]) < mu
            y = np.array(list(map(int, y))) # converting to binomial!!!
            snr = np.sqrt(np.var(mu)/np.mean(v))
        else:
            # y = np.array([each[0] for each in np.random.rand(n, 1)]) < mu
            pass

        return snr, y, mu, v
    
    def optimize_func(self, x, X, btrue, b0, model, SNR):
        """Optimize function
        """
        return ((self.glmpredict(X, np.exp(x)*btrue, b0, model)[0] - SNR)**2).mean()

    def sample(self):
        while True:
            df_X, X, btrue, variable_lst, formula = self.testX()
            b0=0
            if self.model not in ['binomial', 'multinomial']:
                raise Exception("Only accept binomial and multinomial")
            try:
                g = minimize(self.optimize_func, x0=0, args=(X.T, btrue, b0, self.model, self.SNR))
            except:
                continue
            try:
                g = minimize(self.optimize_func, x0=g.x, args=(X.T, btrue, b0, self.model, self.SNR))
            except:
                continue
            try:
                g = minimize(self.optimize_func, x0=g.x, args=(X.T, btrue, b0, self.model, self.SNR))
            except:
                continue
            try:
                g = minimize(self.optimize_func, x0=g.x, args=(X.T, btrue, b0, self.model, self.SNR))
            except:
                continue
            break
        btrue = np.exp(g.x)*btrue
        snr, y, mu, v = self.glmpredict(X.T, btrue, b0, self.model)
        # print(snr)
        return df_X, y, variable_lst, btrue, Markdown("$$" + formula + '$$')
    
def parse_formula_polynomial_generator(formula):
    formula = formula.data[2:-2]
    terms = formula.split('+')
    parsed_formula = []
    for term in terms:
        trimmed_term = term.replace('X', '').split('_')[1:]
        parsed_term = []
        for x_i in trimmed_term:
            X = [int(i) for i in x_i.split('^')]
            parsed_term.append(X)
        parsed_formula.append(parsed_term)
    return parsed_formula

def parse_formula_model_polynomial(formula):
    formula_parsed = []
    for term in formula:
        x = term.split(' ')
        monomial = []
        for i in range(len(x)):
            x_i = x[i].split('^')
            x_i[0] = x_i[0][1:]
            x_i = [int(x_i[j]) for j in range(len(x_i))]
            x_i = x_i
            monomial.append(x_i)
        monomial = monomial
        formula_parsed.append(monomial)
    formula_parsed = formula_parsed
    return formula_parsed 

class function_sampler:
    def __init__(self, p, r, num_terms, degree, alpha=None, beta=0.5, rho=0.8, SNR=1, problem_name='', random_state=None):
        self.p = p
        self.r = r
        self.num_terms = num_terms
        self.degree = degree
        self.rho = rho
        self.alpha = alpha
        self.beta = beta
        self.SNR=SNR
        self.r_prime = None
        self.rho_matrix = None
        self.f = None
        self.coef = None
        self.stds = None
        self.random_state = random_state
        self.problem_name=problem_name
        np.random.seed(self.random_state)

    def get_params(self, deep=False):
        pass

    def __str__(self):
        return self.formula.data

    def __repr__(self):
        return self.formula.data
        
    def generate_r_prime(self):
        poly_exp = PolynomialFeatures(degree=self.degree)
        x = np.ones((1, self.p))
        poly_exp.fit(x)
        monomials = poly_exp.get_feature_names_out()
        r_prime = len(monomials)
        self.r_prime = r_prime

    def genearte_rho_matrix(self):
        rho_matrix = np.zeros((self.p, self.p))
        for i in range(self.p):
            for j in range(self.p):
                rho_matrix[i,j] = self.rho ** abs(i-j)
        self.rho_matrix = rho_matrix

    def generate_formula(self):
        df_X, y, variable_lst, btrue, formula = dataGenerator(n=10000, p=self.p, SNR=self.SNR, r=self.r, rho=self.rho, alpha=self.alpha, beta=self.beta, num_terms=self.num_terms, degree=self.degree, random_state=self.random_state).sample()
        self.variable_lst = variable_lst
        X = df_X.values #10000
        self.coef = btrue
        self.f = parse_formula_polynomial_generator(formula)
        self.formula = formula
        self.stds = []
        for variable in variable_lst:
            x = np.ones(len(X))
            for i, p in enumerate(variable):
                x*=X[:, i]**p
            self.stds += [x.std()]

    def sample(self, n, random_state=None):
        np.random.seed(random_state)
        X = np.random.multivariate_normal(mean = np.zeros(self.p), cov=self.rho_matrix, size=n)
        y = np.zeros(n)
        for i, monomial in enumerate(self.f):
            m = self.coef[i]*np.ones(n)
            for j, x in enumerate(monomial):
                m*= X[:, x[0]] if len(x) == 1 else X[:, x[0]]**x[1]
            y += m            
        return X, y
    
    def generate_sampler_idxs(self):
        X = np.random.random(size=(1, self.p))
        expander = PolynomialFeaturesICL(rung=self.degree, include_bias=False)
        Phi = expander.fit_transform(X)
        all_monomials = expander.get_feature_names_out()
        for i, name in enumerate(all_monomials):
            all_monomials[i] = name.replace('x', 'X_').replace(' ', '')
        monomials_used = self.formula.data[2:-2].split('+')
        monomials_used_idxs = []
        for i, monomial in enumerate(all_monomials):
            if monomial in monomials_used:
                monomials_used_idxs += [i]
        self.monomials_used_idxs = monomials_used_idxs
        return self.monomials_used_idxs

    
def monomial_std_dict(theta, X):
    Theta = theta.fit_transform(X)
    monomials = theta.fit(X).get_feature_names_out()

    std = Theta.std(axis=0)

    monomial_std_dict = dict()

    for i in range(len(monomials)):
        trimmed_monomial = str(monomials[i])
        trimmed_monomial = trimmed_monomial.replace('x', '').split(' ')
        if monomials[i] == '1':
            parsed_monomial = [None]
        else:
            parsed_monomial = []
            for x_i in trimmed_monomial:
                if '^' not in x_i:
                    v, p = x_i, 1
                    parsed_monomial.append([v])
                else:
                    v, p = x_i.split('^')
                    parsed_monomial.append([v, p])
        monomial_std_dict[str(parsed_monomial)] = std[i]
        
    return monomial_std_dict


if __name__ == "__main__":
    pass
    sp = {'p': 10, 'r': 5, 'num_terms': 5, 'degree':  5}
    s = function_sampler(**sp)
    s.genearte_rho_matrix()
    s.generate_formula()
    s.generate_r_prime()
    X, y = s.sample(3)


