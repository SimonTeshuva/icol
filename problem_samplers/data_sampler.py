import pandas as pd
import numpy as np
import os

class DATA_SAMPLER:
    def __init__(self, f, random_state=None):
        self.f = f
        np.random.seed(random_state)
        self.random_state = random_state

    def __str__(self):
        return self.f
    
    def __repr__(self):
        return self.f
        
    def get_data(self, target=None, asarray=False):
        df = pd.read_csv(self.f)
        cols = df.columns
        target = cols[-1] if target is None else target
        self.target = target
        y = df[target]
        X =  df.drop(columns=target)

        if asarray:
            self.X, self.y, self.cols = X.values, y.values, cols
            return X.values, y.values
        else:
            self.X, self.y, self.cols = X, y, cols
            return X, y
        
    def sample(self, n):
        in_idx =  np.random.randint(low=0, high=len(self.X), size=n)
        out_idx = list(set(range(0, len(self.X))) - set(in_idx))
        return self.X[in_idx], self.X[out_idx], self.y[in_idx], self.y[out_idx]

if __name__ == "__main__":
    f = os.path.join(os.getcwd(), 'Input', 'single_perovskites', 'single_perovskites_bulk_modulus.csv')
    target = 'bulk_modulus (eV/AA^3)'
    random_state = 0
    n = 10
    sampler = DATA_SAMPLER(f, random_state=random_state)
    X, y = sampler.get_data(target=target, asarray=True)
    X_train, X_test, y_train, y_test = sampler.sample(n=n)

