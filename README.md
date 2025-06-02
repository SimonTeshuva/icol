# icol
** Iterative Correlation Learning in Python **

`icol` allows one to fit extremly sparse linear models from very high dimensional datasets in a computationally efficient manner. Given a feature transformation, it can also be used to fit Symbolic Regression models

---

## Installation 

Install via pip:

``` bash
pip3 install icol
```


## Example
```python

import numpy as np

from icol.icol import PolynomialFeaturesICL, AdaptiveLASSO, ICL, rmse

# Example Data: 10 features, three of which are used to form a degree 3 polynomial with 4 terms. 
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

X_train_transformed = FE.fit_transform(X_train, y)
feature_names = FE.get_feature_names_out()

icl = ICL(s=s, so=so, d=d, fit_intercept=True, normalize=True, pool_reset=False)
icl.fit(X_train_transformed, y_train, feature_names=feature_names, verbose=False)

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

```