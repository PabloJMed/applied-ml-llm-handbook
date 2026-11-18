import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
Xc, yc = make_classification(n_samples=600, n_features=6, n_informative=4, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(Xc, yc, test_size=0.3, random_state=0)
mu, sd = Xtr.mean(0), Xtr.std(0)
def linear_svm(X, y, lam=0.01, epochs=20, seed=0):
    """Pegasos: SGD on the hinge loss with L2 regularisation. y in {0,1}."""
    rng = np.random.default_rng(seed)
    X = np.c_[X, np.ones(len(X))]                 # bias as an extra feature
    y_pm = np.where(y == 1, 1.0, -1.0)
    w, t = np.zeros(X.shape[1]), 0
    for _ in range(epochs):
        for i in rng.permutation(len(y)):
            t += 1
            eta = 1.0 / (lam * t)
            margin = y_pm[i] * (X[i] @ w)
            w *= (1 - eta * lam)                  # shrink: the regulariser
            if margin < 1:                        # inside the margin or wrong: hinge gradient
                w += eta * y_pm[i] * X[i]
    return w
def svm_predict(w, X):
    return (np.c_[X, np.ones(len(X))] @ w > 0).astype(int)
w = linear_svm((Xtr - mu) / sd, ytr)
print(np.mean(svm_predict(w, (Xte - mu) / sd) == yte), LinearSVC().fit((Xtr-mu)/sd, ytr).score((Xte-mu)/sd, yte))
