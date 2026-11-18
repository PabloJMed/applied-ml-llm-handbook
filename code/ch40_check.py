import numpy as np
from collections import Counter
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.decomposition import PCA as SKPCA
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(0)

# linear regression
Xr, yr = make_regression(n_samples=500, n_features=5, noise=10, random_state=0)
def linreg_normal_eq(X, y, l2=0.0):
    Xb = np.c_[np.ones(len(X)), X]
    I = np.eye(Xb.shape[1]); I[0, 0] = 0
    return np.linalg.solve(Xb.T @ Xb + l2 * I, Xb.T @ y)
w = linreg_normal_eq(Xr, yr)
sk = LinearRegression().fit(Xr, yr)
print(np.allclose(w[1:], sk.coef_), np.isclose(w[0], sk.intercept_))

# decision tree (CART, gini)
def gini(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return 1 - np.sum(p ** 2)

def best_split(X, y, min_leaf=5):
    best = (None, None, gini(y))
    n = len(y)
    for j in range(X.shape[1]):
        order = np.argsort(X[:, j]); xs, ys = X[order, j], y[order]
        for i in range(min_leaf, n - min_leaf):
            if xs[i] == xs[i - 1]:
                continue
            left, right = ys[:i], ys[i:]
            score = (len(left) * gini(left) + len(right) * gini(right)) / n
            if score < best[2]:
                best = (j, (xs[i] + xs[i - 1]) / 2, score)
    return best

class Node:
    def __init__(self, pred=None, feature=None, threshold=None, left=None, right=None):
        self.pred, self.feature, self.threshold, self.left, self.right = pred, feature, threshold, left, right

def build_tree(X, y, depth=0, max_depth=4, min_leaf=5):
    pred = Counter(y).most_common(1)[0][0]
    if depth >= max_depth or len(set(y)) == 1 or len(y) < 2 * min_leaf:
        return Node(pred=pred)
    j, t, _ = best_split(X, y, min_leaf)
    if j is None:
        return Node(pred=pred)
    mask = X[:, j] <= t
    return Node(feature=j, threshold=t,
                left=build_tree(X[mask], y[mask], depth + 1, max_depth, min_leaf),
                right=build_tree(X[~mask], y[~mask], depth + 1, max_depth, min_leaf))

def predict_one(node, x):
    while node.pred is None:
        node = node.left if x[node.feature] <= node.threshold else node.right
    return node.pred

Xc, yc = make_classification(n_samples=600, n_features=6, n_informative=4, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(Xc, yc, test_size=0.3, random_state=0)
tree = build_tree(Xtr, ytr, max_depth=4)
acc_mine = np.mean([predict_one(tree, x) == t for x, t in zip(Xte, yte)])
acc_sk = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=0).fit(Xtr, ytr).score(Xte, yte)
print("tree", round(acc_mine, 3), round(acc_sk, 3))

# random forest via bagging sklearn trees
class SimpleForest:
    def __init__(self, n_trees=100, max_features="sqrt", seed=0):
        self.n_trees, self.max_features, self.seed = n_trees, max_features, seed
    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        self.trees = []
        for i in range(self.n_trees):
            idx = rng.integers(0, len(X), len(X))
            t = DecisionTreeClassifier(max_features=self.max_features, random_state=i).fit(X[idx], y[idx])
            self.trees.append(t)
        return self
    def predict_proba(self, X):
        return np.mean([t.predict_proba(X)[:, 1] for t in self.trees], axis=0)
rf = SimpleForest().fit(Xtr, ytr)
print("forest", round(np.mean((rf.predict_proba(Xte) > 0.5) == yte), 3),
      round(RandomForestClassifier(100, random_state=0).fit(Xtr, ytr).score(Xte, yte), 3))

# gradient boosting from scratch (squared loss)
class SimpleGBM:
    def __init__(self, n_trees=200, lr=0.1, max_depth=3):
        self.n_trees, self.lr, self.max_depth = n_trees, lr, max_depth
    def fit(self, X, y):
        self.base = y.mean(); pred = np.full(len(y), self.base); self.trees = []
        for _ in range(self.n_trees):
            residual = y - pred
            t = DecisionTreeRegressor(max_depth=self.max_depth).fit(X, residual)
            pred += self.lr * t.predict(X); self.trees.append(t)
        return self
    def predict(self, X):
        return self.base + self.lr * np.sum([t.predict(X) for t in self.trees], axis=0)
Xtr2, Xte2, ytr2, yte2 = train_test_split(Xr, yr, test_size=0.3, random_state=0)
g = SimpleGBM().fit(Xtr2, ytr2)
skg = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=0).fit(Xtr2, ytr2)
print("gbm rmse", round(np.sqrt(np.mean((g.predict(Xte2) - yte2) ** 2)), 2), round(np.sqrt(np.mean((skg.predict(Xte2) - yte2) ** 2)), 2))

# kNN
def knn_predict(Xtrain, ytrain, Xq, k=5):
    d = ((Xq[:, None, :] - Xtrain[None]) ** 2).sum(-1)
    nn = np.argsort(d, axis=1)[:, :k]
    return np.array([Counter(ytrain[row]).most_common(1)[0][0] for row in nn])
mu, sd = Xtr.mean(0), Xtr.std(0)
print("knn", round(np.mean(knn_predict((Xtr - mu) / sd, ytr, (Xte - mu) / sd) == yte), 3))

# linear SVM with hinge loss SGD (Pegasos style)
def linear_svm(X, y, lam=0.01, epochs=20, seed=0):
    rng = np.random.default_rng(seed)
    y_pm = np.where(y == 1, 1, -1); w = np.zeros(X.shape[1]); b = 0.0; t = 0
    for _ in range(epochs):
        for i in rng.permutation(len(y)):
            t += 1; eta = 1 / (lam * t)
            if y_pm[i] * (X[i] @ w + b) < 1:
                w = (1 - eta * lam) * w + eta * y_pm[i] * X[i]; b += eta * y_pm[i] * 0.01
            else:
                w = (1 - eta * lam) * w
    return w, b
w, b = linear_svm((Xtr - mu) / sd, ytr)
print("svm", round(np.mean((((Xte - mu) / sd) @ w + b > 0) == yte), 3))

# multinomial naive Bayes
class MultinomialNaiveBayes:
    def fit(self, X, y, alpha=1.0):
        self.classes = np.unique(y)
        self.log_prior = np.log(np.array([np.mean(y == c) for c in self.classes]))
        counts = np.array([X[y == c].sum(0) for c in self.classes]) + alpha
        self.log_lik = np.log(counts / counts.sum(1, keepdims=True))
        return self
    def predict(self, X):
        return self.classes[np.argmax(X @ self.log_lik.T + self.log_prior, axis=1)]
Xn = rng.poisson(1.0, size=(300, 20)); yn = (Xn[:, :5].sum(1) > Xn[:, 5:10].sum(1)).astype(int)
print("nb", np.mean(MultinomialNaiveBayes().fit(Xn[:200], yn[:200]).predict(Xn[200:]) == MultinomialNB().fit(Xn[:200], yn[:200]).predict(Xn[200:])))

# k-means
def kmeans(X, k, iters=50, seed=0):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        labels = np.argmin(((X[:, None] - C[None]) ** 2).sum(-1), axis=1)
        newC = np.array([X[labels == j].mean(0) if np.any(labels == j) else C[j] for j in range(k)])
        if np.allclose(newC, C):
            break
        C = newC
    return C, labels
C, labels = kmeans(Xc, 3); print("kmeans sizes", np.bincount(labels))

# PCA via SVD
def pca(X, k):
    Xc_ = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc_, full_matrices=False)
    explained = S ** 2 / np.sum(S ** 2)
    return Xc_ @ Vt[:k].T, Vt[:k], explained[:k]
Z, comps, ev = pca(Xc, 2)
skp = SKPCA(2).fit(Xc)
print("pca", np.allclose(np.abs(comps), np.abs(skp.components_)), np.round(ev, 3), np.round(skp.explained_variance_ratio_, 3))
