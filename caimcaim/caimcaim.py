"""
CAIM
=====

# CAIM (class-attribute interdependence maximization) algorithm for
        supervised discretization

.. note::
    "L. A. Kurgan and K. J. Cios (2004), CAIM discretization algorithm in
    IEEE Transactions on Knowledge and Data Engineering, vol. 16, no. 2, pp. 145-153, Feb. 2004.
    doi: 10.1109/TKDE.2004.1269594"
    .. _a link: http://ieeexplore.ieee.org/document/1269594/

.. module:: caimcaim
   :platform: Unix, Windows
   :synopsis: A simple, but effective discretization algorithm

"""


import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CAIMD(BaseEstimator, TransformerMixin):

    def __init__(self, categorical_features='auto'):
        """
        CAIM discretization class

        Parameters
        ----------
        categorical_features : 'auto' or 'all' or list/array of indices or list of labels
        Specify what features are treated as categorical.
        - 'auto' (default): Only those features whose number of unique values exceeds the number of classes
                            of the target variable by 2 times or more
        - array of indices: array of categorical feature indices
        - list of labels: column labels of a pandas dataframe

        Example
        ---------
        >>> from caimcaim import CAIMD
        >>> from sklearn.datasets import load_iris
        >>> iris = load_iris()
        >>> X = iris.data
        >>> y = iris.target

        >>> caim = CAIMD()
        >>> x_disc = caim.fit_transform(X, y)
        """

        if isinstance(categorical_features, str):
            self._features = categorical_features
            self.categorical = None
        elif (isinstance(categorical_features, list)) or (isinstance(categorical_features, np.ndarray)):
            self._features = None
            self.categorical = categorical_features
        else:
            raise CategoricalParamException(
                "Wrong type for 'categorical_features'. Expected 'auto', an array of indicies or labels.")

    def fit(self, X, y):
        """
        Fit CAIM
        Parameters
        ----------
        X : array-like, pandas dataframe, shape [n_samples, n_feature]
            Input array can contain missing values
        y:  array-like, pandas dataframe, shape [n_samples]
            Target variable. Must be categorical.
        Returns
        -------
        self
        """

        self.split_scheme = dict()
        if isinstance(X, pd.DataFrame):
            # self.indx = X.index
            # self.columns = X.columns
            if isinstance(self._features, list):
                self.categorical = [X.columns.get_loc(label) for label in self._features]
            X = X.values
            y = y.values
        if self._features == 'auto':
            self.categorical = self.check_categorical(X, y)
        categorical = self.categorical
        print('Categorical', categorical)

        min_splits = np.unique(y).shape[0]

        for j in range(X.shape[1]):
            if j in categorical:
                continue
            xj = X[:, j]
            xj = xj[np.invert(np.isnan(xj))]
            new_index = xj.argsort()
            xj = xj[new_index]
            yj = y[new_index]
            allsplits = np.unique(xj)[1:-1].tolist()  # potential split points
            global_caim = -1
            mainscheme = [xj[0], xj[-1]]
            best_caim = 0
            k = 1
            while (k <= min_splits) or ((global_caim < best_caim) and (allsplits)):
                split_points = np.random.permutation(allsplits).tolist()
                best_scheme = None
                best_point = None
                best_caim = 0
                k = k + 1
                while split_points:
                    scheme = mainscheme[:]
                    sp = split_points.pop()
                    scheme.append(sp)
                    scheme.sort()
                    c = self.get_caim(scheme, xj, yj)
                    if c > best_caim:
                        best_caim = c
                        best_scheme = scheme
                        best_point = sp
                if (k <= min_splits) or (best_caim > global_caim):
                    mainscheme = best_scheme
                    global_caim = best_caim
                    try:
                        allsplits.remove(best_point)
                    except ValueError:
                        raise NotEnoughPoints('The feature #' + str(j) + ' does not have' +
                                              ' enough unique values for discretization!' +
                                              ' Add it to categorical list!')

            self.split_scheme[j] = mainscheme
            print('#', j, ' GLOBAL CAIM ', global_caim)
        return self

    def transform(self, X):
        """
        Discretize X using a split scheme obtained with CAIM.
        Parameters
        ----------
        X : array-like or pandas dataframe, shape [n_samples, n_features]
            Input array can contain missing values
        Returns
        -------
        X_di : sparse matrix if sparse=True else a 2-d array, dtype=int
            Transformed input.
        """

        if isinstance(X, pd.DataFrame):
            self.indx = X.index
            self.columns = X.columns
            X = X.values
        X_di = X.copy()
        categorical = self.categorical

        scheme = self.split_scheme
        for j in range(X.shape[1]):
            if j in categorical:
                continue
            sh = scheme[j]
            sh[-1] = sh[-1] + 1
            xj = X[:, j]
            # xi = xi[np.invert(np.isnan(xi))]
            for i in range(len(sh) - 1):
                ind = np.where((xj >= sh[i]) & (xj < sh[i + 1]))[0]
                X_di[ind, j] = i
        if hasattr(self, 'indx'):
            return pd.DataFrame(X_di, index=self.indx, columns=self.columns)
        return X_di

    def fit_transform(self, X, y):
        """
        Fit CAIM to X,y, then discretize X.
        Equivalent to self.fit(X).transform(X)
        """
        self.fit(X, y)
        return self.transform(X)

    def get_caim(self, scheme, xi, y):
        sp = self.index_from_scheme(scheme[1:-1], xi)
        sp.insert(0, 0)
        sp.append(xi.shape[0])
        n = len(sp) - 1
        isum = 0
        for j in range(n):
            init = sp[j]
            fin = sp[j + 1]
            Mr = xi[init:fin].shape[0]
            val, counts = np.unique(y[init:fin], return_counts=True)
            maxr = counts.max()
            isum = isum + (maxr / Mr) * maxr
        return isum / n

    def index_from_scheme(self, scheme, x_sorted):
        split_points = []
        for p in scheme:
            split_points.append(np.where(x_sorted > p)[0][0])
        return split_points

    def check_categorical(self, X, y):
        categorical = []
        ny2 = 2 * np.unique(y).shape[0]
        for j in range(X.shape[1]):
            xj = X[:, j]
            xj = xj[np.invert(np.isnan(xj))]
            if np.unique(xj).shape[0] < ny2:
                categorical.append(j)
        return categorical


class CategoricalParamException(Exception):
    # Raise if wrong type of parameter
    pass


class NotEnoughPoints(Exception):
    # Raise if a feature must be categorical, not continuous
    pass
