CAIM (class-attribute interdependence maximization) method for supervised discretization
=====

# A Python implementation of CAIM algorithm for datasets with missing values

*"L. A. Kurgan and K. J. Cios (2004), CAIM discretization algorithm, in IEEE Transactions on Knowledge and Data Engineering, vol. 16, no. 2, pp. 145-153, Feb. 2004. doi: 10.1109/TKDE.2004.1269594"*
[http://ieeexplore.ieee.org/document/1269594](http://ieeexplore.ieee.org/document/1269594)

Installation
------------

Install the following requirements:

 * [NumPy](http://numpy.org/)
 * [scikit-learn](scikit-learn.org)


```
pip3 install caimcaim
```

Example of usage
-----

```python
>>> from caimcaim import CAIMD
>>> from sklearn.datasets import load_iris
>>> iris = load_iris()
>>> X = iris.data
>>> y = iris.target

>>> caim = CAIMD()
>>> x_disc = caim.fit_transform(X, y)
```
