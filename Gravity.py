
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
from scipy import stats
from scipy import optimize
import matplotlib.pyplot as plt
#import warnings


# In[2]:

from Matrix import *
from TLD import *
from AuxFunctions import *


# In[3]:

import os


# In[4]:

dist_cont = [d for d in dir(stats) if
             isinstance(getattr(stats,d), stats.rv_continuous) and
             hasattr(getattr(stats,d), 'fit')]


# In[5]:

#log of distributions not working properly
dist_cont_not_working = ['levy_stable']


# In[6]:

dist_cont_working = list(set(dist_cont) - set(dist_cont_not_working))
#dist_continu_working


# In[7]:

#TODO: CHECK is Tanner area == !?
def Tanner_func(x, alpha, beta):
    return (x ** beta) * np.exp(alpha * x)


# In[8]:

#TEMPORARY: fit method does not work as expected.
class Tanner(stats.rv_continuous):
    def _pdf(self, x, alpha, beta):
        return (x ** beta) * np.exp(alpha * x)
    def _stats(self, alpha, beta):
        #starting values for fit
        return 0., 0., 0., 0.


# In[9]:

tanner = Tanner(name='tanner', shapes='alpha, beta')


# In[32]:

#tanner.fit(Ynorm) #TMP don't re-run this cell. it's for demonstration only.
(0.00064354474335964512,
 4.1048820704712746e-08,
 -0.00019229306301766426,
 0.75015919468560166)


# In[10]:

class Tanner(stats.rv_continuous):
    def _pdf(self, x, alpha, beta):
        return (x ** beta) * np.exp(alpha * x)
    def _stats(self, alpha, beta):
        #starting values for fit
        return 0., 0., 0., 0.
    #TODO: Review. It would be ideal if Tanner could use built in fit method,
    #      but this implementation works similarly. Wath ydata.index.values 
    def fit(self, ydata, p0=[0.15,-0.9], *args, **kwargs):
        return optimize.curve_fit(self._pdf, ydata.index.values, ydata, *args, **kwargs)[0]


# In[11]:

tanner = Tanner(name='tanner', shapes='alpha, beta')
stats.tanner = tanner #as if it had always been in scipy.stats


# In[16]:

#TMP don't re-run this cell. it's for demonstration only.
#tanner.fit(Ynorm, p0=[0.15,-0.9])
np.array([-3.39581168,  6.73004206])


# In[17]:

#TMP don't re-run this cell. it's for demonstration only.
#optimize.curve_fit(Tanner_func, X, Ynorm, p0=[0.15,-0.9])
(np.array([-3.39579654,  6.72999835]), np.array([[ 0.16630225, -0.47650378],
        [-0.47650378,  1.38303769]]))


# In[12]:

#src: http://stackoverflow.com/questions/6620471/fitting-empirical-distribution-to-theoretical-ones-with-scipy-python
def fit_distribs(TLD, distrib_names, *args, **kwargs):
    '''Returns a dictionary of distributions fitted for each column in TLD.
    Drops NaN values.'''
    distribs = {}
    for col in TLD:
        distribs[col] = [getattr(stats, fn)(
                            *getattr(stats, fn).fit(TLD[col].dropna(), *args, **kwargs))
                         for fn in distrib_names]
    return distribs


# In[13]:

def fit_distribs_to_df(distrib_dict: dict, xdata: pd.Series) -> pd.DataFrame:
    '''Returns a dataframe with all the distributions in distrib_dict applied to xdata.'''
    df = pd.DataFrame.from_dict(
    {(k, d.dist.name): d.pdf(xdata) for k,dlst in distrib_dict.items() for d in dlst})
    df.index = xdata
    return df


# In[14]:

def append_distribs_to_df(TLD, distrib_names, level=0, include_ydata=True,
                          *args, **kwargs):
    '''Returns TLD expanded with distrib_names fitted for each column.
    xdata is taken from TLD index level as specified by level.'''
    TLDdistribs = fit_distribs(TLD, distrib_names)
    df = fit_distribs_to_df(TLDdistribs, TLD.index.get_level_values(level))
    if include_ydata:
        for col in TLD:
            df.loc[:,(col,'ydata')] = TLD[col]
        df.sort_index(axis=1, level=0, inplace=True)
    return df


# In[12]:

#TODO
def n_best_fitting_distrib(data, distrib_names, n, *args, **kwargs):
    '''returns the best n fitting distributions of the specified
    list of distirbutions.'''
    ...


# In[13]:

def lognorm_mu_sigma(params):
    '''Returns the lognorm parameters (mu, sigma) from params
    estimated from scipy.stats.lognorm.fit(obs_data)'''
    shape, loc, scale = params
    mu, sigma = np.log(scale), shape
    return mu, sigma


# In[92]:

def ApplyGravityModel(c: Matrix,
                      TO: pd.DataFrame,
                      TD: pd.DataFrame,
                      f: stats.rv_continuous,
                      furness=True,
                      *args, **kwargs) -> Matrix:
    '''Returns a matrix Tij = Oi*Dj*f(cij)
    c        - cost matrix
    TO       - trip origins
    TD       - trip destinations
    f        - deterrence function (object)
    furness  - return furnessed matrix with TO, TD
    *args, **kwargs - parameters to pass to furness method
    c, TO and TD must have the same number of columns and the same column names'''
    
    same_cols = all([c1==c2==c3 for c1,c2,c3 in zip(c.columns, TO.columns, TD.columns)])
    if not same_cols:
        raise ValueError('c, TO and TD must have the same number of columns and the same column names')
    
    gravity = c.apply(f.pdf)
    synthetic = gravity.mul(TO, axis=1, level=0).mul(TD, axis=1, level=1)

    if furness:
        return synthetic.furness(TO,TD, *args, **kwargs)
    else:
        return synthetic
