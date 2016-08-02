
# coding: utf-8

# In[1]:

import pandas as pd


# In[2]:

df = pd.DataFrame(data=[range(6) for row in range(3)], columns=['A1', 'A2', 'A3', 'B1', 'B2', 'B3'])
df


# In[67]:

def mat(n):
    mat = pd.DataFrame({'O': [x+1 for x in range(n) for _ in range(n)],
                        'D': [x+1 for x in range(n)] * n,
                        'T1':[int((x%3)!=0) for x in range(n*n)],
                        'T2':[x+1 for x in range(n*n)],
                        'T3':[-(x%n)**2+n*(x%n) for x in range(n*n)]})
    mat = mat.set_index(['O', 'D'])
    #remove intrazonals:
    mat.loc[mat.index.get_level_values(0) == mat.index.get_level_values(1), 'T3'] = 0 
    return mat


# In[68]:

m = mat(7)
m


# In[5]:

def I(n):
    '''returns identity matrix of n x n'''
    matI = pd.DataFrame({'O': [x+1 for x in range(n) for _ in range(n)],
                         'D': [x+1 for x in range(n)] * n})
    matI['T'] = (matI.O == matI.D).apply(int)
    matI = matI.set_index(['O', 'D'])
    return matI


# In[6]:

I3 = I(3)
I3

