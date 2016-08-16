
# coding: utf-8

# In[1]:

# This Notebook demonstrates what can be done with Matrix module


# In[2]:

import pandas as pd


# In[3]:

from AuxFunctions import *
from Matrix import *
from TLD import *


# In[4]:

df = pd.DataFrame(data=[range(6) for row in range(3)], columns=['A1', 'A2', 'A3', 'B1', 'B2', 'B3'])
df


# In[5]:

m = mat(7)
m


# In[6]:

exmat = pd.DataFrame({'O': [10,10,20,20],
                      'D': [10,20,10,20],
                      'T':[3,2,4,1]}).set_index(['O','D'])
exmat


# In[7]:

I3 = I(3)
I3


# In[7]:

ex_matrix = pd.read_csv('example_data\ex_matrix_1.csv', index_col=[0,1])
ex_skimdist = pd.read_csv('example_data\ex_skimdist_1.csv', index_col=[0,1])


# In[8]:

mat = Matrix(m)
mat


# In[10]:

zoning1 = zoning(list(range(3)))
zoning2 = zoning(list(range(10)))
zoning3 = zoning([5+i for i in range(5)])
zoning4 = zoning('A B C'.split())


# In[11]:

mat.complete(zoning2)


# In[12]:

mat.matrix.T3


# In[13]:

mat.TDp.matrix.T3


# In[14]:

basicmapping = pd.DataFrame({
        'sectors': 'A A B B B C C'.split(),
        'zones':   [1,2,3,4,5,6,7,]
    })

mapping = pd.DataFrame({
        'sectors': 'A B A B B B C C C'.split(),
        'zones':   [1,2,2,3,4,5,6,7,5],
        'Val1':    [1,4,4,2,1,3,1,4,2],
        'Val2':    [3,0.01,1,2,1,3,3,1,0.01]
    })


# In[15]:

mat.rezone(basicmapping, ['zones', 'sectors'])


# In[16]:

rezoned = mat.rezone(mapping, ['zones', 'sectors'], weight_cols=['Val1', 'Val2'])
rezoned


# In[17]:

rezoned.TOTALS


# In[18]:

mat.TOTALS


# In[38]:

ex_matrix = Matrix(pd.DataFrame.from_csv('example_data\ex_matrix_1.csv', index_col=[0,1]))
ex_matrix


# In[37]:

ex_skimdist = Matrix(pd.DataFrame.from_csv('example_data\ex_skimdist_1.csv', index_col=[0,1]))
ex_skimdist


# In[25]:

DEMMAND = Matrix.read_EMME('example_data\Demand_EMME.txt')
DEMMAND


# In[26]:

DIST = Matrix.read_EMME('example_data\Dist_EMME.txt')
DIST


# In[27]:

tTO = randomizeTE(mat.TO)
tTO


# In[28]:

tTD = randomizeTE(mat.TD)
tTD


# In[29]:

fmat = mat.furness(tTO, tTD)
fmat


# In[31]:

max([max(x,y) for x,y in zip((fmat.TO - tTO).abs().max(), (fmat.TD - tTD).abs().max())])


# In[40]:

ex_TLD = TLD(ex_matrix, ex_skimdist, 'meters',1)


# In[42]:

ex_TLD.sum()


# In[43]:

ex_matrix.sum()

