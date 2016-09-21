
# coding: utf-8

# In[1]:

# This Notebook demonstrates what can be done with Matrix module
# also used for development playground...


# In[2]:

import pandas as pd
import scipy.stats as stats
import pylab
import matplotlib.pyplot as plt


# In[3]:

import os


# In[4]:

from AuxFunctions import *
from Matrix import *
from TLD import *
from Gravity import *


# # Matrix Examples

# In[5]:

df = pd.DataFrame(data=[range(6) for row in range(3)], columns=['A1', 'A2', 'A3', 'B1', 'B2', 'B3'])
df


# In[6]:

m = mat(7)
m


# In[7]:

mat7 = Matrix(m)
mat7


# In[8]:

exmat = pd.DataFrame({'O': [10,10,20,20],
                      'D': [10,20,10,20],
                      'T':[3,2,4,1]}).set_index(['O','D'])
exmat


# In[9]:

I3 = I(3)
I3


# In[10]:

ex_matrixp = os.path.join('example_data', 'ex_matrix_1.csv')
ex_matrix = pd.read_csv(ex_matrixp, index_col=[0,1])
ex_skimdistp = os.path.join('example_data', 'ex_skimdist_1.csv')
ex_skimdist = pd.read_csv(ex_skimdistp, index_col=[0,1])


# In[11]:

zoning1 = Zoning(list(range(3)))
zoning2 = Zoning(list(range(10)))
zoning3 = Zoning([5+i for i in range(5)])
zoning4 = Zoning('A B C'.split())


# In[12]:

mat7.complete(zoning3)


# In[13]:

mat7.matrix.T3


# In[14]:

mat7.TDp.matrix.T3


# from imp import reload
# import Matrix
# 
# reload(Matrix)
# from Matrix import *

# In[15]:

basicmapping = pd.DataFrame({
        'sectors': 'A A B B B C C'.split(),
        'zones':   [1,2,3,4,5,6,7],
        'Val1':    [1,4,4,2,1,3,1],
        'Val2':    [3,0,1,2,1,3,0]
    })

mapping = pd.DataFrame({
        'sectors': 'A B A B B B C C C'.split(),
        'zones':   [1,2,2,3,4,5,6,7,5],
        'Val1':    [1,4,4,2,1,3,1,4,2],
        'Val2':    [3,0.01,1,2,1,3,3,1,0.01]
    })


# In[16]:

mat7.rezone(basicmapping, ['zones', 'sectors'])


# In[17]:

mat7.rezone(basicmapping, ['zones', 'sectors'], mapping_split_cols=['Val1', 'Val2'])


# In[18]:

rezoned = mat7.rezone(mapping, ['zones', 'sectors'], mapping_split_cols=['Val1', 'Val2'])
rezoned


# In[19]:

rezoned.TOTALS


# In[20]:

mat7.TOTALS


# In[21]:

cost7 = mat7.apply(lambda x: abs(randomizeSeries(x, 0.2)))
cost7


# In[22]:

mat7_rezoned_weighted = mat7.rezone(mapping, ['zones', 'sectors'], ['Val1', 'Val2'], True, cost7)
mat7_rezoned_weighted


# In[23]:

DEMMANDp = os.path.join('example_data', 'Demand_EMME.txt')
DEMMAND = Matrix.read_EMME(DEMMANDp)
DEMMAND


# In[24]:

DISTp = os.path.join('example_data', 'Dist_EMME.txt')
DIST = Matrix.read_EMME(DISTp)
DIST


# In[25]:

tTO = randomizeTE(mat7.TO)
tTO


# In[26]:

tTD = randomizeTE(mat7.TD)
tTD


# In[27]:

fmat7 = mat7.furness(tTO, tTD)
fmat7


# In[28]:

max([max(x,y) for x,y in zip((fmat7.TO - tTO).abs().max(), (fmat7.TD - tTD).abs().max())])


# # TLD examples

# In[29]:

ex_matrixf = os.path.join('example_data', 'ex_matrix_1.csv')
ex_matrix = Matrix(pd.DataFrame.from_csv(ex_matrixf, index_col=[0,1]))
ex_matrix


# In[30]:

ex_skimdistf = os.path.join('example_data', 'ex_skimdist_1.csv')
ex_skimdist = Matrix(pd.DataFrame.from_csv(ex_skimdistf, index_col=[0,1]))
ex_skimdist


# In[31]:

## Fill intrazonals
ex_skimdist = ex_skimdist.complete(ex_matrix.index)


# In[32]:

ex_TLD = TLD.from_mat(ex_matrix, ex_skimdist, 5)
ex_TLD


# In[33]:

ex_TLD.sum()


# In[34]:

dst = mat7.copy()
dst['T1'] = (dst.index.get_level_values(0)**2 - dst.index.get_level_values(1)**2)**2
dst['T2'] = dst['T1'] / dst.index.get_level_values(0)
dst['T2'] = dst['T1'] / dst.index.get_level_values(1)
dst.columns = 'D1 D2 D3'.split()
dst


# In[35]:

TLD_single = TLD.from_mat_single(mat7,dst,dist_band=5)
TLD_single


# In[36]:

TLD_multi = TLD.from_mat(mat7,dst,dist_band=5)
TLD_multi


# In[37]:

mat7.sum()


# In[38]:

TLD_single.sum()


# In[39]:

TLD_multi.sum()


# In[40]:

TLD_multi.norm.sum()


# In[41]:

TLD_multi.band_agg(10).sum()


# In[42]:

OutputName = os.path.join('.', 'example_outputs', 'TLD.png')
TLD_multi.to_JPG(OutputName)


# In[43]:

oFileNamePattern = os.path.join('example_outputs', 'TLD_{}.png')
TLD_multi.cols_to_JPGs(oFileNamePattern)


# In[44]:

TLD1 = TLD_multi.copy()
TLD2 = TLD_multi.copy() + 3
TLD3 = TLD_multi.copy()
TLD3 = TLD3.apply(lambda x: x + TLD3.index.get_level_values(0))
TLDs = [TLD1, TLD2, TLD3]
i = 1
for TLD in TLDs:
    TLD.columns = ['mat{}_{}'.format(i,col) for col in TLD]
    i+=1


# In[45]:

oFileNamePattern = os.path.join('example_outputs', 'TLD_{}.png')
TLD.comparison_to_JPGs(TLDs, oFileNamePattern=oFileNamePattern)


# In[46]:

oFileNamePattern = os.path.join('example_outputs', 'ex_TLD_{}.png')
ex_TLD.cols_to_JPGs(oFileNamePattern)


# # TE comparison

# In[47]:

oFileNamePattern = os.path.join('example_outputs', '{}')
TE_comparison_to_JPGs(mat7,fmat7, oFileNamePattern=oFileNamePattern, prefixes='mat_ fmat7_'.split())


# In[48]:

TE_RegressionStats(mat7,fmat7, prefixes='mat_ fmat7_'.split())


# # Gravity

# In[49]:

TLD_multi_for_ParamEst = TLD_multi.loc[1:,:]
TLD_multi_for_ParamEst = TLD_multi_for_ParamEst.mid_band(factor=-0.5)
TLD_multi_for_ParamEst = TLD_multi_for_ParamEst.norm
TLD_multi_for_ParamEst


# In[50]:

#lognorm gamma exponnorm exponpow tanner
dists = fit_distribs(TLD_multi_for_ParamEst, 'lognorm exponnorm exponpow'.split())
dists


# In[51]:

tTO = randomizeTE(mat7.TO)
tTD = randomizeTE(mat7.TO)
tTO


# In[52]:

syn7 = mat7.ApplyGravityModel(tTO, tTD, dists, furness=False)
syn7


# In[53]:

syn7.columns.get_level_values(0)


# ##TODO: Debug
# syn7.loc[:,:] = 1
# syn7.furness(tTO, tTD)

# In[55]:

cols = [(k, d.dist.name) for k,dlst in dists.items() for d in dlst]
#cols
pd.MultiIndex.from_tuples(cols)


# In[56]:

df = pd.DataFrame(index=mat7.index, columns=pd.MultiIndex.from_tuples(cols))
#df[(['foo', 'bar'])] = 1
df


# # DEV

# In[ ]:



