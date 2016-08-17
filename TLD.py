
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os


# In[2]:

import glob


# In[64]:

from Matrix import Matrix
from AuxFunctions import zip_df_cols, trim_index_df, duplicates_in_list


# In[4]:

def nband(n):
    '''
    Returns a function that returns the corresponding n_band for x
    nband is a wrapper that parametrises n for a banding function
    Useful for pandas.groupby(nband(n))
    
    >>> [nband(2)(x) for x in range(7)]
    [0, 0, 2, 2, 4, 4, 6]
    '''
    return lambda x: int(x/n)*n


# In[5]:

def band_agg_TLD(TLD, n):
    '''Takes a TLD dataframe (distances as index, one column per TLD),
    and returns a TLD dataframe aggregated to bands of n.'''
    TLDband = TLD.index.get_level_values(-1)[1]
    if n < TLDband:
        ErrMsg = '''input n ({}) < TLD band aggregation ({})
            This function cannot be used to disaggregate a TLD'''.format(n, TLDband)
        raise ValueError(ErrMsg)
    TLDn = TLD.groupby(nband(n)).sum() #TLD by bands
    TLDn.index = TLDn.index + n #re-index to top end of each band
    TLDn.at[0,:]=0 #TLD with initial zero value
    return TLDn.sort_index()


# In[6]:

def normalize_TLD(TLD):
    '''Normalizes TLD so TLD will contain proportion of trips 
    for each distance band rather thab absolute number of trips.'''
    return TLD.apply(lambda x: x / x.sum())


# In[7]:

def trim_index_TLD(TLD, index_names_to_keep='from', inplace=False):
    '''Wrapper for trim_index_df, adapted for TLD.'''
    return trim_index_df(TLD, index_names_to_keep, inplace)


# In[8]:

def to_numeric_TLD(TLD):
    '''Returns TLD where strings have been converted to numbers.'''
    tmp_index_names = TLD.index.names
    TLD.index = pd.to_numeric(TLD.index)
    TLD.index.names = tmp_index_names #I can't remmeber now why this is necessary
    return TLD.apply(lambda x: pd.to_numeric(x))


# In[9]:

def truncate_TLD(TLD, dist):
    '''Truncates a dataframe based on the index values.'''
    return TLD.loc[TLD.index < dist]


# In[10]:

def avgdist(TLD,col,level=-1):
    '''Returns the average distance (weighted average, SUMPRODUCT)
    of specified column in TLD. TLD should contain totals, not proportions.'''
    return (TLD[col] * TLD.index.get_level_values(-1)).sum()


# In[11]:

def TLD_col(mat, dist_band, dist_col=0):
    '''Returns the Trip-Lenght Distribution of mat, 
    based on dist_col, aggregated by dist_band.'''
    #TODO: implement normalized
    
    if isinstance(dist_col, int):
        dist_col = mat.columns[dist_col]
    
    TLD = mat.copy()
    TLD.ix[:,dist_col] = TLD.ix[:,dist_col].apply(nband(dist_band))
    
    TLD = TLD.groupby(by=dist_col).sum()
    TLD.index = TLD.index + dist_band #top end of each band
    TLD.at[0,:]=0 #fill initial zero value
    return TLD.sort_index()


# In[12]:

def TLD_SingleDist(mat, dist, dist_band, dist_col=0):
    '''Returns the Trip-Lenght Distribution of mat, 
    based on distance (dist_col) form dist, aggregated by dist_band.
    mat can have any number of culumns, but only dist_col will be used
    for the TLD. dist_col admits integer and column name.'''
    #TODO: implement normalized
    
    if isinstance(dist_col, int):
        dist_col = dist.columns[dist_col]
    
    df = mat.join(dist.ix[:,[dist_col]]).fillna(0)
    TLD = TLD_col(df, dist_band, dist_col)
    return TLD


# In[13]:

def TLD_MultiDist(mat, dist, dist_band):
    '''Returns the Trip-Length Distribution of mat.
    TLD for each mat column will be based on the corresponding
    column from dist (in order). mat and dist must have the same
    number of columns, or just the first distance column will be
    used.'''
    #TODO: implement normalized
    
    if len(mat.columns) != len(dist.columns):
        return TLD_single(mat, dist, dist_band)
    
    dfs = zip_df_cols([mat,dist])
    TLDs = [TLD_col(df, dist_band, 1) for df in dfs]
    
    TLD = pd.DataFrame()
    for xTLD in TLDs:
        TLD = pd.concat([TLD, xTLD], axis=1)
        
    return TLD


# In[14]:

def read_EMME_TLD(file):
    '''Returns TLD df from an EMME TLD report file, with columns:
    ['from','to','density_abs','density_norm','cumulative_abs','cumulative_norm']
    '''
    
    # EMME_TLD_cols - in order, position matters
    EMME_TLD_cols = ['from','to','density_abs','density_norm','cumulative_abs','cumulative_norm']

    idx_cols = EMME_TLD_cols[:2]
    data_cols = EMME_TLD_cols[2:]
    
    # RegEx to read EMME format:
    NumberPat = r'-?\.?\d*\.?\d+'
    TLDRowPat = r'(?<=\n)\s*({0})\s+({0})\s+({0})\s+({0})\s+({0})\s+({0})'
    EMMErecord_re = re.compile(TLDRowPat.format(NumberPat))
    
    # Read data
    with open(file, 'r') as f:
        f_content = f.read()
        data = EMMErecord_re.findall(f_content)
            
    # Convert data to DataFrame
    df = pd.DataFrame.from_records(data,
                                   columns=EMME_TLD_cols,
                                   index=idx_cols)
    return df


# In[15]:

def read_EMME_TLDs(files):
    '''Reads all TLD reports specified in files
    and returns four DataFrames, with the TLDs combined.
    Recomended: use glob to get the list of files from a pattern.
    Returns one DataFrame for each of the TLD EMME columns:
    ['density_abs','density_norm','cumulative_abs','cumulative_norm']
    '''
    TLDs = [read_EMME_TLD(file) for file in files]
    combinedTLDs = list(PairWiseColumnGroups(TLDs))
    
    filenames = [os.path.basename(file) for file in files]
    for TLD in combinedTLDs:
        TLD.columns = filenames
    density_abs, density_norm, cumulative_abs, cumulative_norm = combinedTLDs
    
    return density_abs, density_norm, cumulative_abs, cumulative_norm


# In[75]:

#TODO: Set xmax, ymax for x and y axes
def TLD_to_JPG(TLD, OutputName='TLD.png', ylabel='Trips', units='',
               prefixes='', suffixes=''):
    '''Produces a graph from TLD, all columns together.
    Includes average distance.
        prefixes         - to prepend to each column. Use as a marker.
        suffixes         - to append to each column. Use as a marker.
    '''
    
    if prefixes:
        try:
            TLD.columns = [prefix+col for col,prefix in zip(TLD.columns,prefixes)]
        except:
            raise ValueError("prefixes must have the same length as df.columns.")
    
    if suffixes:
        try:
            TLD.columns = [col+sufix for col,sufix in zip(TLD.columns,suffixes)]
        except:
            raise ValueError("suffixes must have the same length as df.columns.")
    
    if duplicates_in_list(TLD.columns):
        raise ValueError("Duplicate names in DataFrame's columns.")
    
    plt.clf()
    TLD.plot(title='Trip-Length Distribution', legend=True)
    lgd = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
                      fancybox=True, ncol=len(TLD.columns))
    plt.xlabel('Dist')
    plt.ylabel(ylabel)

    if units:
        col_label = 'Avg Dist ({})'.format(units)
    else:
        col_label = 'Avg Dist'

    table = plt.table(
        cellText=[['{:,.2f}'.format(avgdist(TLD,col))] for col in TLD],
        colWidths = [0.1],
        rowLabels=[' {} '.format(col) for col in TLD],
        colLabels=[col_label],
        loc='upper right')
    #table.set_fontsize(16)
    table.scale(2, 2)

    oName = OutputName
    plt.savefig(oName, bbox_inches='tight')
    plt.close()


# In[37]:

def TLD_cols_to_JPGs(TLD, oFileNamePattern='TLD_{}.png', ylabel='Trips', units=''):
    '''Produces a graph for each column of TLD.
    Names based on oFileNamePattern and column names.
    Includes average distance.'''
    for col in TLD:
        oFname = oFileNamePattern.format(col)
        TLD_to_JPG(TLD[[col]], oFname, ylabel=ylabel, units=units)


# In[71]:

#TODO: output average distances as DataFrame (and export as csv?)
def TLD_comparison_to_JPGs(TLDs, oFileNamePattern='TLD_{}.png', ylabel='Trips', units=''):
    '''Produces comparison graphs of the columns in each TLD in TLDs list.
    Columns are taken pairwise, in positional order.
    Names based on column names.'''
    comparisonTLDs = zip_df_cols(TLDs)
    for TLD in comparisonTLDs:
        TLDname = '-'.join(TLD.columns)
        OutputName = oFileNamePattern.format(TLDname)
        TLD_to_JPG(TLD, OutputName=OutputName, ylabel='Trips', units='')
