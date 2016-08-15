
# coding: utf-8

import numpy as np
import pandas as pd

from Matrix import Matrix
from AuxFunctions import zip_df_cols

def band(Series, dist_band):
    '''Returns Series, binned (classified) in groups by dist_band'''
    return int(Series/dist_band)*dist_band

def TLD_col(mat, dist_band, dist_col=0):
    '''Returns the Trip-Lenght Distribution of mat, 
    based on dist_col, aggregated by dist_band.'''
    
    if isinstance(dist_col, int):
        dist_col = mat.columns[dist_col]
    
    TLD = mat.copy()
    TLD.ix[:,dist_col] = TLD.ix[:,dist_col].apply(band, args=[dist_band])
    
    TLD = TLD.groupby(by=dist_col).sum()
    TLD.index = TLD.index + dist_band #top end of each band
    TLD.at[0,:]=0 #fill initial zero value
    return TLD.sort_index()

def TLD_SingleDist(mat, dist, dist_band, dist_col=0):
    '''Returns the Trip-Lenght Distribution of mat, 
    based on distance (dist_col) form dist, aggregated by dist_band.
    mat can have any number of culumns, but only dist_col will be used
    for the TLD. dist_col admits integer and column name.'''
    
    if isinstance(dist_col, int):
        dist_col = dist.columns[dist_col]
    
    df = mat.join(dist.ix[:,[dist_col]]).fillna(0)
    TLD = TLD_col(df, dist_band, dist_col)
    return TLD

def TLD_MultiDist(mat, dist, dist_band):
    '''Returns the Trip-Length Distribution of mat.
    TLD for each mat column will be based on the corresponding
    column from dist (in order). mat and dist must have the same
    number of columns, or just the first distance column will be
    used.'''
    if len(mat.columns) != len(dist.columns):
        return TLD_single(mat, dist, dist_band)
    
    dfs = zip_df_cols([mat,dist])
    TLDs = [TLD_col(df, dist_band, 1) for df in dfs]
    
    TLD = pd.DataFrame()
    for xTLD in TLDs:
        TLD = pd.concat([TLD, xTLD], axis=1)
        
    return TLD

#TODO
def read_EMME_TLD(file):
    '''returns TLD df from an EMME TLD report'''
    ...
