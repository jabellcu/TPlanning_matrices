
# coding: utf-8

import numpy as np
import pandas as pd

def band(Series, dist_band):
    '''Returns Series, binned (classified) in groups by dist_band'''
    return int(Series/dist_band)*dist_band

def TLD(mat,dist,dist_col,dist_band):
    '''Returns the Trip-Lenght Distribution of mat, 
    based on distance form dist, aggregated by dist_band'''
    TLD = mat.join(dist).fillna(0)
    #TODO: implement dist_col as dictionary
    TLD[dist_col] = TLD[dist_col].apply(band, args=[dist_band])
    
    #TODO: check totals in TLD. First band seem to be missing.
    TLD = TLD.groupby(dist_col).sum()
    TLD = TLD.reindex(TLD.index + dist_band) #top end of each band
    TLD.at[0,:]=0 #fill initial zero value
    return TLD.sort_index()
