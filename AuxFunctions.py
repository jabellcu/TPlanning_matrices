
# coding: utf-8

import numpy as np
import pandas as pd
import re

def ListElementsInStr(s, lst):
    '''returns the elements of lst found in s'''
    regex = re.compile('({})'.format('|'.join(lst)))
    return regex.findall(s)

def sort_by_list_key_func(lst):
    '''Returns a function to be used for sorting.
    The returned function returns an integer
    with the index of each element in the input list,
    if the element is found in the element being sorted.
    Puts elemnts not founf at the beggining.
    
    >>> sorted('a b c A B C'.split(),
    ...         key=sort_by_list_key_func('b c a'.split()))
    ['A', 'B', 'C', 'b', 'c', 'a']
    
    '''
    lst_re = re.compile('({})'.format('|'.join(lst)))
    def sort_func(s):
        'Sorting function'
        try:
            return lst.index(lst_re.search(s).group())
        except:
            return -1
    return sort_func

def sort_df_by_lists(df, lists):
    '''Returns the input dataframe with columns sorted by the appearance of
    each element of each list in input lists in the column names,
    by order of appearance in the input lists
    
    >>> list(sort_df_by_lists(
    ...         pd.DataFrame(columns=['A1', 'A2', 'A3', 'B1', 'B2', 'B3']),
    ...         [['B', 'A'],['3', '1', '2']]
    ...         ).columns)
    ['B3', 'A3', 'B1', 'A1', 'B2', 'A2']
    
    '''
    ## Sort DataFrame's columns
    cols = list(df.columns)
    for lst in lists:
        cols.sort(key=sort_by_list_key_func(lst))
    return df[cols] #sorted! 

def PairWiseColumnGroups(dflist):
    '''generator yields dataframes formed of pair-wise concatenation
    of columns from each df in the input dataframe list.'''
    max_cols = max([len(df.columns) for df in dflist])
    for i in range(max_cols):
        try:
            yield pd.concat([df.iloc[:,i] for df in dflist], axis=1)
        except IndexError:
            raise IndexError('Input dataframes have different number of columns.')

def SetIntras(mat, value=0, inplace=True):
    '''Sets intra zonal values'''
    if inplace:
        mat.loc[mat.index.get_level_values(0) == mat.index.get_level_values(1), :] = value
    else:
        aux = mat.copy()
        aux.loc[mat.index.get_level_values(0) == mat.index.get_level_values(1), :] = value

def duplicates_in_list(lst):
    '''Returns True in there are duplicates in lst.'''
    if len(lst) != len(set(lst)):
        return True
    else:
        return False

def CheckEMMEmatName(s):
    '''Throws an exception id s is not a valid EMME name.'''
    if len(s) < 1 or len(s) > 6:
        ErrMsg = '{} is not a valid EMME name.'.format(s)
        raise NameError(ErrMsg)

def CheckEMMEmatNumber(s):
    '''Trhows an exception id s is not a valid EMME number.'''
    ErrMsg = '{} is not a valid EMME number.'.format(s)
    try:
        if (('mf' not in s[:2] 
        and 'md' not in s[:2]
        and 'mo' not in s[:2]
        and 'ms' not in s[:2])
        or len(s[2:]) < 2
        or len(s[2:]) > 3 
        or not StringIsInt(s[2:])):
            raise NameError(ErrMsg)
    except:
        raise NameError(ErrMsg)

def StringIsInt(s):
    '''True if string represents an int, False otherwise'''
    try: 
        int(s)
        return True
    except ValueError:
        return False

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

def I(n):
    '''returns identity matrix of n x n'''
    matI = pd.DataFrame({'O': [x+1 for x in range(n) for _ in range(n)],
                         'D': [x+1 for x in range(n)] * n})
    matI['T'] = (matI.O == matI.D).apply(int)
    matI = matI.set_index(['O', 'D'])
    return matI

import random

def randomizeSeries(S, fraction):
    '''Adds variability to a pd.Series'''
    return S+pd.Series([random.randint(-int(x/fraction),int(x/fraction)) for x in S], index=S.index)

def randomizeTE(TE):
    return TE.apply(randomizeSeries, args=[10])

def zip_df_cols(dflist):
    '''generator yields dataframes formed of pair-wise concatenation
    of columns from each df in the input dataframe list.'''
    max_cols = max([len(df.columns) for df in dflist])
    for i in range(max_cols):
        try:
            yield pd.concat([df.iloc[:,i] for df in dflist], axis=1)
        except IndexError:
            raise IndexError('Input dataframes have different number of columns.')

