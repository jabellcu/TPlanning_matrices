
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import scipy.stats as stats
import pylab

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

def trim_index_df(df: pd.DataFrame, index_names_to_keep: list, inplace=False):
    '''Drops all indexes except for specified index names.'''
    
    indexes_to_drop = list(df.index.names)
    try:
        indexes_to_drop.remove(index_names_to_keep)
    except ValueError:
        try:
            for idxn in index_names_to_keep:
                indexes_to_drop.remove(idxn)
        except ValueError:
            pass
    
    if inplace:
        df.reset_index(level=indexes_to_drop, drop=True, inplace=True)
    else:
        return df.reset_index(level=indexes_to_drop, drop=True)

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

def flatten_cols(df, sep='_', strip=True, inplace=True):
    '''Turns a MultiIndex-column dataframe into a simple-column dataframe.'''

    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("Columns are already flat")
    
    if strip:
        cols = [sep.join(col_vals).strip() for col_vals in df.columns.values]
    else:
        cols = [sep.join(col_vals) for col_vals in df.columns.values]
    
    if inplace:
        df.columns = cols
    else:
        newdf = df.copy()
        newdf.columns = cols
        return newdf

def split_cols(df, sep='_', inplace=True):
    '''Turns a simple-column dataframe into a MultiIndex-column by
    splitting the labels'''

    if isinstance(df.columns, pd.MultiIndex):
        raise ValueError("Columns are already MultiIndex")
    
    tup = [c.split(sep) for c in df.columns.values]
    newidx = pd.MultiIndex.from_tuples(tup)

    if inplace:
        df.columns = newidx
    else:
        newdf = df.copy()
        newdf.columns = newidx
        return newdf


def zip_df_cols(dflist, default_col_index=None):
    '''generator yields dataframes formed of pair-wise concatenation of columns
    from each df in the input dataframe list.
    If the dataframes are of uneven length, missing values are filled-in with
    the first column of each df. This is usefull when broadcasting one column
    to the rest of the dataframes'''
    dflist_coln = [len(df.columns) for df in dflist]
    max_coln = max(dflist_coln)
    for i in range(max_coln):
        if default_col_index is not None:
            #default to first column for df with not enough
            use_cols = [i if i <= (coln-1) else 0 for coln in dflist_coln]
            yield pd.concat([df.iloc[:,col] for col,df in zip(use_cols, dflist)], axis=1)
        else:
            try:
                yield pd.concat([df.iloc[:,i] for df in dflist], axis=1)
            except IndexError:
                raise IndexError('Input dataframes have different number of columns.')

def df_difference(dfi, dff, percent=False):
    '''Returns dff-dfi difference. Asumes compatible indexes.
    Columns substracted positionally, because if names are overlapping
    then it is just better to do "dff-dfi").
    percentage=True will return (dff-dfi)/dfi instead.'''
    
    diff = pd.DataFrame()
    max_coln = max(len(dfi.columns), len(dff.columns))
    
    for i in range(max_coln):
        try:
            coli = dfi.columns[i]
            colf = dff.columns[i]
            col = '{}-{}'.format(colf, coli)
            
            if percent:
                diff[col] = (dff[colf] / dfi[coli]) -1
            else:
                diff[col] = dff[colf] - dfi[coli]
                
        except:
            raise IndexError('Input dataframes have different number of columns.')
    return diff

def ScatterPlot_ConsecutiveColPairs(df, oFileNamePattern='{}', title='',
                xaxis_eq_yaxis=True, homogeneous_axis=True, min_axis=None,
                prefixes='', suffixes='', output_df=False, **kwargs):
    
    '''Produces scatterplot graphs of consecutive df columns.
        xaxis_eq_yaxis   - both x and y axis' maximum values are the same
        homogeneous_axis - all scatterplots for input df have the same axis
        min_axis         - minimum axis values
        prefixes         - to prepend to each column. Use as a marker.
        suffixes         - to append to each column. Use as a marker.
    '''
    
    if prefixes:
        try:
            df.columns = [prefix+col for col,prefix in zip(df.columns,prefixes)]
        except:
            raise ValueError("prefixes must have the same length as df.columns.")
    
    if suffixes:
        try:
            df.columns = [col+sufix for col,sufix in zip(df.columns,suffixes)]
        except:
            raise ValueError("suffixes must have the same length as df.columns.")
    
    if duplicates_in_list(df.columns):
        raise ValueError("Duplicate names in DataFrame's columns.")
        
    cols = df.columns
    #TODO: review xaxis_eq_yaxis and homogeneous_axis
    minv = df.min().min()
    maxv = df.max().max()

    if min_axis is not None:
        minv = min_axis
    
    for colin, colfn in zip(cols, cols[1:]):
        ColPairName = ' - '.join([colin, colfn])
        
        xminv = df[colin].min()
        xmaxv = df[colin].max()
        yminv = df[colfn].min()
        ymaxv = df[colfn].max()

        if homogeneous_axis:
            #use the global min/max
            xminv = minv
            xmaxv = maxv
            yminv = minv
            ymaxv = maxv
        
        if xaxis_eq_yaxis:
            #use local min/max for each scatterplot
            minv = min(xminv, yminv)
            maxv = max(xmaxv, ymaxv)
            xminv = minv
            xmaxv = maxv
            yminv = minv
            ymaxv = maxv

        #regression
        slope, intercept, r_val, p_val, slope_std_err = stats.linregress(df[colin], df[colfn])
        r_val = r_val ** 2
        regression_formula = '$y={:.2f}*x{:+.2f}$\n$R^2: {:.3f}$'.format(
                                slope, intercept, r_val)

        #format plot
        if title:
            df.plot(kind= 'scatter', x=colin, y=colfn,
                    title=title.format(ColPairName), legend =True)
        else:
            df.plot(kind= 'scatter', x=colin, y=colfn, legend =True)

        #plot regression line
        plt.plot(df[colin], slope * df[colin] + intercept)

        pylab.annotate(regression_formula, xycoords='axes fraction',
                       xy=(.60,.05), fontsize=16,
                       bbox=dict(facecolor='white', edgecolor='black',
                                 boxstyle='round,pad=0.3'))

        axes = plt.gca()
        axes.set_xlim([xminv,xmaxv])
        axes.set_ylim([yminv,ymaxv])

        #export plot
        oFileName = oFileNamePattern.format(ColPairName)
        plt.savefig(oFileName)
        plt.close()

        if output_df:
            df.to_csv(oFileName + '.csv')

def RegressionStats_ConsecutiveColPairs(df, prefixes='', suffixes='', **kwargs):
    '''Returns a dataframe with the regression stats of consecutive df columns.
        prefixes         - to prepend to each column. Use as a marker.
        suffixes         - to append to each column. Use as a marker.
    '''
    
    if prefixes:
        try:
            df.columns = [prefix+col for col,prefix in zip(df.columns,prefixes)]
        except:
            raise ValueError("prefixes must have the same length as df.columns.")
    
    if suffixes:
        try:
            df.columns = [col+sufix for col,sufix in zip(df.columns,suffixes)]
        except:
            raise ValueError("suffixes must have the same length as df.columns.")
    
    if duplicates_in_list(df.columns):
        raise ValueError("Duplicate names in DataFrame's columns.")
    
    regression_df = pd.DataFrame(columns=['slope', 'intercept', 'R2'])
    cols = df.columns
    
    for colin, colfn in zip(cols, cols[1:]):
        ColPairName = ' - '.join([colin, colfn])
        
        slope, intercept, r_val, p_val, slope_std_err = stats.linregress(df[colin], df[colfn])
        r_val = r_val ** 2
        regression_df.loc[ColPairName] = [slope, intercept, r_val]
    
    return regression_df

def Compare_ConsecutiveColPairs(df, oFileNamePattern='{}', 
        output_scatterplots=True, output_regresion_stats=True, **kwargs):
    '''Produces scatterplots and regresion statistics of consecutive column pairs.
    df can be a single dataframe or a list of dataframes.
    Returns dataframe regression stats. Parameters:
        xaxis_eq_yaxis   - both x and y axis' maximum values are the same
        homogeneous_axis - all scatterplots for input df have the same axis
        min_axis         - minimum axis values
        prefixes         - to prepend to each column. Use as a marker.
        suffixes         - to append to each column. Use as a marker.
    '''
    if isinstance(df, list):

        if len(df) == 1:
            tdfs = df
        else:
            tdfs = zip_df_cols(df, default_col_index=0)
        
        regression_stats_dfs = []
        for df in tdfs:

            if output_scatterplots:
                ScatterPlot_ConsecutiveColPairs(df, oFileNamePattern, **kwargs)
            regression_stats_dfs.append(RegressionStats_ConsecutiveColPairs(df))
            
        regression_stats = pd.concat(regression_stats_dfs)
        
    elif isinstance(df, pd.DataFrame):
        
        if output_scatterplots:
            ScatterPlot_ConsecutiveColPairs(df, **kwargs)
        regression_stats = RegressionStats_ConsecutiveColPairs(df)
        
    else:
        
        raise ValueError("Input must be a pandas DataFrame or list of DataFrames")
        
    if output_regresion_stats:
        regression_stats.to_csv(oFileNamePattern.replace('{}', 'regression_stats') + '.csv')
    return regression_stats

def allequal(lst):
    '''Returns True if all elements of lst are identical'''
    return not lst or lst.count(lst[0]) == len(lst)
