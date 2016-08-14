
# coding: utf-8

# In[1]:

import re


# In[ ]:

def ListElementsInStr(s, lst):
    '''returns the elements of lst found in s'''
    regex = re.compile('({})'.format('|'.join(lst)))
    return regex.findall(s)


# In[2]:

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


# In[3]:

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


# In[4]:

def PairWiseColumnGroups(dflist):
    '''generator yields dataframes formed of pair-wise concatenation
    of columns from each df in the input dataframe list.'''
    max_cols = max([len(df.columns) for df in dflist])
    for i in range(max_cols):
        try:
            yield pd.concat([df.iloc[:,i] for df in dflist], axis=1)
        except IndexError:
            raise IndexError('Input dataframes have different number of columns.')


# In[ ]:

def SetIntras(mat, value=0, inplace=True):
    '''Sets intra zonal values'''
    if inplace:
        mat.loc[mat.index.get_level_values(0) == mat.index.get_level_values(1), :] = value
    else:
        aux = mat.copy()
        aux.loc[mat.index.get_level_values(0) == mat.index.get_level_values(1), :] = value


# In[ ]:

def duplicates_in_list(lst):
    '''Returns True in there are duplicates in lst.'''
    if len(lst) != len(set(lst)):
        return True
    else:
        return False

