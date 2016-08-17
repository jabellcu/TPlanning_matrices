
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import scipy.stats as stats
import pylab

from AuxFunctions import duplicates_in_list, CheckEMMEmatName, zip_df_cols
from AuxFunctions import CheckEMMEmatNumber, ScatterPlot_ConsecutiveColPairs
from AuxFunctions import RegressionStats_ConsecutiveColPairs

def Zoning(zones: list, names=['O', 'D']) -> pd.MultiIndex:
    '''Returns a MultiIndex object with zones for origins and destinations.
    zones can be a list of zones: a square zoning system will be returned
    zones can also be [origs, dests]: all combinations from origs to dests
    are returned. origs and dests are list of zones.'''
    
    if all(isinstance(elem, list) for elem in zones):
        ODs = zones
        if not any(duplicates_in_list for lst in zones):
            raise ValueError('There are duplicated zones')
    
    elif isinstance(zones,list):
        ODs = [zones for name in names]
        if duplicates_in_list(zones):
            raise ValueError('There are duplicated zones')
    
    else:
        raise ValueError('"zones" must be a list or a list of lists')
    
    idx = pd.MultiIndex.from_product(ODs, names=names)
    
    if idx.names != names:
        raise ValueError('Zoning could not be created from {}.'.format(zones) + 
                         '\n"zones" must be a list or a list of lists')
    
    return idx

class Matrix(pd.DataFrame):
    '''A Matrix in Transport Planning is a pandas DataFrame,
    with Origins and Destinations as MultiIndex levels: [O, D]'''

    @property
    def _constructor(self):
        '''Matrix operations returns Matrix objects.'''
        return Matrix
    
    @property
    def Os(self):
        '''Returns origin names without duplicates.'''
        return list(self.index.get_level_values(0).unique())
    
    @property
    def Ds(self):
        '''Returns destination names without duplicates.'''
        return list(self.index.get_level_values(1).unique())
    
    @property
    def TO(self):
        '''Returns trip-ends for origins.'''
        return self.groupby(level=0).sum()
    
    @property
    def TD(self):
        '''Returns trip-ends for destinations.'''
        return self.groupby(level=1).sum()

    @property
    def TE(self):
        '''Returns trip-ends for both origins and destinations.'''
        # this is just a wrapper of TEs function.
        # This allows accesing it as a property,
        # which is consistent with TO and TD usage.
        return self.TEs()

    def TEs(self, index_name='zone', suffixes=['_TO', '_TD']):
        '''Returns Trip Ends: both trip origins and trip destinations
        in a single DataFrame. Allows customization of index name and suffixes,
        as this can be useful in some cases.'''
        TE = pd.merge(self.TO, self.TD, suffixes=suffixes,
                     left_index=True, right_index=True)
        TE.index.names = [index_name]
        return TE
    
    @property
    def TOTALS(self):
        '''Returns the matrix totals.'''
        return self.sum()
    
    def TransposeOD(self, sort=True):
        '''Transposes a matrix in ODT format: swaps Origins and Destinations.'''
        mat_T = self.swaplevel()
        mat_T.index.names = self.index.names
        if sort:
            mat_T = mat_T.sort_index()
        return mat_T
    
    @property
    def TOp(self):
        '''Returns origin proportions: Pij = Tij / TOi'''
        return self.groupby(level=0).apply(lambda x: (x/x.sum()).fillna(0))
    
    @property
    def TDp(self):
        '''Returns destination proportions: Pij = Tij / TDj'''
        return self.groupby(level=1).apply(lambda x: (x/x.sum()).fillna(0))
    
    @property
    def proportions(self):
        return self.apply(lambda x: x/sum(x), axis=1).fillna(0)
    
    @property
    def matrix(self):
        '''Returns matrix as a tradicional 2D matrix.'''
        return self.to_panel()
    
    #TODO:
    @staticmethod
    def from_panel(panel):
        ...
    
    def rezone(self, mapping: pd.DataFrame, mapping_cols=['old', 'new'],
               weight_cols=None, calculate_proportions=True,
               min_weight=0.00000001):
        '''Changes the zoning system based on mapping.
        A mapping is a correspondence between old zones and new zones.
        
           weights - ['Owght', 'Dwght'] to use for zone disaggregation
           calculate_proportions - if True, weight proportions will be 
               calculated and applied
           min_weight - value for weights with value zero'''
        
        if weight_cols:
            
            if len(weight_cols)!=2:
                raise ValueError("weight_cols must be as in ['Owght', 'Dwght']")
            
            Owght, Dwght = weight_cols
            
            #cap to min_weight
            Omap = mapping[mapping_cols].copy()
            Omap[Owght] = mapping[Owght].where(mapping[Owght] > min_weight, min_weight)
            Dmap = mapping[mapping_cols].copy()
            Dmap[Dwght] = mapping[Dwght].where(mapping[Dwght] > min_weight, min_weight)
            
            if calculate_proportions:
                #proportions always respect 'old' mapping column
                Omap[Owght] = Omap.groupby(mapping_cols[0])[Owght].apply(lambda x: (x/x.sum()))
                Dmap[Dwght] = Dmap.groupby(mapping_cols[0])[Dwght].apply(lambda x: (x/x.sum()))
        else:
            Omap = mapping.reset_index()[mapping_cols]
            Dmap = Omap
        
        suffixes = ['_' + n for n in self.index.names]
        mat = pd.merge(self.reset_index(), Omap.reset_index(),
                      left_on=self.index.names[-2], right_on=mapping_cols[0])
        mat = pd.merge(mat, Dmap.reset_index(),
                      left_on=self.index.names[-1], right_on=mapping_cols[0],
                      suffixes=suffixes)
        
        if weight_cols:
            if Owght == Dwght:
                Owght, Dwght = ['{}{}'.format(Owght,s) for s in suffixes]
                
            for col in self:
                mat[col] = mat[col] * mat[Owght] * mat[Dwght]
        
        NewODnames = ['{}{}'.format(mapping_cols[1],s) for s in suffixes]
        
        aux_cols = list(set(mat.columns) - set(self.columns) - set(NewODnames))
        mat = mat.drop(aux_cols, axis=1)
        
        mat = mat.groupby(NewODnames).sum()
        mat = Matrix(mat)
        
        if not all(x==y for x,y in zip(self.TOTALS, mat.TOTALS)):
            print("WARNING: rezoned matrix does not preserve the matrix totals.")
        
        return mat
    
        #TODO: update this function to allow outputing a weighted average (eg: for costs)
        # Deal with 0 trips in wght file, and:
        ## 1) Multiply src_file by wght_file
        ## 2) Disaggregate src x wght as if it was a demand matrix
        ## 3) Disaggregate wght as a demmand matrix as well
        ## 4) merge src_wght_hyl and wght_hyl on the first two columns:
        ## 5) Divide src x wght / wght at hyl level
        ## Aggregate!
        
    def complete(self, zones):
        '''Completes the matrix index with specified zones. Ignores existing zones.'''
        if isinstance(zones, pd.MultiIndex):
            #zones is a zoning system already (MultiIndex)
            zoning = zones
        elif isinstance(zones, list):
            #zones is just a list that needs to be expanded
            zoning = Zoning(zones)
        else:
            raise ValueError('"zones" must be list of zones or zoning system (MultiIndex)')
        zoning_union = self.index.union(zoning)
        return self.reindex(index=zoning_union)
    
    def submatrix(self, zoning: pd.MultiIndex):
        '''Returns a submatrix with the origins and destinations specified in zoning'''
        zoning_intersect = self.index.intersection(zoning)
        return self.reindex(zoning_intersect)
    
    @staticmethod
    def read_TBA3(file, mat_type='VALUE'):
        '''Reads a text file containing one or more matrices in TBA3 format.
        This is one of SATURN-friendly formats'''
        
        df = pd.read_csv(file, sep=' {1,}',
                              names='O D UC {}'.format(mat_type).split(),
                              index_col=[0,1,2],
                              header=None, engine='python')
        
        df = df.unstack()
        if mat_type == 'VALUE':
            #keep only UC lvl if no mat_type specified
            df.columns = df.columns.droplevel()
        mat = Matrix(df)
        
        return mat
        
    @staticmethod
    def read_EMME(file):
        '''Reads a text file containing one or more matrices in EMME format.
        Accepts matrices, trip origins, trip destinations and constants.
        Assumes one single value per row in the EMME files.
        '''

        EMMErecord_cols = {
            'md': ['zone', '_TD'],
            'mo': ['zone', '_TO'],
            'mf': ['O', 'D', '']
            } #TODO: remove difference by TO /TD / T ??

        ## RegEx to read EMME format:
        mat_re = re.compile(r'a matrix\s*=\s*(mo|md|mf|ms)(\d+)\s+(\w+?)\s+(-?\d+)\s+(.+?)\n(.*)\n(?=a matrix|d matrix|\Z|\s*\n)',
                           re.DOTALL | re.MULTILINE)
        # re groups:
        # mat_type, mat_num, mat_name, mat_default, mat_desc, mat_data

        EMMErecord_re = {
            'md': re.compile(r'\s*all\s+(\d+)\s*:\s*(-?\.?\d+\.?\d*)\n'),
            'mo': re.compile(r'\s*(\d+)\s+all\s*:\s*(-?\.?\d+\.?\d*)\n'),
            'mf': re.compile(r'\s*(\d+)\s+(\d+)\s*:\s*(-?\.?\d+\.?\d*)\n')
            } #TODO: implement ms

        ## Read Data
        data = {}
        filen = os.path.basename(file)
        fn, fext = os.path.splitext(filen)
        data[fn] = {}
        with open(file, 'r') as f:
            fcontent = f.read()
            #each source file might contain several matrices
            mat_blocks = mat_re.findall(fcontent)
            for matb in mat_blocks:
                mat_type, mat_num, mat_name, mat_default, mat_desc, mat_data = matb
                mat_rows = EMMErecord_re[mat_type].findall(mat_data)
                #TODO: Use named tuples
                data[fn][mat_name] = dict(zip(
                   'mat_type, mat_num, mat_default, mat_desc, mat_rows'.split(', '),
                   [mat_type, mat_num, mat_default, mat_desc, mat_rows]))

        ## Convert to DataFrame
        matrix = Matrix()
        for fn in data:
            for matn in data[fn]:
                mat_data = data[fn][matn]

                #convert rows into df, setting column names and index
                df_cols = EMMErecord_cols[mat_data['mat_type']]
                df_idx_cols = df_cols[:-1]
                df_data_cols = df_cols[-1]

                mat = Matrix.from_records(mat_data['mat_rows'],
                                               columns=df_cols,
                                               index=df_idx_cols)

                #this avoids repeating names:
                mat_id = '{}{}'.format(fn, df_data_cols)
                #numeric is needed for Matrix methods to work as expected
                matrix[mat_id] = pd.to_numeric(mat[df_data_cols])           

        return matrix
        
    def furness(self, TO, TD, tolerance=0.001, max_iter=100):
        '''Use FRATAR algorithm to adjust (balance) the matrix
        to target origins and destinations (TO, TD), within a certain tolerance.
        Will not always converge, hence cap maximum iterations to max_iter.'''
        
        sTO = self.TO
        sTD = self.TD
        fmat = self.copy()
        i = 1
        
        while True:
            # Balancing factors (Nan ~> 0, inf ~> 1):
            A = (TO / fmat.TO).fillna(0).replace([np.inf, -np.inf], 1)
            B = (TD / fmat.TD).fillna(0).replace([np.inf, -np.inf], 1)
            fmat = fmat.mul(A, axis=1, level=0).mul(B, axis=1, level=1)
        
            i += 1
            within_tol = not (any((fmat.TO - TO) > tolerance) or any((fmat.TD - TD) > tolerance))
            
            if within_tol:
                break
            if max_iter and (i >= max_iter):
                break
        
        return fmat
    
    def to_EMME(self, OutputName,
                file_header='', mat_number_start=100, mat_comment='', 
                default_val=0, decimals=4):
        '''Will write each of the columns of a dataframe (matrix)
        as stacked EMME matrices in a single file.
        Missing values are ignored.
        Matrix nnumbers will be sequential with column order,
        starting with mat_number_start.'''
        df_dict = self.to_dict()
        with open(OutputName, "w") as OutputFile:
            if file_header:
                OutputFile.write(file_header)
            
            for col in df_dict:
                mat = df_dict[col]
                mat_name = '{}'.format(col)
                CheckEMMEmatName(mat_name)
                
                mat_number = 'mf{0:02d}'.format(mat_number_start + col)
                CheckEMMEmatNumber(mat_number)
                
                if self.columns.nlevels > 1:
                    #for MultiIndex, use first column level for mat_type
                    mat_type = self.columns.names[0]
                    mat_cmnt = '{} {}: {}'.format(mat_comment, mat_type, col)
                else:
                    mat_cmnt = '{}: {}'.format(col, mat_comment)               

                # Write matrix headers:
                OutputFile.write("\nd matrix={}".format(mat_number))
                OutputFile.write("\na matrix={} {} {} '{}'".format(
                                    mat_number, mat_name, default_val, mat_cmnt))

                # Write data:
                for ODpair in mat:
                    O, D = ODpair
                    val = mat[ODpair]
                    #Missing values won't be written
                    if pd.notnull(val):
                        OutputFile.write('\n {} {}: {:.{dec}f}'.format(O, D, val, dec=decimals))

def TE_comparison_to_JPGs(mati, matf, oFileNamePattern='{}', title='',
                xaxis_eq_yaxis=True, homogeneous_axis=True, min_axis=0,
                prefixes='', suffixes=''):
    '''Produces scatterplots of trip ends in mati and matf.
    mati and matf columns will be compared pairwise, so must be ordered.
    Wrapper of ScatterPlot_ConsecutiveColPairs (with flavor).
        prefixes         - to prepend to each column. Use as a marker.
        suffixes         - to append to each column. Use as a marker.
    '''
    for df in zip_df_cols([mati.TE, matf.TE]):
        ScatterPlot_ConsecutiveColPairs(df, oFileNamePattern=oFileNamePattern,
                title=title, xaxis_eq_yaxis=xaxis_eq_yaxis,
                homogeneous_axis=homogeneous_axis, min_axis=min_axis,
                prefixes=prefixes, suffixes=suffixes)

def TE_RegressionStats(mati, matf, prefixes='', suffixes=''):
    '''Returns a dataframe with the regression statistics of mati and matf trip
    ends. mati and matf columns will be compared pairwise, so must be ordered.
    Wrapper of RegressionStats_ConsecutiveColPairs (with flavor).
        prefixes         - to prepend to each column. Use as a marker.
        suffixes         - to append to each column. Use as a marker.
    '''
    return pd.concat([RegressionStats_ConsecutiveColPairs(
                        df, prefixes=prefixes, suffixes=suffixes)
                     for df in zip_df_cols([mati.TE, matf.TE])])
