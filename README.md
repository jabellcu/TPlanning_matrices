# TPlanning_matrices
Matrix manipulation toolkit for transport planners written in python.

 - Trip-ends
 - Zone matrix aggregation to sectors
 - Sector matrix disagregation to zones
 - Conversion from one zoning system to another (Generalization for both cost and demand -trip- matrices)
 - Matrix segmentation (by time periods, demand segments, etc)

Assumes matrices are pandas dataframes, with "Origin" and "Destination" as multiindex, and one column per matrix. Matrices could be trips or cost.

# **WORK IN PROGRESS**
Just a collection notebooks with tools for now
