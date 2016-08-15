# TPlanning_matrices
Matrix manipulation toolkit for transport planners written in python.

 - Matrix input / output in different formats.
   (e.g.: EMME, TBA3)
 - Calculating trip-ends.
 - Conversion from one zoning system to another.
   (Generalization for both cost and demand -trip- matrices)
 - Matrix proportions for origins, destinations, columns.
   (e.g.: segmentation by time periods, demand segments, etc)
 - Calculating Trip-Length Distributions.
 - Estimate Gravity Parameters.
 - Apply gravity models.

Assumes matrices are pandas dataframes, with "Origin" and "Destination" as multiindex, and one column per matrix. Matrices could be trips or cost.

# **WORK IN PROGRESS**
