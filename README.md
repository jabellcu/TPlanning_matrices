# TPlanning_matrices
Matrix manipulation toolkit for transport planners written in python.

Matrices:
 - Input / Output in different formats (e.g.: EMME, TBA3).
 - Calculating trip-ends.
 - Conversion from one zoning system to another.
   (Generalization for both cost and demand -trip- matrices)
 - Proportions for origins, destinations, columns.
   (e.g.: segmentation by time periods, demand segments, etc)
 - Produce trip-end comparisons between matrices: scatterplots and
   regression statistics.

Trip-Length Distributions:
 - Calculating Trip-Length Distributions from matrices.
 - Input / Output in different formats (e.g.: EMME, TBA3).
 - Adjust starting point.
 - Truncate maximum distance.
 - Calculate TLD proportions and average distance.
 - Aggregate TLD to different bands.
 - Produce TLD graphs.

Gravity models:
 - Estimate Gravity Parameters based on different functions.
 - Apply gravity models.

Assumes matrices are pandas dataframes, with "Origin" and "Destination" as multiindex, and one column per matrix. Matrices could be trips or cost.

# **WORK IN PROGRESS**
