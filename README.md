# TensorGlue
This is a very early python implementation of recsys engine that currently includes 3 types of algorithms:
* item-to-item (with basic items similarity measure)
* SVD-based matrix factorization
* tensor factorization

Contextual information (such as movie genre or product category) is encapsulated into an additional dimension, s.t. full data is represented as a 3rd order tensor. Encapsulation of the contextual information requires additional pre-processing which was described in my talk at [TDA 2016](http://www.esat.kuleuven.be/stadius/TDA2016/program.php) conference and will be also explained in my future paper.
Pre-processing is implemented for both matrix (optionally) and tensor factorization.

Current version was tested only on Windows x64 with latest anaconda package. Major dependencies are:
* pandas
* numpy
* numba (used only for tensor decomposition)

**Important note:** Please, be aware, that evaluation of tensor factorization method is performed in batch (e.g. for all test users at once) and requires considerable amount of computer memory. Memory load can be controlled with `chunk` attribute of the model. General advise is to have PC with >8Gb of RAM.
