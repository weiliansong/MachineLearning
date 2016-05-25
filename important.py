# coding: utf-8
srcdir = '/u/amo-d0/ugrad/wso226/MachineLearning/tools/caffe-sl/examples/mnist_sl/feature/naive_iter_2000/test/ip2_mat'
np.concatenate([loadmat(x) for x in glob.glob(srcdir+'/*.mat')])]
np.concatenate([loadmat(x) for x in glob.glob(srcdir+'/*.mat')])
import numpy as np
testset = np.concatenate([loadmat(x) for x in glob.glob(srcdir+'/*.mat')])
import glob
testset = np.concatenate([loadmat(x) for x in glob.glob(srcdir+'/*.mat')])
from scipy.io import loadmat
testset = np.concatenate([loadmat(x) for x in glob.glob(srcdir+'/*.mat')])
testset = np.concatenate([loadmat(x).reshape(1, -1) for x in glob.glob(srcdir+'/*.mat')])
testset = np.concatenate([loadmat(x)['feat'].reshape(1, -1) for x in glob.glob(srcdir+'/*.mat')])
testset.shape
testset = np.concatenate([loadmat(x)['feat'].reshape(100, -1)) for x in glob.glob(srcdir+'/*.mat')])
testset = np.concatenate([loadmat(x)['feat'].reshape(100, -1) for x in glob.glob(srcdir+'/*.mat')])
testset.shape
get_ipython().magic(u'history')
get_ipython().magic(u'ls ')
get_ipython().magic(u'cd custom/')
get_ipython().magic(u'ls ')
get_ipython().magic(u'save important.py')
get_ipython().magic(u'save --help')
get_ipython().magic(u'save')
get_ipython().magic(u'save important 1-23')
