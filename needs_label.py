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
get_ipython().magic(u'clear ')
get_ipython().magic(u'ls ')
get_ipython().magic(u'cd ../feature/naive_iter_2000/test/')
get_ipython().magic(u'cd ip2')
get_ipython().magic(u'ls ')
np.fromfile('0', dtype=np.float32)
labels = np.fromfile('0', dtype=np.float32)
labels.shape
print labels[:100]
get_ipython().magic(u'ls ')
get_ipython().magic(u'cd ../../../')
get_ipython().magic(u'ls ')
get_ipython().magic(u'cd ../')
get_ipython().magic(u'ls ')
get_ipython().magic(u'cd data')
get_ipython().magic(u'ls ')
with open(t10k.lst, 'r') as lbf:
    test_label = np.array([int(line.split()[1]) for line in lbf])
with open('t10k.lst', 'r') as lbf:
    test_label = np.array([int(line.split()[1]) for line in lbf])
test_label.shape
print test_label[:50]
test_dataset = testset
test_dataset.shape
import pyflann
import caffe
caffe.set_mode_cpu()
MODEL_FILE = '../custom/lenet_naive.prototxt'
PRETRAINED_MODEL = '../lenet_naive_iter_2000.caffemodel'
net = caffe.Net(MODEL_FILE, PRETRAINED_MODEL, caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_raw_scale('data', 255)
net.blobs['data'].reshape(1, 3, 28, 28)
image = caffe.io.load_image('../data/004192.png')
transformed_image = transformer.preprocess('data', image)
net.blobs['data'].data[...] = transformed_image
output = net.forward()
sample_feat = output['ip2norm'][0]
print sample_feat
flann = FLANN()
flann = flann.FLANN()
flann = pyflann.FLANN()
result, dists = flann.nn(test_dataset, sample_feat, 10)
print result
print dists
print test_label[7960]
import matplotlib.pyplot as plt
plt.imshow(image)
plt.show()
print test_label[7823]
print test_label[7439]
print test_label[6359]
print test_label[4465]
print test_label[2791]
get_ipython().magic(u'ls ')
get_ipython().magic(u'cd ../')
get_ipython().magic(u'ls ')
get_ipython().magic(u'cd custom/')
get_ipython().magic(u'ls ')
get_ipython().magic(u'save needs_label 1-82')
