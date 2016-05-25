# coding: utf-8
import numpy as np
import glob
from scipy.io import loadmat
x = 0
counter = []
while x < 10000:
    print x
    x += 100
    counter.append(x)
for x in counter:
    print x
counter = []
x = 0
while x < 10000:
    counter.append(x)
    x += 100
print counter
get_ipython().magic(u'ls ')
test_set = []
test_set_o = []
for x in counter:
    file_name = 'feat_' + str(x) + '.mat'
    test_set_o.append(loadmat(file_name).reshape(100, -1))
for x in counter:
    file_name = 'feat_' + str(x) + '.mat'
    test_set_o.append(loadmat(file_name)['feat'].reshape(100, -1))
print test_set_o
print test_set_o.shape
print test_set_o[0].shape
get_ipython().magic(u'clear ')
test_set = np.concatenate(test_set_o)
print test_set.shape
print test_set[0]
get_ipython().magic(u'cd ../../../../data/')
with open('t10k.lst', 'r') as lbf:
        test_label = np.array([int(line.split()[1]) for line in lbf])
print test_label.shape
get_ipython().magic(u'paste')
image = caffe.io.load_image('../data/009062.png')
transformed_image = transformer.preprocess('data', image)
net.blobs['data'].data[...] = transformed_image
output = net.forward()
sample_feat = output['ip2norm'][0]
flann = pyflann.FLANN()
result, dists = flann.nn(test_set, sample_feat, 10)
print result
print test_label[2823]
print dists
print test_set[1]
print test_set[0]
get_ipython().magic(u'ls ')
get_ipython().magic(u'cd ../feature/naive_iter_2000/test/ip2_mat/')
get_ipython().magic(u'ls ')
for x in counter:
    file_name = 'feat_' + str(x) + '.mat'
    test_set_o.append(loadmat(file_name)['feat'].reshape(100, -1))
    print file_name
test_set_0
print test_set_o[0]
print test_set_o[1]
test_set = np.concatenate(test_set_o[i] for i in len(test_set_o))
len(test_set_o)
get_ipython().magic(u'ls ')
test_set_o = []
for x in counter:
    file_name = 'feat_' + str(x) + '.mat'
    test_set_o.append(loadmat(file_name)['feat'].reshape(100, -1))
    print file_name
test_set_o.len
len(test_set_o)
test_set = np.concatenate(test_set_o[i] for i in range(len(test_set_o)))
for x in test_set_o:
    for sample in test_set_o[x]:
        test_set.append(sample)
for x in len(test_set_o):
    for sample in len(test_set_o[x]):
        test_set.append(test_set_o[x][sample])
for x in range(len(test_set_o)):
    for sample in range(len(test_set_o[x])):
        test_set.append(test_set_o[x][sample])
test_set = []
for x in range(len(test_set_o)):
    for sample in range(len(test_set_o[x])):
        test_set.append(test_set_o[x][sample])
test_set[0]
test_set[1]
test_set_o[1]
test_set_o[100]
test_set[100]
result, dists = flann.nn(test_set, sample_feat, 10)
test_set.shape
test_set[0]
for x in test_set[:5]:
    print x
get_ipython().magic(u'clear ')
test_set_g = []
for idx, x in enumerate(test_set):
    test_set_g[idx] = x
test_set_g = np.asarray(test_set)]
test_set_g = np.asarray(test_set)
test_set_g.shape
result, dists = flann.nn(test_set_g, sample_feat, 10)
print result
print test_label[2823]
import matplotlib.pyplt as plt
get_ipython().magic(u'ls ')
get_ipython().magic(u'cd ../../../../custom/')
get_ipython().magic(u'save important_2 1-81')
