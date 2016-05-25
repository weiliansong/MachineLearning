import numpy as np
import glob
from scipy.io import loadmat
import caffe
import pyflann

MAT_SOURCE = '/u/amo-d0/ugrad/wso226/MachineLearning/tools/caffe-sl/examples/mnist_sl/feature/naive_iter_2000/train/ip2_mat'
SL_ROOT = '/u/amo-d0/ugrad/wso226/MachineLearning/tools/caffe-sl/examples/mnist_sl'
PRETRAINED_MODEL = '/u/amo-d0/ugrad/wso226/MachineLearning/tools/caffe-sl/examples/mnist_sl/custom/lenet_naive_iter_2000.caffemodel'
MODEL_FILE = '/u/amo-d0/ugrad/wso226/MachineLearning/tools/caffe-sl/examples/mnist_sl/custom/lenet_naive.prototxt'
DATA_SOURCE = '/u/amo-d0/ugrad/wso226/MachineLearning/tools/caffe-sl/examples/mnist_sl/data/'
IMAGE_SOURCE = '/u/amo-d0/ugrad/wso226/MachineLearning/tools/caffe-sl/examples/mnist_sl/data/t10k/*'

counter = []
x = 0
while x < 60000:
    counter.append(x)
    x += 600

train_set = []
for x in counter:
    file_name = MAT_SOURCE + '/feat_' + str(x) + '.mat'
    temp_set = loadmat(file_name)['feat'].reshape(600, -1)
    for idx in range(len(temp_set)):
        train_set.append(temp_set[idx])
train_set = np.asarray(train_set)

# Training Labels
with open(DATA_SOURCE + 'train.lst', 'r') as lbf:
    training_label = np.array([int(line.split()[1]) for line in lbf])

# Test Images files and labels
test_images = []
with open(DATA_SOURCE + 't10k.lst', 'r') as lbf:
    for line in lbf:
        test_images.append((str(line.split()[0]), int(line.split()[1])))
test_images = sorted(test_images)

# Caffe Network
net = caffe.Net(MODEL_FILE, PRETRAINED_MODEL, caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
#transformer.set_raw_scale('data', 255)
net.blobs['data'].reshape(10000, 3, 28, 28)

transformed_images = np.zeros(net.blobs['data'].data.shape)
for idx, file in enumerate(test_images):
    image = caffe.io.load_image(DATA_SOURCE + 't10k/' + file[0])
    transformed_images[idx] = transformer.preprocess('data', image)

net.blobs['data'].data[...] = transformed_images
output = net.forward()
test_feats = output['ip2']

# KNN
flann = pyflann.FLANN()
result, dists = flann.nn(train_set, test_feats, 1)

pred = []
for nn in result:
    pred.append(training_label[nn])

test_label = []
# Accuracy testing
for file in test_images:
    test_label.append(file[1])
pred = np.asarray(pred)
test_label = np.asarray(test_label)
accy = (pred == test_label).sum() * 100.0 / test_label.shape[0]
print accy
