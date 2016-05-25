import numpy as np
import caffe
import pyflann

# Setup
n_iter = 54000
test_interval = 600
test_iter = 10

mini_batch_size = 1000
num_mini_batch = 54

mini_batches_images = np.empty((54, 10, 100), dtype=np.object_)
mini_batches_labels = np.empty((54, 10, 100), dtype=np.uint8)

train_loss = np.zeros(n_iter)
test_acc = np.zeros(int(np.ceil(n_iter / test_interval)))
# output = np.zeros(n_iter, 8, 10)

ROOT_DIR = '/u/amo-d0/ugrad/wso226/MachineLearning/tools/caffe-sl/'
SL_DIR = ROOT_DIR + 'examples/mnist_sl/'
DATA_SOURCE = '/u/amo-d0/ugrad/wso226/MachineLearning/tools/caffe-sl/examples/mnist_sl/data/'

TRAIN_LABEL_SOURCE = '/u/amo-d0/ugrad/wso226/MachineLearning/tools/caffe-sl/examples/mnist_sl/data/separated_train/'
TRAIN_IMG_SOURCE = '/u/amo-d0/ugrad/wso226/MachineLearning/tools/caffe-sl/examples/mnist_sl/data/train/'
TEST_LABEL_SOURCE = '/u/amo-d0/ugrad/wso226/MachineLearning/tools/caffe-sl/examples/mnist_sl/data/train.lst'
TEST_IMG_SOURCE = '/u/amo-d0/ugrad/wso226/MachineLearning/tools/caffe-sl/examples/mnist_sl/data/t10k/*'

TEST_NET = '/u/amo-d0/ugrad/wso226/MachineLearning/tools/caffe-sl/examples/mnist_sl/custom/lenet_naive.prototxt'
TRAIN_NET = '/u/amo-d0/ugrad/wso226/MachineLearning/tools/caffe-sl/examples/mnist_sl/lenet_naive_train_test.prototxt'

# Only prefix, attach iteration number after
CAFFE_MODEL = '/u/amo-d0/ugrad/wso226/MachineLearning/tools/caffe-sl/examples/mnist_sl/custom/lenet_naive_iter_'
SOLVER = '/u/amo-d0/ugrad/wso226/MachineLearning/tools/caffe-sl/examples/mnist_sl/custom/lenet_naive_solver_custom.prototxt'

def main():
    generateMiniBatches()


# Tested
def generateMiniBatches():
    print 'Generating all mini batches...'
    for label in range(10):
        with open(TRAIN_LABEL_SOURCE + 'train_' + str(label) + '.lst', 'r') as lbf:
            for batch_num in range(num_mini_batch):
                for sample_num in range(100):
                    line = lbf.readline()
                    mini_batches_images[batch_num][label][sample_num] = line.split()[0]
                    mini_batches_labels[batch_num][label][sample_num] = line.split()[1]
    print "Finished generating batches..."
                
def generateTriplets(new_batch_num, trained_features):
    print 'Began generating triplets...'
    print 'Forwarding mini-batch through trained network...'
    test_net = caffe.Net(TEST_NET, CAFFE_MODEL, caffe.TEST)
    test_net.blobs['data'].reshape(mini_batch_size, 3, 28, 28)
    test_net.blobs['data'].data[...] = transformImages(new_batch_num, test_net)
    output = net.forward()
    new_batch_feats = output['ip2']

    # Set K at run time!!!
    print 'Finding knn... (K = 10)'
    flann = pyflann.FLANN()
    result, dists = flann.nn(trained_features, new_batch_feats, 10)
    
    print 'Generating triplets...'
    for idx, new_train_feats in enumerate(new_batch_feats):
        
    
def extractFeatures(trained_net):
    return trained_net.blobs['ip2'].data
    
def transformImages(new_batch_num, test_net):
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformed_images = np.empty(test_net.blobs['data'].data.shape)
    counter = 0
    for label in mini_batches[new_batch_num]:
        for image in label:
            transformed_images[counter] = caffe.io.load_image(TRAIN_IMG_SOURCE + image)
            counter += 1
    return transformed_images

def solve():
    for it in range(n_iter):
        solver.step(1)

        train_loss[it] = solver.net.blobs['loss'].data

        # Store the output on the first test batch to avoid data reloading
        solver.test_nets[0].forward(start='conv1')
        output[it] = solver.test_nets[0].blobs['ip2'].data[:8]

        # Run full test every test_interval
        if it % test_interval == 0:
            print 'Iteration', it, 'testing...'
            correct = 0
            for test_it in range(test_iter):
                solver.test_nets[0].forward()
            test_acc[it // test_interval] == correct / 1e4
