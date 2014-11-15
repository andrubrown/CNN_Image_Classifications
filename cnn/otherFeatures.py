from PIL import Image
from numpy import array
from decaf.scripts.imagenet import DecafNet
from decaf.scripts import imagenet
from decaf.util import smalldata
from decaf.util import visualize
from matplotlib import pyplot
import numpy as np
import pylab
import numpy, scipy.io
import glob

pylab.rcParams['figure.figsize'] = (10.0, 10.0)
net = DecafNet('/home/administrator/Experiments/CNN/imagenet_pretrained/imagenet.decafnet.epoch90', '/home/administrator/Experiments/CNN/imagenet_pretrained/imagenet.decafnet.meta')

# Bus
my_output = np.array(range(1,4097))
imageList = glob.glob("/home/administrator/Experiments/CNN/Bus/*")
for i in imageList:
    img = Image.open(i)
    car = array(img)
    # Run a classification pass to create all the intermediate features
    scores = net.classify(car, center_only=True)
    feat = net.feature('fc7_cudanet_out')[0]
    my_output = numpy.vstack((my_output, feat))

print my_output.shape
# save feat
scipy.io.savemat('busFeature.mat',mdict={'my_output':my_output})

# Minivan
my_output = np.array(range(1,4097))
imageList = glob.glob("/home/administrator/Experiments/CNN/Minivan/*")
for i in imageList:
    img = Image.open(i)
    car = array(img)
    # Run a classification pass to create all the intermediate features
    scores = net.classify(car, center_only=True)
    feat = net.feature('fc7_cudanet_out')[0]
    my_output = numpy.vstack((my_output, feat))

print my_output.shape
# save feat
scipy.io.savemat('MinivanFeature.mat',mdict={'my_output':my_output})

# Motorcycle
my_output = np.array(range(1,4097))
imageList = glob.glob("/home/administrator/Experiments/CNN/Motorcycle/*")
for i in imageList:
    img = Image.open(i)
    car = array(img)
    # Run a classification pass to create all the intermediate features
    scores = net.classify(car, center_only=True)
    feat = net.feature('fc7_cudanet_out')[0]
    my_output = numpy.vstack((my_output, feat))

print my_output.shape
# save feat
scipy.io.savemat('MotorcycleFeature.mat',mdict={'my_output':my_output})

# SUV
my_output = np.array(range(1,4097))
imageList = glob.glob("/home/administrator/Experiments/CNN/SUV/*")
for i in imageList:
    img = Image.open(i)
    car = array(img)
    # Run a classification pass to create all the intermediate features
    scores = net.classify(car, center_only=True)
    feat = net.feature('fc7_cudanet_out')[0]
    my_output = numpy.vstack((my_output, feat))

print my_output.shape
# save feat
scipy.io.savemat('SUVFeature.mat',mdict={'my_output':my_output})

# Van
my_output = np.array(range(1,4097))
imageList = glob.glob("/home/administrator/Experiments/CNN/Van/*")
for i in imageList:
    img = Image.open(i)
    car = array(img)
    # Run a classification pass to create all the intermediate features
    scores = net.classify(car, center_only=True)
    feat = net.feature('fc7_cudanet_out')[0]
    my_output = numpy.vstack((my_output, feat))

print my_output.shape
# save feat
scipy.io.savemat('VanFeature.mat',mdict={'my_output':my_output})


