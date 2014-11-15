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
from svm import *
from svmutil import *
import cPickle
from random import shuffle

x = []
y = []
pylab.rcParams['figure.figsize'] = (10.0, 10.0)
net = DecafNet('/home/administrator/Experiments/CNN/imagenet_pretrained/imagenet.decafnet.epoch90', '/home/administrator/Experiments/CNN/imagenet_pretrained/imagenet.decafnet.meta')

# car
imageList = glob.glob("/home/administrator/Experiments/kai_data_for_final_model/Car/*")
for i in imageList:
    img = Image.open(i)
    car = array(img)
    # Run a classification pass to create all the intermediate features
    scores = net.classify(car, center_only=True)
    feat = net.feature('fc7_cudanet_out')[0]
    x_single = feat.tolist()
    x_single.append(1)
    x.append(x_single)

# pickup
imageList = glob.glob("/home/administrator/Experiments/kai_data_for_final_model/Pickup/*")
for i in imageList:
    img = Image.open(i)
    car = array(img)
    # Run a classification pass to create all the intermediate features
    scores = net.classify(car, center_only=True)
    feat = net.feature('fc7_cudanet_out')[0]
    x_single = feat.tolist()
    x_single.append(2)
    x.append(x_single)

# suv
imageList = glob.glob("/home/administrator/Experiments/kai_data_for_final_model/SUV/*")
for i in imageList:
    img = Image.open(i)
    car = array(img)
    # Run a classification pass to create all the intermediate features
    scores = net.classify(car, center_only=True)
    feat = net.feature('fc7_cudanet_out')[0]
    x_single = feat.tolist()
    x_single.append(3)
    x.append(x_single)

# van
imageList = glob.glob("/home/administrator/Experiments/kai_data_for_final_model/Van/*")
for i in imageList:
    img = Image.open(i)
    car = array(img)
    # Run a classification pass to create all the intermediate features
    scores = net.classify(car, center_only=True)
    feat = net.feature('fc7_cudanet_out')[0]
    x_single = feat.tolist()
    x_single.append(4)
    x.append(x_single)

# minivan
imageList = glob.glob("/home/administrator/Experiments/kai_data_for_final_model/Minivan/*")
for i in imageList:
    img = Image.open(i)
    car = array(img)
    # Run a classification pass to create all the intermediate features
    scores = net.classify(car, center_only=True)
    feat = net.feature('fc7_cudanet_out')[0]
    x_single = feat.tolist()
    x_single.append(5)
    x.append(x_single)

# bus
imageList = glob.glob("/home/administrator/Experiments/kai_data_for_final_model/Bus/*")
for i in imageList:
    img = Image.open(i)
    car = array(img)
    # Run a classification pass to create all the intermediate features
    scores = net.classify(car, center_only=True)
    feat = net.feature('fc7_cudanet_out')[0]
    x_single = feat.tolist()
    x_single.append(6)
    x.append(x_single)

# motorcycle
imageList = glob.glob("/home/administrator/Experiments/kai_data_for_final_model/Motorcycle/*")
for i in imageList:
    img = Image.open(i)
    car = array(img)
    # Run a classification pass to create all the intermediate features
    scores = net.classify(car, center_only=True)
    feat = net.feature('fc7_cudanet_out')[0]
    x_single = feat.tolist()
    x_single.append(7)
    x.append(x_single)

shuffle(x)
a=[]
b=[]
for xx in x:
    a.append(xx[:-1]) # get the feature vector
    b.append(xx[-1]) # get the label


cPickle.dump(a,open('x.train','wb'))
cPickle.dump(b,open('y.train','wb'))



'''
x = cPickle.load(open('x.train','rb'))
y = cPickle.load(open('y.train','rb'))
m = svm_train(y,x,'-c 4')
p_lable, p_acc, p_val = svm_predict(y,x,m)
'''
