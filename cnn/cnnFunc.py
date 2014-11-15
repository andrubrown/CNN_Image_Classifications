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
from svm import *
from svmutil import *
import cPickle
import time

def classify(img_path):

	input_img = Image.open(img_path)
	car = array(input_img)
	
	all_lables = ['Others','Car','Pickup','SUV']
	x=[]
	y=[0]
	
	time_before_load = time.clock()
	
	pylab.rcParams['figure.figsize'] = (10.0, 10.0)
	net = DecafNet('/home/administrator/Experiments/CNN/imagenet_pretrained/imagenet.decafnet.epoch90', '/home/administrator/Experiments/CNN/imagenet_pretrained/imagenet.decafnet.meta')
	m = svm_load_model('3classes.model')
	
	time_after_load = time.clock()
	
	# Run a classification pass to create all the intermediate features
	scores = net.classify(car, center_only=True)
	feat = net.feature('fc7_cudanet_out')[0]
	x_single = feat.tolist()
	x.append(x_single)
	
	p_lable, p_acc, p_val = svm_predict(y,x,m)
	#print all_lables[int(p_lable[0])]
	time_after_first_predict = time.clock()
	#print "Loading time is " + str(time_after_load - time_before_load)
	#print "First predicting time is " + str(time_after_first_predict - time_after_load)
	return all_lables[int(p_lable[0])]

'''
time.sleep(5) 

# batch test
imageList = glob.glob("/home/administrator/Experiments/CNN/Pickup/*")
for i in imageList:
    x=[]
    y=[0]
    img = Image.open(i)
    car = array(img)
    # Timing begin
    start = time.clock()
    # Run a classification pass to create all the intermediate features
    scores = net.classify(car, center_only=True)
    feat = net.feature('fc7_cudanet_out')[0]
    x_single = feat.tolist()
    x.append(x_single)
    p_lable, p_acc, p_val = svm_predict(y,x,m)
    print all_lables[int(p_lable[0])]
    stop = time.clock()
    print "Testing time is " + str(stop - start)
'''
