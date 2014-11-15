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
import time
import shutil
import os

all_lables =['Others','Car','Pickup','SUV','Van','Minivan','Bus','Motorcycle']

# load decaf model
pylab.rcParams['figure.figsize'] = (10.0, 10.0)

# CHANGE HERE
net = DecafNet('/home/administrator/Experiments/CNN/imagenet_pretrained/imagenet.decafnet.epoch90', '/home/administrator/Experiments/CNN/imagenet_pretrained/imagenet.decafnet.meta')
m = svm_load_model('7classes.model')

# get initial count
count = {'Car':0,'Pickup':0,'SUV':0,'Van':0,'Minivan':0,'Bus':0,'Motorcycle':0}

# keep classifying
isDone = False
while(not isDone):
# CHANGE HERE
    imageList = glob.glob("/home/administrator/Experiments/CNN/libsvm-3.18/python/input/*")
    #print imageList
    for i in imageList:
        # get the extension of the file, txt or png
        flag = i.strip().split('.')[-1]
        if flag == 'txt':
            isDone = True
            continue
        
        x=[]
        y=[0]
        img = Image.open(i)
        car = array(img)
        # Run a classification pass to create all the intermediate features
        scores = net.classify(car, center_only=True)
        feat = net.feature('fc7_cudanet_out')[0]
        x_single = feat.tolist()
        x.append(x_single)
        p_lable, p_acc, p_val = svm_predict(y,x,m)
        predict = all_lables[int(p_lable[0])]
        # add predicted label to the file
        i_new = '.'.join(i.strip().split('.')[:-1]) + '_' + predict + '.' + flag
        os.rename(i, i_new)
        count[predict] += 1
# CHANGE HERE
        shutil.move(i_new,'./processed/')
    time.sleep(3)

# rewrite the count file
# CHANGE HERE
svcount = open('./result/svcount.txt','w')
for item in count:
    svcount.write(item + '\t' + str(count[item]) + '\n')
svcount.close()
svdone = open('./result/svdone.txt','w')
svdone.close()


