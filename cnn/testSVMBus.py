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

my_output = []
pylab.rcParams['figure.figsize'] = (10.0, 10.0)
net = DecafNet('/home/administrator/Experiments/CNN/imagenet_pretrained/imagenet.decafnet.epoch90', '/home/administrator/Experiments/CNN/imagenet_pretrained/imagenet.decafnet.meta')
y = []
imageList = glob.glob("/home/administrator/Experiments/CNN/Bus/*")
for i in imageList:
    img = Image.open(i)
    car = array(img)
    # Run a classification pass to create all the intermediate features
    scores = net.classify(car, center_only=True)
    feat = net.feature('fc7_cudanet_out')[0]
    print type(feat)
    print feat.shape
    x = feat.tolist()
    my_output.append(x)
    y.append(1)

x = my_output
y = [1,2,2,2]
print y
print x[0]


