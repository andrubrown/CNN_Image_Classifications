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

pylab.rcParams['figure.figsize'] = (10.0, 10.0)
net = DecafNet('/home/administrator/Experiments/CNN/imagenet_pretrained/imagenet.decafnet.epoch90', '/home/administrator/Experiments/CNN/imagenet_pretrained/imagenet.decafnet.meta')

lena = Image.open("pickup.png")
lena = array(lena)

# Run a classification pass to create all the intermediate features
scores = net.classify(lena, center_only=True)
print 'blobs:', net._net.blobs.keys()


# Show the input image
image = net.feature('data')[0,::-1]
image -= image.min()
image /= image.max()
_ = visualize.show_single(image)

# Show the first layer filters
filters = net._net.layers['conv1'].param()[0].data()
_ = visualize.show_multiple(filters.T)
pyplot.title('First layer Filters')

# show the first layer output. There are 96 channels,
# but we will just show 32 channels and their corresponding filters.
feat = net.feature('conv1_cudanet_out')[0,::-1, :, ::3]
filters = net._net.layers['conv1'].param()[0].data()
_ = visualize.show_multiple(filters.T[::3])
pyplot.title('Filters')
pyplot.figure()
_ = visualize.show_channels(feat)
pyplot.title('Output')

# show what the second layer filters look like.
# It is a bit complex, since there are 128 filters,
# and each filter is 5*5*48. We show the channels separately,
# and show only the first 48 filters, one row per filter.
filters = net._net.layers['conv2'].param()[0].data()
# make the right filter shape
filters = filters.T.reshape(128, 5, 5, 48)
filters = filters.swapaxes(2,3).swapaxes(1,2).reshape(128*48, 5, 5)
_ = visualize.show_multiple(filters[:48*48], ncols=48)
pyplot.title('Second layer filters, each filter is shown as a row of channels.')

# show what the second layer output look like.
# There are 256 channels, but we are going to only show the first 36 filters.
feat = net.feature('conv2_cudanet_out')[0, ::-1, :]
_ = visualize.show_channels(feat[:, :, :36])

# show what the third layer output look like.
# There are 384 channels
feat = net.feature('conv3_cudanet_out')[0, ::-1, :]
print feat.shape
_ = visualize.show_channels(feat)

# show what the fourth layer output look like.
# There are 384 channels
feat = net.feature('conv4_cudanet_out')[0, ::-1, :]
print feat.shape
_ = visualize.show_channels(feat)


# show what the fifth layer output look like.
# There are 256 channels.
feat = net.feature('conv5_cudanet_out')[0, ::-1, :]
print feat.shape
_ = visualize.show_channels(feat)

# show what the fifth layer output look like after ReLU and pooling.
# There are 256 channels. This is the last layer of
# convolutional neural networks. After this layer, all computations
# are fully connected.
feat = net.feature('pool5_cudanet_out')[0, ::-1, :]
print feat.shape
_ = visualize.show_channels(feat, bg_func=np.max)

# show the feature of the first fully connected layers.
feat = net.feature('fc6_cudanet_out')[0]
print feat.shape
pyplot.plot(feat)
pyplot.title('feature')
pyplot.figure()
_ = pyplot.hist(feat, bins=100)



# show the feature of the second fully connected layers.
feat = net.feature('fc7_cudanet_out')[0]
# save feat
scipy.io.savemat('feat7.mat',mdict={'feat7':feat})


print feat.shape
pyplot.plot(feat)
pyplot.figure()
_ = pyplot.hist(feat, bins=100)

# show the final prediction probability
feat = net.feature('probs_cudanet_out')[0]
print feat.shape
pyplot.plot(feat)

# Now, print the top 5 predictions.
print net.top_k_prediction(scores, 5)[1]

'''
img = Image.open("dog.jpg")
dog = array(img)
scores = net.classify(dog)
print net.top_k_prediction(scores, 5)

img = Image.open("cat.jpg")
cat = array(img)
scores = net.classify(cat)
print net.top_k_prediction(scores, 5)


img = Image.open("people.png")
people = array(img)
scores = net.classify(people)
print net.top_k_prediction(scores, 5)

img = Image.open("face.jpg")
people = array(img)
scores = net.classify(people)
print net.top_k_prediction(scores, 5)

img = Image.open("car.png")
people = array(img)
scores = net.classify(people)
print net.top_k_prediction(scores, 5)

img = Image.open("car1.png")
people = array(img)
scores = net.classify(people)
print net.top_k_prediction(scores, 5)

img = Image.open("car2.png")
people = array(img)
scores = net.classify(people)
print net.top_k_prediction(scores, 5)

img = Image.open("pickup.png")
people = array(img)
scores = net.classify(people)
print net.top_k_prediction(scores, 5)

img = Image.open("pickup1.png")
people = array(img)
scores = net.classify(people)
print net.top_k_prediction(scores, 5)

img = Image.open("pickup2.png")
people = array(img)
scores = net.classify(people)
print net.top_k_prediction(scores, 5)
'''
