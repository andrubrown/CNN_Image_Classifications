Use CNN model (CUDA Convnet from Alex and Decaf from Yangqing to accomplish image classification tasks.

1, dog vs cat dataset:
	It's a kaggle competition, to write an algorithm to classify whether images contain either a dog or a cat (http://www.kaggle.com/c/dogs-vs-cats).
	We use decaf pretrained model, get the 6th fully connneted feature, and build our own SVM model to classify the images. Our best performance is 94.4%.
	
2, vehicle counting:
	This is a more challenging project. We need to extract vehicle images form videos, then use those images to create a training dataset and train our own multiclass classifier.
	I have tried CUDA convnet to create a new network architecture. But the training dataset is small, we didn't benefit much from the complex deep network(95% accuracy in 2 hours training).
	Also, I tried the decaf model similar with DogVCat, and get 87% accuracy in 7 different vihicle classes.
	To integrate as a real-time system, I build a classfy function, read images from the output of the video, precess it, label and count it, write the counted vehicles as output.
	

=====

Decaf is a framework that implements convolutional neural networks, with the
goal of being efficient and flexible. It allows one to easily construct a
network in the form of an arbitrary Directed Acyclic Graph (DAG) and to
perform end-to-end training.

For more usage check out [the wiki](https://github.com/UCB-ICSI-Vision-Group/decaf-release/wiki).
A great place to start is running [ImageNet classification on an image](https://github.com/UCB-ICSI-Vision-Group/decaf-release/wiki/imagenet).

For the pre-trained imagenet DeCAF feature and its analysis, please see [technical report on arXiv](http://arxiv.org/abs/1310.1531). 
