from svm import *
from svmutil import *
import cPickle

x = cPickle.load(open('x.train','rb'))
y = cPickle.load(open('y.train','rb'))

# -t 2 is RBF kernel
m = svm_train(y[:],x[:],'-t 2')
p_lable, p_acc, p_val = svm_predict(y[5000:],x[5000:],m)
svm_save_model('7classesF.model',m)
