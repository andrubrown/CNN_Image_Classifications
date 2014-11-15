from svm import *
from svmutil import *
import cPickle

x = cPickle.load(open('x.train','rb'))
y = cPickle.load(open('y.train','rb'))

a=[]
a.append(x[2888])
b=[]
b.append(y[2888])

m = svm_load_model('3classes.model')
p_lable, p_acc, p_val = svm_predict(b,a,m)
#label = libsvm.svm_predict(m,a)
print p_lable
