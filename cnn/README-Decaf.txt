%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is a README file for using pre-trained Decaf network
% for image classifications.
% Kai Zhou
% 2014/06/12
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%% Installation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
0, Install python on Linux: sudo apt-get install python3.1

1, Decaf depends on some python packages and C libraries, you should install those first
    1.1 install scipy and numpy: sudo apt-get install python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose
    1.2 install cython: visit cython.org, download the package, run "sudo python setup.py install"

    1.3 install scikit-image, six 1.3 etc

2, cd /decaf-release-master/decaf, run "make" in command line

3, cd /decaf-release-master, run "python setup.py build" and "python setup.py install"

%%%%%%%%%% Test %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Read "Lena-Demo.pdf" for details

%%%%%%%%%% Run new experiment %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
1, Use python scripts(carFeature.py) to get stacked feature representation, in matrix format

2, Change the matrix format into csv, add the first row as attribute titles, and the last column 0 and 1 as classes

3, load into weka, change last column 0 and 1 from numeric to nominal

4, choose SMO parameters, run 10 fold CV, get the accuracy.

Class:
Pickup 0
Car 1
Bus 2
Minivan 3
Van 4
SUV 5
Motorcycle 6
