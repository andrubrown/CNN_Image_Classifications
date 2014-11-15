% This is a README file for the cuda-convnet experiments for the Vihcle counting project.
% Kai Zhou, 2014/08/01

1. Data preparation
    1.1 resize images to 224-224
    1.2 split into 6 batches, cross validation.
        1200 cars, 1200 pickups, 1200 minivans(augmented), 1200 SUVs(augmented)
        car C label 0, pickup P 1, minivan MV 2, SUV S 3


train my CIFAR format
python convnet.py --data-path=.\storage2\tiny\my-cifar\ --save-path=.\storage2\tmp\my-model --test-range=6 --train-range=1-5 --layer-def=.\example-layers\layers-19pct.cfg --layer-params=.\example-layers\layer-params-19pct.cfg --data-provider=cifar --test-freq=13

train my Car format
python convnet.py --data-path=.\storage2\tiny\my-car-10\ --save-path=.\storage2\tmp\my-model --test-range=6 --train-range=1-5 --layer-def=.\example-layers\layers-mycar-10.cfg --layer-params=.\example-layers\layer-params-mycar-10.cfg --data-provider=car --test-freq=13


make batch
python make_car_batches.py .\cvBatch\1 .\cvBatch\out

train the new CIFAR
python convnet.py --data-path=.\storage2\tiny\cifar-10-batches-py-colmajor\ --save-path=.\storage2\tmp --test-range=6 --train-range=1-5 --layer-def=.\example-layers\layers-19pct.cfg --layer-params=.\example-layers\layer-params-19pct.cfg --data-provider=cifar --test-freq=13

Resume training CIFAR
python convnet.py -f .\storage2\tmp\ConvNet__2014-07-31_10.02.25\ --save-path=.\

Predict CIFAR
python shownet.py -f .\storage2\tmp\ConvNet__2014-07-31_10.02.25\ --show-preds=probs
